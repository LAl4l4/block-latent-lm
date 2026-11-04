"""seam 各实验共用的小工具：Transformer 组件、混合精度、日志与安全中断。

这些函数原在阶段 1 的块级潜变量系统里（latent_ar/model.py、latent_ar/control.py、train.py），
那部分代码删除后搬到这里，行为不变；原代码见 git tag stage1-latent-blocks。
"""

from contextlib import nullcontext
import json
from pathlib import Path
import signal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """在最后一个维度上归一化；平方和在 FP32 中计算以提高数值稳定性。"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """门控前馈网络：一半投影作为门，另一半提供被调节的特征。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


def precision_context(device: torch.device, precision: str):
    """返回混合精度上下文的工厂函数；CPU 始终使用 FP32。"""
    if device.type != "cuda" or precision == "fp32":
        return nullcontext

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bf16; use fp16 or fp32")
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown precision: {precision}")

    return lambda: torch.autocast("cuda", dtype=dtype)


def append_record(path: Path, record: dict):
    """追加一行 JSONL，方便后续画曲线。"""
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


class GracefulStop:
    """第一次中断请求保存后退出；第二次中断允许立即退出。

    不能在梯度累计到一半时，把模型、优化器和随机状态当成完整一步保存。
    因此第一次 Ctrl+C 只设置标志，由训练循环在安全边界执行保存。
    """

    def __init__(self):
        self.requested = False
        self._previous_handlers = {}

    def _request_stop(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt

        self.requested = True
        print("\n[中断] 将完成当前更新、保存检查点，然后退出。", flush=True)

    def __enter__(self):
        signals = [signal.SIGINT]

        # Windows 的 Ctrl+Break 也按相同的安全退出方式处理。
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)

        for signum in signals:
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._request_stop)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)

        return False
