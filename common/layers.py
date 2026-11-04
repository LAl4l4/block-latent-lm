"""Transformer 组件（原在阶段 1 的 latent_ar/model.py，行为不变）。"""


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
