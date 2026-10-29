"""训练退出控制：把第一次 Ctrl+C 延迟到本次参数更新完成后处理。"""

import signal


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
