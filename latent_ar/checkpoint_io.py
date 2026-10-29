"""完整检查点的发布：Windows 临时读锁重试，持续锁定则保留版本文件。"""

from pathlib import Path
import time


def publish_checkpoint(temporary, destination, step, attempts=8):
    temporary, destination = Path(temporary), Path(destination)
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            temporary.replace(destination)
            return destination
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(min(.05 * 2**attempt, .5))
    # 不删除仍然有效的 last.pt；新版本使用独立路径，避免因读锁中断训练。
    fallback = destination.with_name(f"{destination.stem}-step{step}{destination.suffix}")
    if fallback.exists():
        fallback = fallback.with_name(f"{fallback.stem}-{time.time_ns()}{fallback.suffix}")
    temporary.replace(fallback)
    print(f"[保存] {destination} 被占用，完整检查点另存为 {fallback}", flush=True)
    return fallback
