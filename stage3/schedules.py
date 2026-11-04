"""阶段 3 的可见关系：在阶段 2 的调度上加块窗口。"""


import torch

from stage2.schedules import visibility


def window_visibility(schedule, prompt, blocks, block_size, stagger=None, segment=1,
                      window=2):
    """在 visibility 上加块窗口：第 t 块只读第 t−window..t 块（prompt 算第 −1 块）。

    schedule 为 "zonly" 时生成位置一个词都不读（解码器只靠 z，等于 CALM 的块内并行解码）。
    窗口只限制每层读哪些位置，多层叠加后感受野可以更远，但读到的都是已生成的词，不会泄漏。
    """
    if window < 0:
        raise ValueError("window must be nonnegative")
    base = "causal" if schedule == "zonly" else schedule
    allowed = visibility(base, prompt, blocks, block_size, stagger, segment)
    n = prompt + blocks * block_size
    position = torch.arange(n)
    generated = position >= prompt
    block = torch.where(generated, (position - prompt) // block_size, -1)
    near = block[:, None] - block[None, :] <= window
    if schedule == "zonly":
        near = ~generated[:, None]
    return allowed & torch.where(generated[:, None], near, True)
