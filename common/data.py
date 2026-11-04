"""在拼接后的故事 token 流上取定长窗口。"""


import numpy as np
import torch


class Windows:
    """在拼接后的故事 token 流上取定长窗口；故事之间已有 EOS 分隔。"""

    def __init__(self, path, length):
        self.tokens = np.memmap(path, mode="r", dtype="<u4")
        self.length = length
        if len(self.tokens) <= length:
            raise ValueError("Token stream shorter than one window")

    def batch(self, starts):
        return torch.from_numpy(
            np.stack([self.tokens[s : s + self.length] for s in starts]).astype(np.int64)
        )

    def random_starts(self, rng, count):
        return [rng.randrange(len(self.tokens) - self.length) for _ in range(count)]
