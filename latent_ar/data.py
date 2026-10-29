"""读取准备好的故事 token，并构造 prompt / 续写训练窗口。"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch


def crop_story(
    tokens, rng: random.Random, min_prompt: int, max_prompt: int, max_target: int
):
    """随机选窗口与 prompt 边界，target 最长 max_target，允许自然结束。"""
    if not 1 <= min_prompt <= max_prompt or max_target < 1:
        raise ValueError("Invalid prompt/target lengths")
    if len(tokens) < 2:
        raise ValueError("A story needs at least one content token plus EOS")
    upper = min(max_prompt, len(tokens) - 1)
    prompt_length = rng.randint(min(min_prompt, upper), upper)
    start = rng.randint(0, len(tokens) - 1 - prompt_length)
    boundary = start + prompt_length
    # prompt 与 target 的边界先确定，之后分块时也不能从 target 借 token。
    # 原始故事末尾已有 EOS；窗口截断不会人为追加新的 EOS。
    prompt = np.asarray(tokens[start:boundary], dtype=np.int64).copy()
    target = np.asarray(tokens[boundary : boundary + max_target], dtype=np.int64).copy()
    return prompt, target


def collate(examples, pad_id: int):
    """只构造 prompt 与 target，不生成文本历史或 teacher-forcing 输入。

    labels 的补齐位置为 -100，只用于计算损失，不传进模型。
    """
    batch = len(examples)
    lengths = torch.tensor([len(p) for p, _ in examples], dtype=torch.long)
    max_prompt = int(lengths.max())
    max_target = max(len(y) for _, y in examples)
    prompt = torch.full((batch, max_prompt), pad_id, dtype=torch.long)
    labels = torch.full((batch, max_target), -100, dtype=torch.long)
    for index, (p, y) in enumerate(examples):
        prompt[index, : len(p)] = torch.as_tensor(p)
        labels[index, : len(y)] = torch.as_tensor(y)
    return {
        "prompt": prompt,
        "prompt_lengths": lengths,
        "labels": labels,
    }


class StoryDataset:
    """用内存映射读取 token，避免每次把全部故事加载进内存。"""

    def __init__(
        self,
        directory: Path,
        split: str,
        min_prompt: int,
        max_prompt: int,
        max_target: int,
        pad_id: int,
        limit: int | None = None,
    ):
        self.tokens = np.memmap(directory / f"{split}.bin", mode="r", dtype="<u4")
        self.offsets = np.load(directory / f"{split}.offsets.npy", mmap_mode="r")
        self.size = len(self.offsets) - 1
        if limit is not None:
            if limit < 1:
                raise ValueError("Dataset limit must be positive")
            self.size = min(self.size, limit)
        if not self.size or self.offsets[-1] != len(self.tokens):
            raise ValueError("Invalid prepared dataset")
        self.window = (min_prompt, max_prompt, max_target)
        self.pad_id = pad_id

    def example(self, index: int, rng: random.Random):
        story = self.tokens[self.offsets[index] : self.offsets[index + 1]]
        return crop_story(story, rng, *self.window)

    def batch(self, rng: random.Random, batch_size: int):
        """有放回随机采样故事，每次重新截窗口；按 step 而不是 epoch 训练。"""
        examples = [
            self.example(rng.randrange(self.size), rng) for _ in range(batch_size)
        ]
        return collate(examples, self.pad_id)

    def evaluation_batches(self, batch_size: int, max_batches: int | None, seed: int):
        """每篇故事使用固定窗口，便于不同训练步骤之间比较验证指标。"""
        count = (
            self.size
            if max_batches is None
            else min(self.size, batch_size * max_batches)
        )
        for start in range(0, count, batch_size):
            examples = [
                self.example(i, random.Random(seed + i))
                for i in range(start, min(start + batch_size, count))
            ]
            yield collate(examples, self.pad_id)
