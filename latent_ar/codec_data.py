"""codec 重建数据：真实故事中配置长度的块，含自然 EOS 和短尾块。"""

import torch

from latent_ar.data import StoryDataset


class CodecDataset:
    def __init__(self, directory, split, chunk_size, pad_id, limit=None):
        self.corpus = StoryDataset(directory, split, 1, 1, chunk_size, pad_id, limit)
        self.size = self.corpus.size
        self.chunk_size = chunk_size
        self.pad_id = pad_id

    def batch(self, rng, batch_size):
        ids = torch.full((batch_size, self.chunk_size), self.pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        lengths = torch.empty(batch_size, dtype=torch.long)
        for row in range(batch_size):
            index = rng.randrange(self.size)
            begin, end = self.corpus.offsets[index : index + 2]
            story = self.corpus.tokens[begin:end]
            start = rng.randrange(len(story))
            # 额外包含短块，防止只会还原完整块、不认识 prompt 尾块。
            requested = (
                rng.randint(1, self.chunk_size)
                if rng.random() < 0.25
                else self.chunk_size
            )
            block = story[start : start + requested].astype("int64")
            length = len(block)
            ids[row, :length] = torch.from_numpy(block)
            labels[row, :length] = torch.from_numpy(block)
            lengths[row] = length
        return {"ids": ids, "lengths": lengths, "labels": labels}
