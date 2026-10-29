"""文字结构对照：先在小输出头内选定整块，再编码成实际回灌 latent。

不是纯连续向量采样。decoder 继续只翻译，主干不会等待 decoder 的文字。
小头内部四次分类采样取代了对高维连续分布的拟合。
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.codec import BlockCodec
from latent_ar.model import RMSNorm
from latent_ar.staged import LatentPriorModel, StagedModelConfig


@dataclass
class TokenHeadConfig(StagedModelConfig):
    token_head_dim: int = 256
    token_temperature: float = 0.8
    token_top_k: int = 50

    def __post_init__(self):
        super().__post_init__()
        if (
            self.token_head_dim < 1
            or self.token_temperature < 0
            or self.token_top_k < 0
        ):
            raise ValueError("Invalid token-head settings")


class TokenHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.token_head_dim)
        self.condition = nn.Linear(cfg.dim, cfg.token_head_dim)
        self.rnn = nn.GRU(cfg.token_head_dim, cfg.token_head_dim, batch_first=True)
        self.norm = RMSNorm(cfg.token_head_dim)
        self.output = nn.Linear(cfg.token_head_dim, cfg.vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, context, previous):
        shape = context.shape[:-1]
        hidden = (
            self.condition(context).tanh().reshape(1, -1, self.condition.out_features)
        )
        inputs = self.embedding(previous.reshape(-1, previous.shape[-1]))
        states, _ = self.rnn(inputs, hidden)
        return self.output(self.norm(states)).reshape(*shape, previous.shape[-1], -1)


class LatentTokenHeadModel(LatentPriorModel):
    architecture = "latent_codec_token_head_v6"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.latent_head = TokenHead(cfg)
        self.latent_head.apply(BlockCodec._initialize)

    def generation_generators(self, generator, device):
        seeds = torch.randint(
            0, 2**62, (2,), device=device, generator=generator
        ).tolist()
        return tuple(torch.Generator(device=device).manual_seed(seed) for seed in seeds)

    def draw_noise(self, shape, device, generator=None):
        return torch.rand(
            (*shape, self.cfg.chunk_size), device=device, generator=generator
        )

    def sample_latent(self, context, generator=None, noise=None):
        shape = context.shape[:-1]
        flat = context.reshape(-1, self.cfg.dim)
        if noise is None:
            noise = self.draw_noise(shape, context.device, generator)
        if noise.shape != (*shape, self.cfg.chunk_size):
            raise ValueError("Token head requires one uniform input per slot")
        noise = noise.reshape(len(flat), self.cfg.chunk_size)
        state = self.latent_head.condition(flat).tanh()[None]
        previous = torch.full((len(flat), 1), self.cfg.bos_id, device=context.device)
        chosen = []
        for slot in range(self.cfg.chunk_size):
            hidden, state = self.latent_head.rnn(
                self.latent_head.embedding(previous), state
            )
            scores = self.latent_head.output(
                self.latent_head.norm(hidden[:, 0])
            ).float()
            scores[:, self.cfg.pad_id] = -torch.inf
            scores[:, self.cfg.bos_id] = -torch.inf
            if self.cfg.token_temperature == 0:
                token = scores.argmax(-1)
            else:
                scores = scores / self.cfg.token_temperature
                if self.cfg.token_top_k:
                    threshold = scores.topk(
                        min(self.cfg.token_top_k, scores.shape[-1]), dim=-1
                    ).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -torch.inf)
                cumulative = scores.softmax(-1).cumsum(-1)
                token = (
                    (noise[:, slot, None] >= cumulative)
                    .sum(-1)
                    .clamp_max(scores.shape[-1] - 1)
                )
            chosen.append(token)
            previous = token[:, None]
        ids = torch.stack(chosen, -1)
        # 采样得到的具体代码经 encoder 变成 latent，回灌与真实历史使用同一接口。
        latent, _ = self.codec.encode(
            ids, torch.full((len(ids),), self.cfg.chunk_size, device=ids.device)
        )
        return latent[:, 0].reshape(*shape, self.cfg.dim)

    def prior_statistics(
        self, prompt, prompt_lengths, labels, shuffle_context=False, generator=None
    ):
        contexts, targets, valid = self.teacher_contexts(prompt, prompt_lengths, labels)
        if shuffle_context:
            contexts = contexts.roll(1, 0)
        blocks = F.pad(
            labels,
            (0, targets.shape[1] * self.cfg.chunk_size - labels.shape[1]),
            value=-100,
        ).reshape(*valid.shape, self.cfg.chunk_size)
        inputs = torch.full_like(blocks, self.cfg.pad_id)
        inputs[:, :, 0] = self.cfg.bos_id
        inputs[:, :, 1:] = blocks[:, :, :-1].masked_fill(
            blocks[:, :, :-1] == -100, self.cfg.pad_id
        )
        logits = self.latent_head(contexts, inputs)
        loss = F.cross_entropy(
            logits.float().flatten(0, 2),
            blocks.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        return {
            "text_nll_sum": loss,
            "tokens": int((labels != -100).sum()),
            "blocks": int(valid.sum()),
            "scored_blocks": int(valid.sum()),
            "objective_sum": loss,
        }
