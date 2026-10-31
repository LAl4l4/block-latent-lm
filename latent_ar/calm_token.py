"""CALM 历史主干加块内 token AR 头；这是顺序选词对照。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from latent_ar.calm import CalmConfig, CalmModel, shuffle_valid_contexts
from latent_ar.codec import BlockCodec
from latent_ar.token_head import LatentTokenHeadModel, TokenHead


@dataclass
class CalmTokenConfig(CalmConfig):
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


class CalmTokenModel(CalmModel):
    """每块先串行选词，再经 VAE 编码；并非单步并行 latent 生成。"""

    architecture = "latent_calm_token_v1"

    def __init__(self, cfg: CalmTokenConfig):
        super().__init__(cfg)
        self.latent_head = TokenHead(cfg)
        self.latent_head.apply(BlockCodec._initialize)

    # 使用历史实现的逐槽均匀采样与 VAE 均值编码；两者接口相同。
    draw_noise = LatentTokenHeadModel.draw_noise
    sample_latent = LatentTokenHeadModel.sample_latent

    def prior_statistics(
        self,
        prompt,
        prompt_lengths,
        labels,
        shuffle_context=False,
        generator=None,
        **ignored
    ):
        del generator
        contexts, means, _, valid = self.teacher_contexts(
            prompt, prompt_lengths, labels
        )
        if shuffle_context:
            contexts = shuffle_valid_contexts(contexts, valid)
        blocks = F.pad(
            labels,
            (0, means.shape[1] * self.cfg.chunk_size - labels.shape[1]),
            value=-100,
        ).reshape(*valid.shape, self.cfg.chunk_size)
        previous = torch.full_like(blocks, self.cfg.pad_id)
        previous[:, :, 0] = self.cfg.bos_id
        previous[:, :, 1:] = blocks[:, :, :-1].masked_fill(
            blocks[:, :, :-1].eq(-100), self.cfg.pad_id
        )
        logits = self.latent_head(contexts, previous)
        loss = F.cross_entropy(
            logits.float().flatten(0, 2),
            blocks.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        return {
            "text_nll_sum": loss,
            "objective_sum": loss,
            "objective_count": int(labels.ne(-100).sum()),
            "tokens": int(labels.ne(-100).sum()),
            "blocks": int(valid.sum()),
            "scored_blocks": int(valid.sum()),
        }
