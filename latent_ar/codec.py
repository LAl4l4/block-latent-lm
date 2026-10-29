"""阶段一：四 token 块的 encoder / decoder 重建，不训练语言主干。"""

import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.model import ChunkEncoder, ModelConfig, TextDecoder


class BlockCodec(nn.Module):
    """真实 token 块 → 连续 latent → 块内双向还原。

    decoder 仍只能看到 latent。共享输入词表和输出词表的权重，减少重复
    参数；这不会建立绕过 latent 的信息通路。
    """

    architecture = "block_codec_v1"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.encoder_dim != cfg.decoder_dim:
            raise ValueError("Codec embedding sharing requires encoder_dim=decoder_dim")
        self.cfg = cfg
        self.encoder = ChunkEncoder(cfg)
        self.decoder = TextDecoder(cfg)
        self.decoder.output.weight = self.encoder.embedding.weight
        self.apply(self._initialize)
        with torch.no_grad():
            self.encoder.embedding.weight[cfg.pad_id].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, ids, lengths):
        return self.encoder(ids, lengths)

    def decode(self, latents):
        return self.decoder(latents).flatten(1, 2)

    def forward(self, ids, lengths):
        latents, _ = self.encode(ids, lengths)
        return self.decode(latents)[:, : ids.shape[1]]

    @torch.no_grad()
    def encode_labels(self, labels):
        """把带 -100 padding 的目标转换成真实 latent；padding 不参与编码。"""
        lengths = (labels != -100).sum(-1)
        if bool((lengths < 1).any()):
            raise ValueError("Every target must contain at least one token")
        ids = labels.masked_fill(labels == -100, self.cfg.pad_id)
        return self.encode(ids, lengths)


def reconstruction_statistics(logits, labels):
    """返回可累加的重建损失、token 正确数及完整块正确数。"""
    valid = labels != -100
    predictions = logits.argmax(-1)
    correct = (predictions == labels) & valid
    complete = (correct | ~valid).all(-1)
    full_blocks = valid.all(-1)
    loss = F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
        reduction="sum",
    )
    return {
        "loss_sum": loss,
        "tokens": int(valid.sum()),
        "correct_tokens": int(correct.sum()),
        "blocks": labels.shape[0],
        "correct_blocks": int(complete.sum()),
        "full_blocks": int(full_blocks.sum()),
        "correct_full_blocks": int((complete & full_blocks).sum()),
    }
