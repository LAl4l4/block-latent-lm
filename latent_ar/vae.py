"""带可采样连续表示的块级文本 VAE。"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.codec import BlockCodec
from latent_ar.model import ChunkEncoder, ModelConfig, TextDecoder


@dataclass
class VAEConfig(ModelConfig):
    vae_kl_weight: float = 0.001
    vae_kl_floor: float = 0.5
    vae_input_dropout: float = 0.15
    vae_latent_dropout: float = 0.15

    def __post_init__(self):
        super().__post_init__()
        if self.vae_kl_weight < 0 or self.vae_kl_floor < 0:
            raise ValueError("VAE KL settings must be nonnegative")
        if not 0 <= self.vae_input_dropout < 1 or not 0 <= self.vae_latent_dropout < 1:
            raise ValueError("VAE dropout probabilities must be in [0, 1)")


class BlockVAE(nn.Module):
    """块内双向 decoder 只读取连续 latent，encoder 与输出词表共享权重。"""

    architecture = "block_vae_v1"

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        if cfg.encoder_dim != cfg.decoder_dim:
            raise ValueError("VAE embedding sharing requires encoder_dim=decoder_dim")
        self.cfg = cfg
        self.encoder = ChunkEncoder(cfg)
        self.posterior = nn.Linear(cfg.dim, 2 * cfg.dim)
        self.decoder = TextDecoder(cfg)
        self.decoder.output.weight = self.encoder.embedding.weight
        self.apply(BlockCodec._initialize)
        with torch.no_grad():
            self.encoder.embedding.weight[cfg.pad_id].zero_()

    def encode_posterior(self, ids, lengths, corrupt=False, generator=None):
        """返回均值、对数标准差和有效块掩码；输入掩盖不改变真实长度。"""
        if corrupt and self.training and self.cfg.vae_input_dropout:
            mask = torch.rand(ids.shape, device=ids.device, generator=generator)
            ids = ids.masked_fill(mask < self.cfg.vae_input_dropout, self.cfg.pad_id)
        features, valid = self.encoder(ids, lengths)
        mean, log_std = self.posterior(features).chunk(2, dim=-1)
        # 限制指数运算的数值范围；这是稳定性改编而非论文参数。
        log_std = log_std.clamp(-8.0, 4.0)
        return mean, log_std, valid

    def encode(self, ids, lengths):
        mean, _, valid = self.encode_posterior(ids, lengths)
        return mean, valid

    def decode(self, latents):
        return self.decoder(latents).flatten(1, 2)

    def forward(self, ids, lengths, generator=None):
        mean, log_std, _ = self.encode_posterior(
            ids, lengths, corrupt=True, generator=generator
        )
        noise = torch.randn(
            mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
        )
        latent = mean + log_std.exp() * noise
        if self.training and self.cfg.vae_latent_dropout:
            keep = torch.rand(latent.shape, device=latent.device, generator=generator)
            keep = (keep >= self.cfg.vae_latent_dropout).to(latent.dtype)
            latent = latent * keep / (1 - self.cfg.vae_latent_dropout)
        return self.decode(latent)[:, : ids.shape[1]]

    def reconstruction_statistics(self, ids, lengths, labels, generator=None):
        """CE 与有效块 KL 分别求和；先逐维裁剪 KL，再对维度和块求和。"""
        mean, log_std, valid = self.encode_posterior(
            ids, lengths, corrupt=True, generator=generator
        )
        noise = torch.randn(
            mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
        )
        latent = mean + log_std.exp() * noise
        if self.training and self.cfg.vae_latent_dropout:
            keep = torch.rand(latent.shape, device=latent.device, generator=generator)
            latent = latent * (keep >= self.cfg.vae_latent_dropout).to(latent.dtype)
            latent = latent / (1 - self.cfg.vae_latent_dropout)
        logits = self.decode(latent)[:, : labels.shape[1]]
        token_valid = labels != -100
        ce = F.cross_entropy(
            logits.float().flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        kl = 0.5 * (
            mean.float().square()
            + (2 * log_std.float()).exp()
            - 1
            - 2 * log_std.float()
        )
        kl_sum = kl.clamp_min(self.cfg.vae_kl_floor).sum(-1).masked_select(valid).sum()

        correct = (logits.argmax(-1) == labels) & token_valid
        padding = (-labels.shape[1]) % self.cfg.chunk_size
        token_blocks = F.pad(token_valid, (0, padding)).reshape(-1, self.cfg.chunk_size)
        correct_blocks = F.pad(correct, (0, padding)).reshape(-1, self.cfg.chunk_size)
        present = token_blocks.any(-1)
        complete = (correct_blocks | ~token_blocks).all(-1) & present
        full = token_blocks.all(-1)
        return {
            "loss_sum": ce + self.cfg.vae_kl_weight * kl_sum,
            "ce_sum": ce,
            "kl_sum": kl_sum,
            "tokens": int(token_valid.sum()),
            "blocks": int(valid.sum()),
            "correct_tokens": int(correct.sum()),
            "correct_blocks": int(complete.sum()),
            "full_blocks": int(full.sum()),
            "correct_full_blocks": int((complete & full).sum()),
        }
