"""使用可训练文本历史压缩和后验能量目标的 CALM 式主干。"""

# 生成头结构及评分目标参考 CALM 官方实现：
# https://github.com/shaochenze/calm/blob/main/models/modeling_energy.py

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from latent_ar.codec import BlockCodec
from latent_ar.model import LatentTransformer
from latent_ar.vae import BlockVAE, VAEConfig


@dataclass
class CalmConfig(VAEConfig):
    noise_dim: int = 64
    energy_candidates: int = 8
    target_samples: int = 100
    energy_layers: int = 4
    energy_batch_blocks: int = 32

    def __post_init__(self):
        super().__post_init__()
        if (
            min(
                self.noise_dim,
                self.energy_layers,
                self.target_samples,
                self.energy_batch_blocks,
            )
            < 1
        ):
            raise ValueError(
                "CALM energy dimensions and sample counts must be positive"
            )
        if self.energy_candidates < 2:
            raise ValueError("Energy score requires at least two candidates")


class ConditionalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        self.down = nn.Linear(dim, dim)

    def forward(self, x, context):
        context = context.expand_as(x)
        gate, value = self.ff(torch.cat((self.norm(x), context), dim=-1)).chunk(
            2, dim=-1
        )
        return x + self.down(F.silu(gate) * value)


class CalmHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.noise = nn.Linear(cfg.noise_dim, cfg.dim)
        self.context = nn.Linear(cfg.dim, cfg.dim)
        self.noise_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.context_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            ConditionalBlock(cfg.dim) for _ in range(cfg.energy_layers)
        )
        self.final_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.final_hidden = nn.Linear(cfg.dim, cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, context, noise):
        x = self.noise_norm(self.noise(noise))
        context = self.context_norm(self.context(context))
        for block in self.blocks:
            x = block(x, context)
        return self.output(F.silu(self.final_hidden(self.final_norm(x))))


def posterior_energy_statistics(samples, targets):
    """输入均为 [块数, 样本数, 维度]；排斥项平均所有非对角有序配对。"""
    if (
        samples.ndim != 3
        or targets.ndim != 3
        or samples.shape[0] != targets.shape[0]
        or samples.shape[-1] != targets.shape[-1]
        or samples.shape[1] < 2
        or targets.shape[1] < 1
    ):
        raise ValueError("Invalid candidate or posterior sample shapes")
    x, y = samples.float(), targets.float()
    n = x.shape[1]
    attraction = torch.cdist(
        x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    ).mean((1, 2))
    spread = torch.cdist(x, x, p=2, compute_mode="donot_use_mm_for_euclid_dist").sum(
        (1, 2)
    ) / (n * (n - 1))
    return (2 * attraction - spread).sum(), attraction.sum(), spread.sum()


def shuffle_valid_contexts(contexts, valid):
    """仅在相同目标块位置的有效样本间错配，避免把padding状态作为历史。"""
    shuffled = contexts.clone()
    for position in range(valid.shape[1]):
        rows = valid[:, position].nonzero(as_tuple=True)[0]
        if len(rows) > 1:
            shuffled[rows, position] = contexts[rows.roll(1), position]
    return shuffled


class CalmModel(nn.Module):
    architecture = "latent_calm_v1"

    def __init__(self, cfg: CalmConfig):
        super().__init__()
        self.cfg = cfg
        self.codec = BlockVAE(cfg)
        self.backbone = LatentTransformer(cfg)
        self.history_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.history_compressor = nn.Sequential(
            nn.Linear(cfg.chunk_size * (cfg.dim + 1), 2 * cfg.dim),
            nn.SiLU(),
            nn.Linear(2 * cfg.dim, cfg.dim),
            nn.LayerNorm(cfg.dim, eps=1e-6),
        )
        self.latent_head = CalmHead(cfg)
        self.backbone.apply(BlockCodec._initialize)
        self.history_embedding.apply(BlockCodec._initialize)
        self.history_compressor.apply(BlockCodec._initialize)
        self.latent_head.apply(BlockCodec._initialize)
        nn.init.zeros_(self.latent_head.output.weight)
        nn.init.zeros_(self.latent_head.output.bias)
        self.codec.requires_grad_(False)
        self.codec.eval()

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def encode_history(self, ids, lengths):
        batch, width = ids.shape
        if bool((lengths < 1).any()) or bool((lengths > width).any()):
            raise ValueError("Each history length must be within the input width")
        mask = torch.arange(width, device=ids.device)[None, :] < lengths[:, None]
        embedded = self.history_embedding(ids) * mask.unsqueeze(-1)
        padding = (-width) % self.cfg.chunk_size
        slots = F.pad(mask, (0, padding)).reshape(batch, -1, self.cfg.chunk_size)
        embedded = F.pad(embedded, (0, 0, 0, padding)).reshape(
            batch, slots.shape[1], -1
        )
        features = torch.cat((embedded, slots.to(embedded.dtype)), dim=-1)
        valid = slots.any(-1)
        return self.history_compressor(features) * valid.unsqueeze(-1), valid

    def teacher_contexts(self, prompt, prompt_lengths, labels):
        lengths = labels.ne(-100).sum(-1)
        if bool((lengths < 1).any()):
            raise ValueError("Every target must contain at least one token")
        ids = labels.masked_fill(labels.eq(-100), self.cfg.pad_id)
        with torch.no_grad():
            mean, log_std, target_valid = self.codec.encode_posterior(ids, lengths)
        prompt_encoded, prompt_valid = self.encode_history(prompt, prompt_lengths)
        target_encoded, _ = self.encode_history(ids, lengths)
        batch, prompt_width, _ = prompt_encoded.shape
        steps = mean.shape[1]
        prompt_blocks = prompt_valid.sum(-1)
        history = torch.cat((prompt_encoded, target_encoded[:, :-1]), 1)
        valid = torch.cat((prompt_valid, target_valid[:, :-1]), 1)
        positions = torch.cat(
            (
                torch.arange(prompt_width, device=prompt.device).expand(batch, -1),
                prompt_blocks[:, None] + torch.arange(steps - 1, device=prompt.device),
            ),
            1,
        )
        hidden, _ = self.backbone(history, positions, valid)
        indices = torch.cat(
            (
                (prompt_blocks - 1)[:, None],
                torch.arange(
                    prompt_width, prompt_width + steps - 1, device=prompt.device
                ).expand(batch, -1),
            ),
            1,
        )
        contexts = hidden.gather(1, indices[..., None].expand(-1, -1, self.cfg.dim))
        return contexts, mean, log_std, target_valid

    def draw_noise(self, shape, device, generator=None):
        return (
            torch.rand((*shape, self.cfg.noise_dim), device=device, generator=generator)
            - 0.5
        )

    def sample_latent(self, context, generator=None, noise=None):
        if noise is None:
            noise = self.draw_noise(context.shape[:-1], context.device, generator)
        return self.latent_head(context, noise)

    def prior_statistics(
        self,
        prompt,
        prompt_lengths,
        labels,
        shuffle_context=False,
        generator=None,
        **ignored
    ):
        contexts, mean, log_std, valid = self.teacher_contexts(
            prompt, prompt_lengths, labels
        )
        if shuffle_context:
            contexts = shuffle_valid_contexts(contexts, valid)
        contexts, mean, log_std = contexts[valid], mean[valid], log_std[valid]
        sums = None
        for start in range(0, contexts.shape[0], self.cfg.energy_batch_blocks):
            h = contexts[start : start + self.cfg.energy_batch_blocks]
            mu = mean[start : start + self.cfg.energy_batch_blocks].float()
            log_sigma = log_std[start : start + self.cfg.energy_batch_blocks].float()
            noise = self.draw_noise(
                (h.shape[0], self.cfg.energy_candidates), h.device, generator
            )
            eps = torch.randn(
                (h.shape[0], self.cfg.target_samples, self.cfg.dim),
                device=h.device,
                generator=generator,
            )
            targets = mu[:, None, :] + log_sigma.exp()[:, None, :] * eps

            def score_chunk(context, noise_values, posterior_samples):
                predictions = self.latent_head(context[:, None, :], noise_values)
                return posterior_energy_statistics(predictions, posterior_samples)

            if torch.is_grad_enabled():
                values = checkpoint(
                    score_chunk,
                    h,
                    noise,
                    targets,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                values = score_chunk(h, noise, targets)
            sums = (
                values if sums is None else tuple(a + b for a, b in zip(sums, values))
            )
        return {
            "energy_sum": sums[0],
            "attraction_sum": sums[1],
            "spread_sum": sums[2],
            "objective_sum": sums[0],
            "objective_count": int(valid.sum()),
            "blocks": int(valid.sum()),
            "tokens": int(labels.ne(-100).sum()),
        }

    def generation_generators(self, generator, device):
        return generator, None

    def generate(self, *args, **kwargs):
        from latent_ar.feedback import generate_with_feedback

        if kwargs.get("feedback_mode", "text") != "text":
            raise ValueError("CALM requires text history compression for feedback")
        return generate_with_feedback(self, *args, **kwargs)
