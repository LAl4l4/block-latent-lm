"""单步连续分布生成头，以及按有效块计算的 Energy loss。

同一个上下文配不同噪声，产生下一块 latent 的不同候选。使用普通欧氏
距离（按 sqrt(dim) 缩放），不能改成平方距离，否则会丢失分布评分性质。
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from latent_ar.codec import BlockCodec
from latent_ar.model import RMSNorm, SwiGLU
from latent_ar.staged import LatentPriorModel, StagedModelConfig


@dataclass
class EnergyModelConfig(StagedModelConfig):
    noise_dim: int = 128
    energy_hidden_dim: int = 512
    energy_layers: int = 2
    energy_candidates: int = 8
    objective: str = "energy"
    cycle_weight: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if min(self.noise_dim, self.energy_hidden_dim, self.energy_layers) < 1:
            raise ValueError("Energy head dimensions and layers must be positive")
        if self.energy_candidates < 2:
            raise ValueError("Energy loss requires at least two independent candidates")
        if self.objective not in {"energy", "token_ce", "block_mixture"}:
            raise ValueError("Unknown latent prior objective")
        if not 0 <= self.cycle_weight < float("inf"):
            raise ValueError("cycle_weight must be nonnegative and finite")


class EnergyBlock(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.condition = nn.Linear(context_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        self.ff = SwiGLU(hidden_dim, hidden_dim)

    def forward(self, state, context):
        return state + self.ff(self.norm(state + self.condition(context)))


class EnergyHead(nn.Module):
    """噪声提供随机选择，上下文决定哪些选择合理；一次前向生成一个向量。"""

    def __init__(self, cfg):
        super().__init__()
        self.noise_dim = cfg.noise_dim
        self.input = nn.Linear(cfg.noise_dim, cfg.energy_hidden_dim)
        self.blocks = nn.ModuleList(
            [
                EnergyBlock(cfg.dim, cfg.energy_hidden_dim)
                for _ in range(cfg.energy_layers)
            ]
        )
        self.norm = RMSNorm(cfg.energy_hidden_dim)
        self.output = nn.Linear(cfg.energy_hidden_dim, cfg.dim)

    def forward(self, context, noise):
        if noise.shape[-1] != self.noise_dim:
            raise ValueError("Incorrect Energy head noise dimension")
        state = self.input(noise)
        for block in self.blocks:
            state = block(state, context)
        return self.output(self.norm(state))


def energy_statistics(samples, target, valid):
    """samples: [B,T,N,D]，target: [B,T,D]，valid: [B,T]。

    无偏估计 E||sample-target|| - 0.5 E||sample-sample'||。
    第二项只平均不同候选的配对，不能把对角线当成独立样本。
    """
    if samples.ndim != 4 or samples.shape[:2] != target.shape[:2]:
        raise ValueError("Expected candidate samples [batch, blocks, candidates, dim]")
    count, dim = samples.shape[-2:]
    if (
        count < 2
        or target.shape != samples.shape[:2] + (dim,)
        or valid.shape != target.shape[:2]
    ):
        raise ValueError("Invalid Energy loss shapes or candidate count")
    if not bool(valid.any()):
        raise ValueError("Energy loss needs at least one valid block")

    # 先去掉 padding，并在 FP32 计算距离，保留候选到生成头/主干的梯度。
    values = samples[valid].float()
    targets = target[valid].float()
    scale = math.sqrt(dim)
    attraction = (
        torch.linalg.vector_norm(values - targets[:, None], dim=-1).mean(-1) / scale
    )
    distances = torch.cdist(
        values, values, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    )
    spread = distances.sum((-1, -2)) / (count * (count - 1) * scale)
    return {
        "energy_sum": (attraction - 0.5 * spread).sum(),
        "attraction_sum": attraction.sum(),
        "spread_sum": spread.sum(),
        "blocks": int(valid.sum()),
    }


def decoder_mixture_statistics(target_log_probs, valid_tokens, chunk_size):
    """候选维先平均概率，再取负对数；块内位置共享同一个 latent 候选。

    输入 [B,L,N] 是每个候选 decoder 对真实 token 的 log probability。
    有限 N 的 NLL 估计有偏，且对应 softmax decoder，不是 argmax 采样似然。
    """
    values = target_log_probs.float().masked_fill(~valid_tokens[..., None], 0)
    candidates = values.shape[-1]
    token_nll = math.log(candidates) - values.logsumexp(-1)
    padding = (-values.shape[1]) % chunk_size
    blocks = (
        F.pad(values, (0, 0, 0, padding))
        .reshape(values.shape[0], -1, chunk_size, candidates)
        .sum(2)
    )
    block_valid = (
        F.pad(valid_tokens, (0, padding))
        .reshape(values.shape[0], -1, chunk_size)
        .any(-1)
    )
    block_nll = math.log(candidates) - blocks.logsumexp(-1)
    return {
        "mixture_token_ce_sum": token_nll[valid_tokens].sum(),
        "mixture_block_nll_sum": block_nll[block_valid].sum(),
    }


def decoder_target_log_probs(model, samples, labels, return_decoded=False):
    """逐候选得到目标文字的 log 概率，训练时重算 decoder 以控制显存。

    不同时保存 N 份完整词表激活；反向时自动汇总候选梯度，再传回主干。
    """

    def decode_one(candidate):
        logits = model.decode(model.denormalize(candidate), labels.shape[1])
        logp = (
            logits.float()
            .log_softmax(-1)
            .gather(-1, labels.clamp_min(0)[..., None])
            .squeeze(-1)
        )
        return (logp, logits.argmax(-1)) if return_decoded else logp

    results = []
    decoded = []
    for index in range(samples.shape[2]):
        candidate = samples[:, :, index]
        if torch.is_grad_enabled() and candidate.requires_grad:
            result = checkpoint(
                decode_one, candidate, use_reentrant=False, preserve_rng_state=False
            )
        else:
            result = decode_one(candidate)
        if return_decoded:
            result, ids = result
            decoded.append(ids)
        results.append(result)
    values = torch.stack(results, dim=-1).masked_fill(labels.eq(-100)[..., None], 0)
    return (values, torch.stack(decoded, -1)) if return_decoded else values


class LatentEnergyModel(LatentPriorModel):
    """复用真实历史因果训练；替换输出头，推理仍只回灌自己生成的 latent。"""

    architecture = "latent_codec_energy_v4"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.latent_head = EnergyHead(cfg)
        self.latent_head.apply(BlockCodec._initialize)

    def draw_noise(self, shape, device, generator=None):
        return torch.randn(
            (*shape, self.cfg.noise_dim), device=device, generator=generator
        )

    def sample_latent(self, context, generator=None, noise=None):
        if noise is None:
            noise = self.draw_noise(context.shape[:-1], context.device, generator)
        return self.denormalize(self.latent_head(context, noise))

    def generation_generators(self, generator, device):
        # 拆开 latent 与选词的 RNG，使词采样温度不改变同一种子的 latent。
        seeds = torch.randint(
            0, 2**62, (2,), device=device, generator=generator
        ).tolist()
        return tuple(torch.Generator(device=device).manual_seed(seed) for seed in seeds)

    def prior_statistics(
        self,
        prompt,
        prompt_lengths,
        labels,
        shuffle_context=False,
        generator=None,
        distribution_diagnostics=False,
    ):
        contexts, targets, valid = self.teacher_contexts(prompt, prompt_lengths, labels)
        if shuffle_context:
            contexts = contexts.roll(1, dims=0)
        noise = self.draw_noise(
            (*contexts.shape[:2], self.cfg.energy_candidates),
            contexts.device,
            generator,
        )
        # 主干只算一次，N 个候选仅重复小生成头；广播上下文不重跑 Transformer。
        samples = self.latent_head(contexts[:, :, None], noise)
        if self.cfg.objective == "energy":
            stats = energy_statistics(samples, self.normalize(targets), valid)
            stats["objective_sum"] = stats["energy_sum"]
            stats["objective_count"] = stats["blocks"]
        else:
            # 分布的 Energy 作为诊断；实际更新由所选的文字目标决定。
            with torch.no_grad():
                stats = energy_statistics(samples, self.normalize(targets), valid)
            result = decoder_target_log_probs(
                self, samples, labels, return_decoded=self.cfg.cycle_weight > 0
            )
            log_probs = result[0] if self.cfg.cycle_weight > 0 else result
            mixture = decoder_mixture_statistics(
                log_probs, labels != -100, self.cfg.chunk_size
            )
            stats.update(mixture)
            stats["sample_ce_sum"] = -log_probs[:, :, 0].sum().detach()
            stats["objective_sum"] = (
                -log_probs.mean(-1).sum()
                if self.cfg.objective == "token_ce"
                else mixture["mixture_block_nll_sum"]
            )
            # 两种文字目标都按 token 数缩放，避免块长造成约 4 倍的梯度尺度差。
            stats["objective_count"] = int((labels != -100).sum())
            stats["tokens"] = stats["objective_count"]
            if self.cfg.cycle_weight > 0:
                decoded = result[1]
                batch, length, count = decoded.shape
                with torch.no_grad():
                    ids = decoded.permute(0, 2, 1).reshape(batch * count, length)
                    encoded, _ = self.codec.encode(
                        ids, torch.full((len(ids),), length, device=ids.device)
                    )
                    canonical = (
                        self.normalize(encoded)
                        .reshape(batch, count, samples.shape[1], -1)
                        .permute(0, 2, 1, 3)
                    )
                    complete = (
                        F.pad(
                            labels != -100,
                            (0, samples.shape[1] * self.cfg.chunk_size - length),
                        )
                        .reshape(batch, -1, self.cfg.chunk_size)
                        .all(-1)
                    )
                # 训练时对齐“自己选定文字的编码”，不是把所有候选拉向同一参考。
                cycle = (samples.float() - canonical).square().mean((-1, -2))
                stats["cycle_sum"] = cycle[complete].sum()
                stats["weighted_cycle_sum"] = self.cfg.cycle_weight * stats["cycle_sum"]
                stats["cycle_blocks"] = int(complete.sum())
            return stats

        # CE 仅作单样本诊断，不参加优化。逐个候选拉向同一份文字会改变分布目标。
        with torch.no_grad():
            logits = self.decode(self.denormalize(samples[:, :, 0]), labels.shape[1])
            stats["sample_ce_sum"] = F.cross_entropy(
                logits.float().flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        stats["tokens"] = int((labels != -100).sum())
        if distribution_diagnostics:
            # 只在验证时解码全部候选；训练的 Energy 目标和开销保持不变。
            with torch.no_grad():
                target_log_probs = []
                for index in range(self.cfg.energy_candidates):
                    scores = self.decode(
                        self.denormalize(samples[:, :, index]), labels.shape[1]
                    )
                    logp = (
                        scores.float()
                        .log_softmax(-1)
                        .gather(-1, labels.clamp_min(0)[..., None])
                        .squeeze(-1)
                    )
                    target_log_probs.append(logp)
                stats.update(
                    decoder_mixture_statistics(
                        torch.stack(target_log_probs, dim=-1),
                        labels != -100,
                        self.cfg.chunk_size,
                    )
                )
        return stats
