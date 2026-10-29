"""Energy + 可微文字 n-gram Brier，保持隐式分布和纯 latent 回灌。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from latent_ar.energy import EnergyModelConfig, LatentEnergyModel, energy_statistics


@dataclass
class EnergyTextConfig(EnergyModelConfig):
    text_score_weight: float = 4.0
    text_score_blocks: int = 64

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.text_score_weight < float("inf") or self.text_score_blocks < 1:
            raise ValueError("Text score weight and block count must be positive")


def decoder_brier_loss(logits, labels):
    """每块的负 Brier reward（1..slots 阶求和）。去除候选自配对。

    奖励目标概率，惩罚独立候选分布的碰撞概率；同一 n-gram 共享 latent。
    """
    blocks, draws, width, vocab = logits.shape
    if draws < 2 or labels.shape != (blocks, width):
        raise ValueError("Brier requires independent samples and matching block labels")
    probability = logits.float().softmax(-1)
    valid = labels != -100
    hit = probability.gather(
        -1, labels.clamp_min(0)[:, None, :, None].expand(-1, draws, -1, 1)
    ).squeeze(-1)
    slots = probability.transpose(1, 2)
    collision = slots @ slots.transpose(-1, -2)
    off_diagonal = ~torch.eye(draws, device=logits.device, dtype=torch.bool)
    losses = logits.new_zeros(blocks, dtype=torch.float32)
    for n in range(1, width + 1):
        total = torch.zeros_like(losses)
        counts = torch.zeros_like(losses)
        for start in range(width - n + 1):
            mask = valid[:, start : start + n].all(-1)
            target_probability = hit[:, :, start : start + n].prod(-1).mean(-1)
            agreement = (
                collision[:, start : start + n].prod(1)[:, off_diagonal].mean(-1)
            )
            total = total + (agreement - 2 * target_probability) * mask
            counts = counts + mask
        losses = losses + total / counts.clamp_min(1)
    return losses


class LatentEnergyTextModel(LatentEnergyModel):
    architecture = "latent_codec_energy_text_v5"

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
            contexts = contexts.roll(1, 0)
        noise = self.draw_noise(
            (*valid.shape, self.cfg.energy_candidates), contexts.device, generator
        )
        samples = self.latent_head(contexts[:, :, None], noise)
        stats = energy_statistics(samples, self.normalize(targets), valid)
        values = samples[valid]
        padded = F.pad(
            labels,
            (0, targets.shape[1] * self.cfg.chunk_size - labels.shape[1]),
            value=-100,
        ).reshape(*valid.shape, self.cfg.chunk_size)[valid]
        count = len(values)
        if self.training and count > self.cfg.text_score_blocks:
            index = torch.randperm(count, device=values.device, generator=generator)[
                : self.cfg.text_score_blocks
            ]
            values, padded = values[index], padded[index]
        losses = []
        for start in range(0, len(values), self.cfg.text_score_blocks):
            logits = self.codec.decoder(
                self.denormalize(values[start : start + self.cfg.text_score_blocks])
            )
            losses.append(
                decoder_brier_loss(
                    logits, padded[start : start + self.cfg.text_score_blocks]
                ).sum()
            )
        stats["text_brier_sum"] = torch.stack(losses).sum() * (count / len(values))
        with torch.no_grad():
            logits = self.decode(self.denormalize(samples[:, :, 0]), labels.shape[1])
            stats["sample_ce_sum"] = F.cross_entropy(
                logits.float().flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        stats["tokens"] = int((labels != -100).sum())
        return stats
