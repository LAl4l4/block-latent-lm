"""连续 latent 去噪回归：目标仍为冻结 codec 的真实 latent，不使用 Energy。"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from latent_ar.codec import BlockCodec
from latent_ar.flow import ConditionalFlowHead
from latent_ar.staged import LatentPriorModel, StagedModelConfig


@dataclass
class DenoisingConfig(StagedModelConfig):
    denoise_hidden_dim: int = 512
    denoise_layers: int = 4
    denoise_steps: int = 32
    context_dropout: float = 0.1
    denoise_guidance: float = 1.5

    def __post_init__(self):
        super().__post_init__()
        if min(self.denoise_hidden_dim, self.denoise_layers, self.denoise_steps) < 1:
            raise ValueError("Denoising dimensions and steps must be positive")
        if not 0 <= self.context_dropout < 1 or not 0 <= self.denoise_guidance < float(
            "inf"
        ):
            raise ValueError("Invalid dropout or guidance")


class LatentDenoisingModel(LatentPriorModel):
    architecture = "latent_codec_denoising_v7"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.latent_head = ConditionalFlowHead(
            cfg.dim, cfg.dim, cfg.denoise_hidden_dim, cfg.denoise_layers
        )
        self.latent_head.apply(BlockCodec._initialize)

    def draw_noise(self, shape, device, generator=None):
        return torch.randn((*shape, self.cfg.dim), device=device, generator=generator)

    def prior_statistics(
        self,
        prompt,
        prompt_lengths,
        labels,
        shuffle_context=False,
        generator=None,
        token_ce_grad=True,
    ):
        context, targets, valid = self.teacher_contexts(prompt, prompt_lengths, labels)
        if shuffle_context:
            context = context.roll(1, 0)
        noise = self.draw_noise(valid.shape, targets.device, generator)
        time = torch.rand((*valid.shape, 1), device=targets.device, generator=generator)
        normalized = self.normalize(targets)
        noisy = (1 - time) * noise + time * normalized
        if self.training and self.cfg.context_dropout:
            keep = torch.rand(
                (*valid.shape, 1), device=targets.device, generator=generator
            )
            context = context * (keep >= self.cfg.context_dropout)
        prediction = self.latent_head(noisy, time, context)
        mse = (prediction.float() - normalized).square().mean(-1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and token_ce_grad):
            logits = self.decode(self.denormalize(prediction), labels.shape[1])
            ce = F.cross_entropy(
                logits.float().flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        return {
            "latent_mse_sum": mse[valid].sum(),
            "token_ce_sum": ce,
            "blocks": int(valid.sum()),
            "tokens": int((labels != -100).sum()),
        }

    def sample_latent(self, context, generator=None, noise=None):
        # x_t=(1-t)epsilon+t*z；回归 clean z，速度为 (z_hat-x_t)/(1-t)。
        if noise is None:
            noise = self.draw_noise(context.shape[:-1], context.device, generator)
        if noise.shape != context.shape[:-1] + (self.cfg.dim,):
            raise ValueError("Denoising noise must match latent shape")
        state = noise.float()
        for index in range(self.cfg.denoise_steps):
            time = torch.full(
                (*context.shape[:-1], 1),
                index / self.cfg.denoise_steps,
                device=context.device,
            )
            clean = self.latent_head(state, time, context).float()
            if self.cfg.denoise_guidance != 1:
                unconditional = self.latent_head(
                    state, time, torch.zeros_like(context)
                ).float()
                clean = unconditional + self.cfg.denoise_guidance * (
                    clean - unconditional
                )
            state = state + (clean - state) / (self.cfg.denoise_steps - index)
        return self.denormalize(state)

    def generation_generators(self, generator, device):
        seeds = torch.randint(
            0, 2**62, (2,), device=device, generator=generator
        ).tolist()
        return tuple(torch.Generator(device=device).manual_seed(s) for s in seeds)
