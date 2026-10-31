"""使用多步条件流匹配作为 CALM 一次生成头的独立对照。"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.calm import CalmConfig, CalmModel, ConditionalBlock, shuffle_valid_contexts
from latent_ar.codec import BlockCodec


@dataclass
class FlowConfig(CalmConfig):
    flow_steps: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.flow_steps < 1:
            raise ValueError("流采样步数必须为正")


def flow_interpolate(noise, target, time):
    """返回线性路径上的位置及其速度；time 的末维须可广播。"""
    return (1 - time) * noise + time * target, target - noise


class FlowHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.state = nn.Linear(cfg.dim, cfg.dim)
        self.context = nn.Linear(cfg.dim, cfg.dim)
        self.time = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )
        self.state_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.context_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            ConditionalBlock(cfg.dim) for _ in range(cfg.energy_layers)
        )
        self.final_norm = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.output = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )

    def forward(self, context, state, time):
        # 固定傅里叶特征使连续时间可被同一个速度网络读取。
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=state.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        phase = time.float() * frequencies * (2 * math.pi)
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        condition = self.context_norm(self.context(context)) + self.time(embedding)
        hidden = self.state_norm(self.state(state))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(self.final_norm(hidden))


class ConditionalFlowModel(CalmModel):
    architecture = "latent_conditional_flow_v1"

    def __init__(self, cfg: FlowConfig):
        super().__init__(cfg)
        self.latent_head = FlowHead(cfg)
        self.latent_head.apply(BlockCodec._initialize)
        nn.init.zeros_(self.latent_head.output[-1].weight)
        nn.init.zeros_(self.latent_head.output[-1].bias)

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
        context, mu, sigma = (
            contexts[valid],
            mean[valid].float(),
            log_std[valid].float().exp(),
        )
        posterior_noise = torch.randn(mu.shape, device=mu.device, generator=generator)
        target = mu + sigma * posterior_noise
        noise = torch.randn(mu.shape, device=mu.device, generator=generator)
        time = torch.rand((*mu.shape[:-1], 1), device=mu.device, generator=generator)
        state, velocity = flow_interpolate(noise, target, time)
        predicted = self.latent_head(context, state, time)
        loss = (predicted.float() - velocity).square().mean(-1).sum()
        blocks = int(valid.sum())
        return {
            "flow_sum": loss,
            "objective_sum": loss,
            "objective_count": blocks,
            "blocks": blocks,
            "tokens": int(labels.ne(-100).sum()),
        }

    def sample_latent(self, context, generator=None, noise=None, steps=None):
        """从标准高斯积分到 VAE 原始坐标；每次生成需要多次网络计算。"""
        steps = self.cfg.flow_steps if steps is None else steps
        if steps < 1:
            raise ValueError("流采样步数必须为正")
        if noise is None:
            noise = torch.randn(
                context.shape, device=context.device, generator=generator
            )
        if noise.shape != context.shape:
            raise ValueError("流初始噪声维度必须与上下文一致")
        state = noise.float()
        for index in range(steps):
            time = torch.full(
                (*state.shape[:-1], 1), index / steps, device=state.device
            )
            state = state + self.latent_head(context, state, time).float() / steps
        return state
