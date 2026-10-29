"""主干的条件分布输出头：对 noise→目标 latent 的速度场做 flow matching。

    x(t) = (1-t) * noise + t * target
    target_velocity = target - noise

网络同时读取 x(t)、时间和主干状态，回归的是条件速度场，而不是直接
用 MSE 把一个确定性向量推向所有可能 target 的均值。
"""

import math

import torch
from torch import nn

from latent_ar.model import RMSNorm


def time_embedding(time, width):
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(width // 2, device=time.device, dtype=torch.float32)
        / max(1, width // 2 - 1)
    )
    angles = time.float() * frequencies * 1000.0
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FlowBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            RMSNorm(width),
            nn.Linear(width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, hidden):
        return hidden + self.layers(hidden)


class ConditionalFlowHead(nn.Module):
    """属于主干的 latent 输出头；不接触文本 token，也不替代 codec decoder。"""

    def __init__(self, latent_dim, context_dim, hidden_dim=512, layers=2, time_dim=64):
        super().__init__()
        if (
            min(latent_dim, context_dim, hidden_dim, layers) < 1
            or time_dim < 2
            or time_dim % 2
        ):
            raise ValueError("Invalid flow head dimensions")
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.context = nn.Linear(context_dim, hidden_dim)
        self.time = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList([FlowBlock(hidden_dim) for _ in range(layers)])
        self.output = nn.Sequential(
            RMSNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, noisy, time, context):
        hidden = self.input(noisy) + self.context(context)
        hidden = hidden + self.time(time_embedding(time, self.time_dim))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)

    def training_loss(self, target, context, valid, generator=None):
        noise = torch.randn(
            target.shape, device=target.device, dtype=torch.float32, generator=generator
        )
        time = torch.rand(
            (*target.shape[:-1], 1), device=target.device, generator=generator
        )
        noisy = (1 - time) * noise + time * target.float()
        velocity = self(noisy, time, context)
        losses = (velocity.float() - (target.float() - noise)).square().mean(-1)
        return losses[valid].sum(), int(valid.sum())

    def sample(self, context, steps=16, generator=None, temperature=1.0, noise=None):
        """固定步长 Euler 积分；步数和初始噪声都属于可复现的推理设置。"""
        if steps < 1 or temperature < 0:
            raise ValueError(
                "Flow steps must be positive and latent temperature nonnegative"
            )
        shape = (*context.shape[:-1], self.latent_dim)
        if noise is None:
            noise = torch.randn(shape, device=context.device, generator=generator)
        if noise.shape != shape:
            raise ValueError("Noise shape must match the sampled latent shape")
        state = noise.float() * temperature
        for index in range(steps):
            time = torch.full((*shape[:-1], 1), index / steps, device=context.device)
            state = state + self(state, time, context).float() / steps
        return state
