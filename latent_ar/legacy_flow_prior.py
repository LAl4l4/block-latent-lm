"""历史 flow 实验：仅为加载已有实验检查点保留，不是当前训练入口。

训练可读取真实历史 latent（teacher forcing）；推理只回灌自己采样的
latent。这个分布差异是阶段三必须测量的对象，不隐瞒为“没有差异”。
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from latent_ar.codec import BlockCodec
from latent_ar.flow import ConditionalFlowHead
from latent_ar.model import LatentTransformer, ModelConfig


@dataclass
class StagedModelConfig(ModelConfig):
    flow_hidden_dim: int = 512
    flow_layers: int = 2
    flow_time_dim: int = 64
    flow_steps: int = 16

    def __post_init__(self):
        super().__post_init__()
        if min(self.flow_hidden_dim, self.flow_layers, self.flow_steps) < 1:
            raise ValueError("Flow dimensions and steps must be positive")
        if self.flow_time_dim < 2 or self.flow_time_dim % 2:
            raise ValueError("flow_time_dim must be positive and even")


class LatentPriorModel(nn.Module):
    architecture = "latent_codec_flow_v3"

    def __init__(self, cfg: StagedModelConfig):
        super().__init__()
        self.cfg = cfg
        self.codec = BlockCodec(cfg)
        self.backbone = LatentTransformer(cfg)
        self.flow_head = ConditionalFlowHead(
            cfg.dim, cfg.dim, cfg.flow_hidden_dim, cfg.flow_layers, cfg.flow_time_dim
        )
        self.backbone.apply(BlockCodec._initialize)
        self.flow_head.apply(BlockCodec._initialize)
        self.codec.requires_grad_(False)
        self.codec.eval()
        self.register_buffer("latent_mean", torch.zeros(cfg.dim))
        self.register_buffer("latent_std", torch.ones(cfg.dim))

    def train(self, mode=True):
        super().train(mode)
        # codec 表示空间一旦交给 prior，就不能继续漂移。
        self.codec.eval()
        return self

    def normalize(self, latent):
        return (latent.float() - self.latent_mean) / self.latent_std

    def denormalize(self, latent):
        return latent.float() * self.latent_std + self.latent_mean

    @torch.no_grad()
    def set_latent_statistics(self, latents):
        values = latents.float().reshape(-1, self.cfg.dim)
        self.latent_mean.copy_(values.mean(0))
        self.latent_std.copy_(values.std(0, unbiased=False).clamp_min(0.1))

    def teacher_contexts(self, prompt, prompt_lengths, labels):
        """为每个目标块建立只看真实前缀的主干状态，全部位置并行计算。"""
        with torch.no_grad():
            encoded, prompt_valid = self.codec.encode(prompt, prompt_lengths)
            targets, target_valid = self.codec.encode_labels(labels)
        batch, prompt_width, _ = encoded.shape
        steps = targets.shape[1]
        lengths = prompt_valid.sum(-1)

        # 预测 z1 时只看 prompt；预测 z2 才能看真实 z1，依此类推。
        history = torch.cat([encoded, targets[:, :-1]], dim=1)
        valid = torch.cat([prompt_valid, target_valid[:, :-1]], dim=1)
        prompt_positions = torch.arange(prompt_width, device=prompt.device).expand(
            batch, -1
        )
        target_positions = lengths[:, None] + torch.arange(
            steps - 1, device=prompt.device
        )
        positions = torch.cat([prompt_positions, target_positions], dim=1)
        hidden, _ = self.backbone(history, positions, valid)
        indices = torch.cat(
            [
                (lengths - 1)[:, None],
                torch.arange(
                    prompt_width, prompt_width + steps - 1, device=prompt.device
                ).expand(batch, -1),
            ],
            dim=1,
        )
        contexts = hidden.gather(1, indices[..., None].expand(-1, -1, self.cfg.dim))
        return contexts, targets, target_valid

    def prior_loss(
        self, prompt, prompt_lengths, labels, generator=None, shuffle_context=False
    ):
        contexts, targets, valid = self.teacher_contexts(prompt, prompt_lengths, labels)
        if shuffle_context:
            contexts = contexts.roll(1, dims=0)
        return self.flow_head.training_loss(
            self.normalize(targets), contexts, valid, generator
        )

    def rollout(
        self,
        prompt,
        prompt_lengths,
        steps,
        generator=None,
        flow_steps=None,
        latent_temperature=1.0,
        noise=None,
    ):
        """推理：只使用 prompt 与自身已采样 latent，绝不读取 target。"""
        if steps < 1:
            raise ValueError("Rollout steps must be positive")
        with torch.no_grad():
            encoded, valid = self.codec.encode(prompt, prompt_lengths)
        batch, width, _ = encoded.shape
        lengths = valid.sum(-1)
        positions = torch.arange(width, device=prompt.device).expand(batch, -1)
        hidden, caches = self.backbone(encoded, positions, valid)
        context = hidden[torch.arange(batch, device=prompt.device), lengths - 1]
        generated = []
        if noise is not None and noise.shape != (batch, steps, self.cfg.dim):
            raise ValueError("Rollout noise must have shape [batch, steps, latent_dim]")

        for index in range(steps):
            sampled = self.flow_head.sample(
                context,
                steps=self.cfg.flow_steps if flow_steps is None else flow_steps,
                generator=generator,
                temperature=latent_temperature,
                noise=None if noise is None else noise[:, index],
            )
            latent = self.denormalize(sampled)
            generated.append(latent)
            if index + 1 < steps:
                valid = torch.cat(
                    [
                        valid,
                        torch.ones(batch, 1, device=prompt.device, dtype=torch.bool),
                    ],
                    dim=1,
                )
                hidden, caches = self.backbone(
                    latent[:, None], (lengths + index)[:, None], valid, caches
                )
                context = hidden[:, 0]
        return torch.stack(generated, dim=1)

    def decode(self, latents, output_tokens):
        return self.codec.decode(latents)[:, :output_tokens]

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        prompt_lengths,
        max_new_tokens=64,
        temperature=0.0,
        top_k=50,
        use_latents=True,
        generator=None,
        min_new_tokens=0,
        flow_steps=None,
        latent_temperature=1.0,
    ):
        if not use_latents:
            raise ValueError("V3 generation requires sampled latents")
        if max_new_tokens < 1 or not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("Invalid generation lengths")
        if temperature < 0 or top_k < 0:
            raise ValueError("Invalid token sampling settings")
        was_training = self.training
        self.eval()
        try:
            # 分开 latent / token 的随机数流，改变词采样不影响主干选择的内容。
            seeds = torch.randint(
                0, 2**62, (2,), generator=generator, device=prompt.device
            ).tolist()
            latent_rng = torch.Generator(device=prompt.device).manual_seed(seeds[0])
            token_rng = torch.Generator(device=prompt.device).manual_seed(seeds[1])
            latents = self.rollout(
                prompt,
                prompt_lengths,
                math.ceil(max_new_tokens / self.cfg.chunk_size),
                generator=latent_rng,
                flow_steps=flow_steps,
                latent_temperature=latent_temperature,
            )
            logits = self.decode(latents, max_new_tokens)
            finished = torch.zeros(
                prompt.shape[0], device=prompt.device, dtype=torch.bool
            )
            outputs = []
            for start in range(0, max_new_tokens, self.cfg.chunk_size):
                scores = logits[:, start : start + self.cfg.chunk_size].float().clone()
                scores[..., self.cfg.pad_id] = -torch.inf
                scores[..., self.cfg.bos_id] = -torch.inf
                positions = torch.arange(
                    start, start + scores.shape[1], device=prompt.device
                )
                scores[:, positions < min_new_tokens, self.cfg.eos_id] = -torch.inf
                if temperature == 0:
                    tokens = scores.argmax(-1)
                else:
                    scores = scores / temperature
                    if top_k:
                        threshold = scores.topk(
                            min(top_k, scores.shape[-1]), dim=-1
                        ).values[..., -1:]
                        scores = scores.masked_fill(scores < threshold, -torch.inf)
                    tokens = torch.multinomial(
                        scores.softmax(-1).reshape(-1, self.cfg.vocab_size),
                        1,
                        generator=token_rng,
                    ).reshape(prompt.shape[0], -1)
                for offset in range(tokens.shape[1]):
                    token = torch.where(finished, self.cfg.eos_id, tokens[:, offset])
                    outputs.append(token)
                    finished |= token == self.cfg.eos_id
                    if bool(finished.all()):
                        break
                if bool(finished.all()):
                    break
            return torch.stack(outputs, dim=1)
        finally:
            self.train(was_training)
