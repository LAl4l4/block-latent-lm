"""阶段二：冻结 codec，以普通因果 Transformer 预测下一块 latent。

训练可读取真实历史 latent（teacher forcing）；推理只回灌自己预测的
latent。这个训练/推理差异由阶段三的自主回灌检查测量。
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.codec import BlockCodec
from latent_ar.model import LatentTransformer, ModelConfig


@dataclass
class StagedModelConfig(ModelConfig):
    """与 codec 共用结构配置；当前预测头只是一个线性层。"""


class LatentPriorModel(nn.Module):
    architecture = "latent_codec_ar_v3"

    def __init__(self, cfg: StagedModelConfig):
        super().__init__()
        self.cfg = cfg
        self.codec = BlockCodec(cfg)
        self.backbone = LatentTransformer(cfg)
        self.latent_head = nn.Linear(cfg.dim, cfg.dim)
        self.backbone.apply(BlockCodec._initialize)
        self.latent_head.apply(BlockCodec._initialize)
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

    def prior_statistics(
        self, prompt, prompt_lengths, labels, shuffle_context=False, token_ce_grad=True
    ):
        """两个监督信号分别求和，由训练器按有效块数/有效 token 数归一化。"""
        contexts, targets, valid = self.teacher_contexts(prompt, prompt_lengths, labels)
        if shuffle_context:
            contexts = contexts.roll(1, dims=0)
        predictions = self.latent_head(contexts)
        squared_error = (
            (predictions.float() - self.normalize(targets)).square().mean(-1)
        )
        mse_sum = squared_error.masked_select(valid).sum()

        # decoder 参数冻结，但这里必须保留梯度，才能监督主干预测的 latent。
        with torch.set_grad_enabled(torch.is_grad_enabled() and token_ce_grad):
            logits = self.decode(self.denormalize(predictions), labels.shape[1])
            ce_sum = F.cross_entropy(
                logits.flatten(0, 1).float(),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        return {
            "latent_mse_sum": mse_sum,
            "token_ce_sum": ce_sum,
            "blocks": int(valid.sum()),
            "tokens": int((labels != -100).sum()),
        }

    def rollout(
        self,
        prompt,
        prompt_lengths,
        steps,
        generator=None,
        noise=None,
    ):
        """推理：只使用 prompt 与自身已预测 latent，绝不读取 target。"""
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

        for index in range(steps):
            latent = self.sample_latent(
                context,
                generator=generator,
                noise=None if noise is None else noise[:, index],
            )
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

    def sample_latent(self, context, generator=None, noise=None):
        """V3 保留确定性预测；随机分布版本覆写这个方法。"""
        return self.denormalize(self.latent_head(context))

    def generation_generators(self, generator, device):
        return None, generator

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
    ):
        if not use_latents:
            raise ValueError("V3 generation requires predicted latents")
        if max_new_tokens < 1 or not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("Invalid generation lengths")
        if temperature < 0 or top_k < 0:
            raise ValueError("Invalid token sampling settings")
        was_training = self.training
        self.eval()
        try:
            latent_rng, token_rng = self.generation_generators(generator, prompt.device)
            latents = self.rollout(
                prompt,
                prompt_lengths,
                math.ceil(max_new_tokens / self.cfg.chunk_size),
                generator=latent_rng,
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
