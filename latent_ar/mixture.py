"""有限 latent 混合分布，对一个文本块的联合概率做精确边缘化。"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.codec import BlockCodec
from latent_ar.model import RMSNorm, SwiGLU
from latent_ar.staged import LatentPriorModel, StagedModelConfig


@dataclass
class MixtureModelConfig(StagedModelConfig):
    mixture_components: int = 8
    mixture_hidden_dim: int = 512
    mixture_layers: int = 2
    mixture_train_blocks: int = 64
    # 旧检查点保持原始 latent 回灌；新实验显式选择实际文本回灌。
    feedback_mode: str = "raw"

    def __post_init__(self):
        super().__post_init__()
        if self.feedback_mode not in {"raw", "text"}:
            raise ValueError("feedback_mode must be raw or text")
        if (
            min(
                self.mixture_components,
                self.mixture_hidden_dim,
                self.mixture_layers,
                self.mixture_train_blocks,
            )
            < 1
        ):
            raise ValueError(
                "Mixture dimensions and training block count must be positive"
            )


def mixture_block_nll(logits, weights, labels):
    """[blocks,K,slots,vocab], [blocks,K], [blocks,slots] -> [blocks]。

    同一分量须解释整块，不能逐 token 独立挑选分量。padding 的概率为 1。
    """
    count, components, slots, vocab = logits.shape
    if weights.shape != (count, components) or labels.shape != (count, slots):
        raise ValueError("Invalid mixture likelihood shapes")
    expanded = labels[:, None].expand(-1, components, -1)
    losses = F.cross_entropy(
        logits.float().reshape(-1, vocab),
        expanded.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(count, components, slots)
    return -torch.logsumexp(weights.float().log_softmax(-1) - losses.sum(-1), -1)


class MixtureHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.components, self.dim = cfg.mixture_components, cfg.dim
        self.input = nn.Linear(cfg.dim, cfg.mixture_hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(cfg.mixture_hidden_dim),
                    SwiGLU(cfg.mixture_hidden_dim, cfg.mixture_hidden_dim),
                )
                for _ in range(cfg.mixture_layers)
            ]
        )
        self.norm = RMSNorm(cfg.mixture_hidden_dim)
        self.output = nn.Linear(
            cfg.mixture_hidden_dim, cfg.mixture_components * cfg.dim
        )
        self.weights = nn.Linear(cfg.mixture_hidden_dim, cfg.mixture_components)

    def forward(self, context):
        state = self.input(context)
        for block in self.blocks:
            state = state + block(state)
        state = self.norm(state)
        return (
            self.output(state).reshape(*context.shape[:-1], self.components, self.dim),
            self.weights(state),
        )


class LatentMixtureModel(LatentPriorModel):
    """一次主干预测 K 个 latent 及权重，生成可选择真实文本回灌。"""

    architecture = "latent_codec_mixture_v5"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.latent_head = MixtureHead(cfg)
        self.latent_head.apply(BlockCodec._initialize)

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
        feedback_mode=None,
        return_trace=False,
    ):
        """新配置逐块解码并回灌；旧配置默认保留原生成行为。

        显式传入 feedback_mode='raw' 也走逐块接口，便于与 text 做配对
        比较。该接口不是异步性能实现。trace 留给后续反馈蒸馏收集数据。
        """
        mode = self.cfg.feedback_mode if feedback_mode is None else feedback_mode
        if mode == "raw" and feedback_mode is None and not return_trace:
            return super().generate(
                prompt,
                prompt_lengths,
                max_new_tokens,
                temperature,
                top_k,
                use_latents,
                generator,
                min_new_tokens,
            )
        from latent_ar.feedback import generate_with_feedback

        return generate_with_feedback(
            self,
            prompt,
            prompt_lengths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_latents=use_latents,
            generator=generator,
            min_new_tokens=min_new_tokens,
            feedback_mode=mode,
            return_trace=return_trace,
        )

    def generation_generators(self, generator, device):
        seeds = torch.randint(
            0, 2**62, (2,), device=device, generator=generator
        ).tolist()
        return tuple(torch.Generator(device=device).manual_seed(s) for s in seeds)

    def sample_latent(self, context, generator=None, noise=None):
        candidates, weights = self.latent_head(context)
        # 显式 U(0,1) 噪声允许 teacher/free 使用相同的逐位置抽样。
        if noise is None:
            noise = torch.rand(
                (*context.shape[:-1], 1), device=context.device, generator=generator
            )
        if noise.shape != (*context.shape[:-1], 1):
            raise ValueError("Mixture noise must end in one uniform coordinate")
        cumulative = weights.float().softmax(-1).cumsum(-1)
        index = (noise >= cumulative).sum(-1).clamp_max(self.cfg.mixture_components - 1)
        latent = candidates.gather(
            -2, index[..., None, None].expand(*index.shape, 1, self.cfg.dim)
        ).squeeze(-2)
        return self.denormalize(latent)

    def prior_statistics(
        self, prompt, prompt_lengths, labels, shuffle_context=False, generator=None
    ):
        contexts, targets, valid = self.teacher_contexts(prompt, prompt_lengths, labels)
        if shuffle_context:
            contexts = contexts.roll(1, 0)
        contexts = contexts[valid]
        padded = F.pad(
            labels,
            (0, targets.shape[1] * self.cfg.chunk_size - labels.shape[1]),
            value=-100,
        ).reshape(*valid.shape, self.cfg.chunk_size)[valid]
        full_count = len(contexts)
        # 均匀抽块的加权和是全量联合 NLL 的无偏估计。验证计算所有有效块。
        if self.training and full_count > self.cfg.mixture_train_blocks:
            index = torch.randperm(
                full_count, device=contexts.device, generator=generator
            )[: self.cfg.mixture_train_blocks]
            contexts, padded = contexts[index], padded[index]
        sums = []
        for start in range(0, len(contexts), self.cfg.mixture_train_blocks):
            candidates, weights = self.latent_head(
                contexts[start : start + self.cfg.mixture_train_blocks]
            )
            logits = self.codec.decoder(self.denormalize(candidates))
            sums.append(
                mixture_block_nll(
                    logits,
                    weights,
                    padded[start : start + self.cfg.mixture_train_blocks],
                ).sum()
            )
        return {
            "text_nll_sum": torch.stack(sums).sum() * (full_count / len(contexts)),
            "tokens": int((labels != -100).sum()),
            "blocks": full_count,
            "scored_blocks": len(contexts),
        }
