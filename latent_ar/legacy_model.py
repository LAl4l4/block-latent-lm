"""历史 V1 架构：仅用于加载和对照旧版带 cross-attention 的检查点。

主要形状：token 为 [batch, token数]，latent 为 [batch, latent数, 512]。
rollout() 只读取 prompt 和自己产生的 latent，不接收真实续写文本。
forward() 用于联合训练；generate() 是同步生成参考实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

Cache = tuple[Tensor, Tensor]


@dataclass
class ModelConfig:
    """模型结构参数；词表与特殊 token 编号来自准备好的训练数据。"""

    vocab_size: int = 8192
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    chunk_size: int = 4
    encoder_dim: int = 256
    dim: int = 512
    layers: int = 8
    heads: int = 8
    ff_dim: int = 1536
    decoder_dim: int = 256
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_ff_dim: int = 768

    def __post_init__(self):
        for width, heads in [
            (self.dim, self.heads),
            (self.decoder_dim, self.decoder_heads),
        ]:
            if heads <= 0 or width % heads or (width // heads) % 2:
                raise ValueError("Attention head dimensions must be positive and even")
        if min(self.chunk_size, self.layers, self.decoder_layers) < 1:
            raise ValueError("Chunk size and layer counts must be positive")


class RMSNorm(nn.Module):
    """在最后一个维度上归一化；平方和在 FP32 中计算以提高数值稳定性。"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return normalized.to(x.dtype) * self.weight


def rope(x: Tensor, positions: Tensor) -> Tensor:
    """为 Q/K 加旋转位置编码。

    x: [batch, head数, 位置数, 每个head的维度]。
    positions: [batch, 位置数]；逻辑位置不计入右侧 padding。
    """
    width = x.shape[-1]
    freq = torch.exp(
        torch.arange(0, width, 2, device=x.device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    angles = positions.float().unsqueeze(-1) * freq
    cos, sin = angles.cos().unsqueeze(1).to(x.dtype), angles.sin().unsqueeze(1).to(
        x.dtype
    )
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)


class SwiGLU(nn.Module):
    """门控前馈网络：一半投影作为门，另一半提供被调节的特征。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class CausalAttention(nn.Module):
    """因果自注意力，既支持完整前缀，也支持带 KV cache 的单步推进。"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        key_valid: Tensor,
        cache: Cache | None = None,
    ) -> tuple[Tensor, Cache]:
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .view(batch, length, 3, self.heads, width // self.heads)
            .unbind(2)
        )
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = rope(q, positions), rope(k, positions)
        past = 0 if cache is None else cache[0].shape[2]
        if cache is not None:
            # 拼接保留旧缓存的计算图，后续损失仍能传回早期 latent。
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        keys = torch.arange(k.shape[2], device=x.device)
        queries = past + torch.arange(length, device=x.device)
        causal = keys[None, :] <= queries[:, None]
        allowed = key_valid[:, None, None, :] & causal[None, None]
        # cached query 的位置从 past 开始。直接用 is_causal=True 的左上角
        # mask 会让单步 query 只能看见第一个 key，因此这里显式计算偏移。
        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=0.0
        )
        value = value.transpose(1, 2).reshape(batch, length, width)
        return self.out(value), (k, v)


class Block(nn.Module):
    """主干的一层：归一化 → 自注意力 → 残差 → 门控前馈 → 残差。"""

    def __init__(self, dim: int, heads: int, ff_dim: int):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attention = CausalAttention(dim, heads)
        self.ff = SwiGLU(dim, ff_dim)

    def forward(self, x: Tensor, positions: Tensor, valid: Tensor, cache: Cache | None):
        update, cache = self.attention(self.norm1(x), positions, valid, cache)
        x = x + update
        return x + self.ff(self.norm2(x)), cache


class ChunkEncoder(nn.Module):
    """把每四个 token 按顺序拼成一块，再映射为一个 512 维向量。

    MLP 只处理块内信息；跨块上下文由后续的主干 Transformer 建立。
    四个有效位置标志用于区分完整块与不足四个 token 的尾块。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.encoder_dim, padding_idx=cfg.pad_id
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.chunk_size * cfg.encoder_dim + cfg.chunk_size, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim),
            RMSNorm(cfg.dim),
        )

    def forward(self, ids: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch, length = ids.shape
        chunks = (length + self.chunk_size - 1) // self.chunk_size
        positions = torch.arange(length, device=ids.device)
        valid = positions[None, :] < lengths[:, None]
        embedded = self.embedding(ids) * valid.unsqueeze(-1)
        # 只在张量里补齐尾块；不使用 target token 填充 prompt。
        extra = chunks * self.chunk_size - length
        embedded = F.pad(embedded, (0, 0, 0, extra)).reshape(batch, chunks, -1)
        slots = F.pad(valid, (0, extra)).reshape(batch, chunks, self.chunk_size)
        latents = self.mlp(torch.cat((embedded, slots.to(embedded.dtype)), dim=-1))
        chunk_valid = slots.any(-1)
        return latents * chunk_valid.unsqueeze(-1), chunk_valid


class LatentTransformer(nn.Module):
    """只接收 latent 的主干；每层返回的新缓存用于下一次自由回灌。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, cfg.ff_dim) for _ in range(cfg.layers)]
        )
        self.norm = RMSNorm(cfg.dim)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        valid: Tensor,
        caches: list[Cache] | None = None,
    ):
        new_caches = []
        for index, block in enumerate(self.blocks):
            x, cache = block(
                x, positions, valid, None if caches is None else caches[index]
            )
            new_caches.append(cache)
        return self.norm(x), new_caches


class CrossAttention(nn.Module):
    """文本状态提供 Q，512 维 latent 记忆投影为 decoder 宽度的 K/V。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.decoder_heads
        self.q = nn.Linear(cfg.decoder_dim, cfg.decoder_dim, bias=False)
        self.kv = nn.Linear(cfg.dim, 2 * cfg.decoder_dim, bias=False)
        self.out = nn.Linear(cfg.decoder_dim, cfg.decoder_dim, bias=False)

    def project_memory(self, memory: Tensor, positions: Tensor) -> Cache:
        batch, length, _ = memory.shape
        k, v = self.kv(memory).view(batch, length, 2, self.heads, -1).unbind(2)
        return rope(k.transpose(1, 2), positions), v.transpose(1, 2)

    def forward(self, x: Tensor, positions: Tensor, memory_kv: Cache, allowed: Tensor):
        batch, length, width = x.shape
        q = self.q(x).view(batch, length, self.heads, -1).transpose(1, 2)
        q = rope(q, positions)
        value = F.scaled_dot_product_attention(
            q, *memory_kv, attn_mask=allowed[:, None], dropout_p=0.0
        )
        return self.out(value.transpose(1, 2).reshape(batch, length, width))


class DecoderBlock(nn.Module):
    """依次读取续写历史、latent 记忆，再通过前馈网络更新文本状态。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (
            RMSNorm(cfg.decoder_dim) for _ in range(3)
        )
        self.attention = CausalAttention(cfg.decoder_dim, cfg.decoder_heads)
        self.cross = CrossAttention(cfg)
        self.ff = SwiGLU(cfg.decoder_dim, cfg.decoder_ff_dim)

    def forward(
        self,
        x,
        token_positions,
        latent_positions,
        self_valid,
        allowed,
        memory_kv,
        cache,
    ):
        update, cache = self.attention(
            self.norm1(x), token_positions, self_valid, cache
        )
        x = x + update
        x = x + self.cross(self.norm2(x), latent_positions, memory_kv, allowed)
        return x + self.ff(self.norm3(x)), cache


class TextDecoder(nn.Module):
    """较小的文本 AR 模型：维护自己的文本历史，读取主干提供的状态。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.decoder_dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.decoder_layers)]
        )
        self.norm = RMSNorm(cfg.decoder_dim)
        self.output = nn.Linear(cfg.decoder_dim, cfg.vocab_size, bias=False)
        # 输入词向量与词表输出共享参数，减少 decoder 的参数量。
        self.output.weight = self.embedding.weight

    def project_memory(self, memory, positions):
        return [block.cross.project_memory(memory, positions) for block in self.blocks]

    def forward(
        self,
        ids: Tensor,
        prompt_valid: Tensor,
        generated_count: int,
        memory_kvs: list[Cache],
        caches: list[Cache] | None = None,
    ):
        batch, length = ids.shape
        start = 0 if caches is None else caches[0][0].shape[2]
        token_positions = torch.arange(start, start + length, device=ids.device).expand(
            batch, -1
        )
        block_indices = token_positions // self.cfg.chunk_size
        # 自注意力使用 token 时钟；跨注意力使用每四个 token 一步的
        # latent 时钟，生成位置紧接每个样本的有效 prompt 块。
        prompt_lengths = prompt_valid.sum(-1)
        latent_positions = prompt_lengths[:, None] + block_indices
        generated_indices = torch.arange(generated_count, device=ids.device)
        visible_generated = (
            generated_indices[None, None, :] <= block_indices[:, :, None]
        )
        # 预测第 1–4 个 token 只允许读取 z1，第 5–8 个可读 z1/z2。
        # prompt 的有效块始终可见，未来生成 latent 即使已算出也不可见。
        allowed = torch.cat(
            (prompt_valid[:, None, :].expand(-1, length, -1), visible_generated), dim=-1
        )
        self_valid = torch.ones(
            batch, start + length, dtype=torch.bool, device=ids.device
        )
        x, new_caches = self.embedding(ids), []
        for index, block in enumerate(self.blocks):
            x, cache = block(
                x,
                token_positions,
                latent_positions,
                self_valid,
                allowed,
                memory_kvs[index],
                None if caches is None else caches[index],
            )
            new_caches.append(cache)
        return self.output(self.norm(x)), new_caches


class LatentARModel(nn.Module):
    """组合 encoder、latent 主干、回灌投影与文本 decoder。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ChunkEncoder(cfg)
        self.backbone = LatentTransformer(cfg)
        self.latent_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim, bias=False), RMSNorm(cfg.dim)
        )
        self.decoder = TextDecoder(cfg)
        self.apply(self._initialize)
        with torch.no_grad():
            self.encoder.embedding.weight[cfg.pad_id].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def rollout(self, prompt: Tensor, prompt_lengths: Tensor, steps: int):
        """自主生成指定数量的 latent，整个展开保留反向传播计算图。

        返回：全部记忆、记忆位置、prompt 有效块 mask、逐步生成的 latent。
        返回逐步列表是为了测试早期 latent 是否能收到后续位置的梯度。
        """
        if steps < 0 or bool((prompt_lengths < 1).any()):
            raise ValueError("Need a nonempty prompt and nonnegative rollout length")
        prompt_latents, prompt_valid = self.encoder(prompt, prompt_lengths)
        lengths = prompt_valid.sum(-1)
        batch, prompt_width, _ = prompt_latents.shape
        prompt_positions = torch.arange(prompt_width, device=prompt.device).expand(
            batch, -1
        )
        generated = []
        if steps:
            # 先处理完整 prompt。z1 来自最后一个有效位置，不来自 padding。
            hidden, caches = self.backbone(
                prompt_latents, prompt_positions, prompt_valid
            )
            last = hidden[torch.arange(batch, device=prompt.device), lengths - 1]
            valid = prompt_valid
            for step in range(steps):
                z = self.latent_head(last)
                generated.append(z)
                if step + 1 < steps:
                    # 主干立即回灌自己产生的 z，不等待文本 decoder。
                    # 最后一个 z 生成后不再多做一次用不到的主干前向。
                    valid = torch.cat(
                        (
                            valid,
                            torch.ones(
                                batch, 1, device=prompt.device, dtype=torch.bool
                            ),
                        ),
                        dim=1,
                    )
                    hidden, caches = self.backbone(
                        z[:, None], (lengths + step)[:, None], valid, caches
                    )
                    last = hidden[:, 0]
        z = torch.stack(generated, dim=1) if generated else prompt_latents[:, :0]
        memory = torch.cat((prompt_latents, z), dim=1)
        # decoder 读取原始 prompt 编码与生成 latent；不是主干的全部 hidden。
        generated_positions = (
            lengths[:, None] + torch.arange(steps, device=prompt.device)[None, :]
        )
        memory_positions = torch.cat((prompt_positions, generated_positions), dim=1)
        return memory, memory_positions, prompt_valid, generated

    def forward(
        self,
        prompt: Tensor,
        prompt_lengths: Tensor,
        decoder_input: Tensor,
        use_latents: bool = True,
    ):
        """训练前向：先自由展开 latent，再并行计算 teacher-forced 文本 logits。"""
        steps = (
            math.ceil(decoder_input.shape[1] / self.cfg.chunk_size)
            if use_latents
            else 0
        )
        memory, positions, valid, _ = self.rollout(prompt, prompt_lengths, steps)
        memory_kvs = self.decoder.project_memory(memory, positions)
        logits, _ = self.decoder(decoder_input, valid, steps, memory_kvs)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt: Tensor,
        prompt_lengths: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        use_latents: bool = True,
        generator: torch.Generator | None = None,
        min_new_tokens: int = 0,
    ) -> Tensor:
        """同步参考生成：先算 latent，再逐 token 解码并严格限制可见范围。

        decoder 维护自己的文本 KV cache；采样结果不会回灌给主干。
        这里没有实现异步队列，因此不能用于证明异步首 token 延迟。
        """
        if max_new_tokens < 1 or temperature < 0 or top_k < 0:
            raise ValueError("Invalid generation settings")
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens must be between 0 and max_new_tokens")
        was_training = self.training
        self.eval()
        try:
            steps = (
                math.ceil(max_new_tokens / self.cfg.chunk_size) if use_latents else 0
            )
            memory, positions, valid, _ = self.rollout(prompt, prompt_lengths, steps)
            memory_kvs = self.decoder.project_memory(memory, positions)
            batch = prompt.shape[0]
            current = torch.full(
                (batch, 1), self.cfg.bos_id, device=prompt.device, dtype=torch.long
            )
            finished = torch.zeros(batch, device=prompt.device, dtype=torch.bool)
            outputs, caches = [], None
            for token_index in range(max_new_tokens):
                logits, caches = self.decoder(current, valid, steps, memory_kvs, caches)
                scores = logits[:, -1].float()
                scores[:, self.cfg.pad_id] = -torch.inf
                scores[:, self.cfg.bos_id] = -torch.inf
                # 显式长度约束：达到最短长度之前屏蔽 EOS。
                # 默认为 0，保留模型原本的停止行为；这不会提高模型质量。
                if token_index < min_new_tokens:
                    scores[:, self.cfg.eos_id] = -torch.inf
                if temperature == 0:
                    token = scores.argmax(-1)
                else:
                    scores = scores / temperature
                    if top_k:
                        threshold = scores.topk(
                            min(top_k, scores.shape[-1]), dim=-1
                        ).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, -torch.inf)
                    token = torch.multinomial(
                        scores.softmax(-1), 1, generator=generator
                    ).squeeze(1)
                token = torch.where(finished, self.cfg.eos_id, token)
                outputs.append(token)
                finished |= token == self.cfg.eos_id
                current = token[:, None]
                if bool(finished.all()):
                    break
            return torch.stack(outputs, dim=1)
        finally:
            self.train(was_training)
