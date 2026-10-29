"""V2：自由 latent 主干 + 只读当前 latent 的块内双向解码器。

主干保留 causal attention 与可微 KV cache；decoder 没有 cross-attention、
文本 token 输入或跨块缓存。每个 latent 展开四个位置，在块内双向计算。
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
    decoder_type: str = "block_parallel"
    chunk_size: int = 4
    encoder_dim: int = 256
    dim: int = 512
    layers: int = 8
    heads: int = 8
    ff_dim: int = 1536
    decoder_dim: int = 256
    decoder_layers: int = 1
    decoder_heads: int = 4
    decoder_ff_dim: int = 768

    def __post_init__(self):
        if self.decoder_type != "block_parallel":
            raise ValueError("V2 requires decoder_type=block_parallel")
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


class LocalSelfAttention(nn.Module):
    """每个 latent 块单独做双向 attention，四个位置互相可见。

    输入的 batch 维已经合并了故事编号和块编号，因此不同块不可能互相
    attention。这里没有因果 mask，也没有 decoder KV cache。
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, slots, width = x.shape
        q, k, v = (
            self.qkv(x).view(batch, slots, 3, self.heads, width // self.heads).unbind(2)
        )
        value = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None,
            is_causal=False,
            dropout_p=0.0,
        )
        return self.out(value.transpose(1, 2).reshape(batch, slots, width))


class LocalDecoderBlock(nn.Module):
    """块内双向 attention 加一个小前馈层；两处都使用残差连接。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.decoder_dim)
        self.norm2 = RMSNorm(cfg.decoder_dim)
        self.attention = LocalSelfAttention(cfg.decoder_dim, cfg.decoder_heads)
        self.ff = SwiGLU(cfg.decoder_dim, cfg.decoder_ff_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.norm1(hidden))
        return hidden + self.ff(self.norm2(hidden))


class TextDecoder(nn.Module):
    """只将当前 z_t 还原成四个 token 分布，不接收任何文本历史。

    [batch, latent数, 512]
        → 展开为 [batch, latent数, 4, 256]
        → 每块独立双向 attention
        → [batch, latent数, 4, vocab_size]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.expand = nn.Linear(cfg.dim, cfg.chunk_size * cfg.decoder_dim)
        self.position = nn.Parameter(torch.empty(cfg.chunk_size, cfg.decoder_dim))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            [LocalDecoderBlock(cfg) for _ in range(cfg.decoder_layers)]
        )
        self.norm = RMSNorm(cfg.decoder_dim)
        self.output = nn.Linear(cfg.decoder_dim, cfg.vocab_size, bias=False)

    def forward(self, latents: Tensor) -> Tensor:
        batch, steps, _ = latents.shape
        # 把“故事 × 块”合成 batch，物理上隔离不同块的 attention。
        hidden = self.expand(latents).reshape(
            batch * steps, self.cfg.chunk_size, self.cfg.decoder_dim
        )
        hidden = hidden + self.position
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(self.norm(hidden))
        return logits.reshape(batch, steps, self.cfg.chunk_size, self.cfg.vocab_size)


class LatentARModel(nn.Module):
    """V2：主干负责跨块信息，decoder 只接收每个块自己的生成 latent。"""

    architecture = "latent_block_v2"

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

    def rollout(self, prompt: Tensor, prompt_lengths: Tensor, steps: int) -> Tensor:
        """自由展开 latent 链，返回 [batch, steps, 512]，不包含 prompt 编码。

        整条链保留计算图。后续步骤只输入自己产生的 latent；解码后的词
        不回灌主干。主干处理完 prompt 后，decoder 也不能直接读取 prompt。
        """
        if steps < 1 or bool((prompt_lengths < 1).any()):
            raise ValueError("Need a nonempty prompt and a positive rollout length")
        prompt_latents, prompt_valid = self.encoder(prompt, prompt_lengths)
        lengths = prompt_valid.sum(-1)
        batch, prompt_width, _ = prompt_latents.shape
        positions = torch.arange(prompt_width, device=prompt.device).expand(batch, -1)
        hidden, caches = self.backbone(prompt_latents, positions, prompt_valid)
        last = hidden[torch.arange(batch, device=prompt.device), lengths - 1]
        valid = prompt_valid
        generated = []

        for step in range(steps):
            z = self.latent_head(last)
            generated.append(z)
            if step + 1 < steps:
                valid = torch.cat(
                    [
                        valid,
                        torch.ones(batch, 1, device=prompt.device, dtype=torch.bool),
                    ],
                    dim=1,
                )
                hidden, caches = self.backbone(
                    z[:, None], (lengths + step)[:, None], valid, caches
                )
                last = hidden[:, 0]
        return torch.stack(generated, dim=1)

    def decode(self, latents: Tensor, output_tokens: int) -> Tensor:
        """各块并行解码，再裁掉最后一个块超出目标长度的槽位。"""
        if not 1 <= output_tokens <= latents.shape[1] * self.cfg.chunk_size:
            raise ValueError("output_tokens exceeds the supplied latent budget")
        return self.decoder(latents).flatten(1, 2)[:, :output_tokens]

    def forward(
        self,
        prompt: Tensor,
        prompt_lengths: Tensor,
        output_tokens: int,
        use_latents: bool = True,
    ) -> Tensor:
        """只接收所需输出长度，不接收真实续写或 decoder_input。"""
        if not isinstance(output_tokens, int) or output_tokens < 1:
            raise ValueError("output_tokens must be a positive integer")
        steps = math.ceil(output_tokens / self.cfg.chunk_size)
        if use_latents:
            latents = self.rollout(prompt, prompt_lengths, steps)
        else:
            # 兼容诊断开关：全零 latent。此时没有任何 prompt 信息进入 decoder，
            # 因而它不是旧版那种仍能读取 prompt 的独立条件语言模型基线。
            latents = self.decoder.expand.weight.new_zeros(
                prompt.shape[0], steps, self.cfg.dim
            )
        return self.decode(latents, output_tokens)

    @torch.inference_mode()
    def generate(
        self,
        prompt: Tensor,
        prompt_lengths: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 50,
        use_latents: bool = True,
        generator: torch.Generator | None = None,
        min_new_tokens: int = 0,
    ) -> Tensor:
        """同步参考生成；默认贪心，先检验 latent 是否能确定一组明确的词。

        四个位置的 logits 并行计算。采样仅选择输出，任何已选 token 都
        不再输入 decoder 或主干。EOS 后的槽位不再输出。
        """
        if max_new_tokens < 1 or temperature < 0 or top_k < 0:
            raise ValueError("Invalid generation settings")
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens must be between 0 and max_new_tokens")
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(
                prompt, prompt_lengths, max_new_tokens, use_latents=use_latents
            )
            batch = prompt.shape[0]
            finished = torch.zeros(batch, device=prompt.device, dtype=torch.bool)
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
                    block_tokens = scores.argmax(-1)
                else:
                    scores = scores / temperature
                    if top_k:
                        threshold = scores.topk(
                            min(top_k, scores.shape[-1]), dim=-1
                        ).values[..., -1:]
                        scores = scores.masked_fill(scores < threshold, -torch.inf)
                    flat = scores.softmax(-1).reshape(-1, self.cfg.vocab_size)
                    block_tokens = torch.multinomial(
                        flat, 1, generator=generator
                    ).reshape(batch, -1)

                # 四个位置已并行得到结果；逐位置检查只是为了按第一个 EOS 停止。
                for offset in range(block_tokens.shape[1]):
                    token = torch.where(
                        finished, self.cfg.eos_id, block_tokens[:, offset]
                    )
                    outputs.append(token)
                    finished |= token == self.cfg.eos_id
                    if bool(finished.all()):
                        break
                if bool(finished.all()):
                    break
            return torch.stack(outputs, dim=1)
        finally:
            self.train(was_training)
