"""阶段 3 的模型：在阶段 2 的双流 Transformer 上加连续块 latent。

LatentLM 每块加 K 个连续潜变量槽（VAE）；WindowLM 是 CALM 式强 z + 小窗口解码器。
"""


import torch
from torch import nn
import torch.nn.functional as F

from common.layers import RMSNorm
from stage2.models import TwoStreamLayer


class LatentLM(nn.Module):
    """每块 K 个连续潜变量槽（各 d_z 维）的双流模型，按 VAE 训练。

    后验 q(z|块) 只读本块真实文字；先验 p(z|可见内容) 由潜变量槽的查询流给出，
    两者都是对角高斯。词 CE + KL 是 −log p(x) 的合法上界（ELBO）。
    """

    def __init__(self, vocab_size, length, blocks, block_size, latents, latent_dim,
                 dim, layers, heads, ff_dim, encoder_dim=64):
        super().__init__()
        self.blocks, self.block_size = blocks, block_size
        self.latents, self.latent_dim = latents, latent_dim
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(length, dim)
        self.latent_position = nn.Embedding(blocks * latents, dim)
        self.latent_in = nn.Linear(latent_dim, dim, bias=False)
        self.query = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList(TwoStreamLayer(dim, heads, ff_dim) for _ in range(layers))
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token.weight
        self.prior_output = nn.Linear(dim, 2 * latent_dim)
        self.encoder_token = nn.Embedding(vocab_size, encoder_dim)
        self.encoder = nn.Sequential(
            nn.Linear(block_size * encoder_dim, dim), nn.GELU(),
            nn.Linear(dim, latents * 2 * latent_dim),
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, ids, prompt):
        """返回后验 (均值, 对数方差)，形状都是 [batch, blocks·latents, d_z]。"""
        words = ids[:, prompt:].reshape(ids.shape[0], self.blocks, -1)
        stats = self.encoder(self.encoder_token(words).flatten(2)).float()
        stats = stats.view(ids.shape[0], self.blocks * self.latents, 2, self.latent_dim)
        return stats[:, :, 0], stats[:, :, 1].clamp(-8, 8)

    def forward(self, ids, visible, prompt, z):
        """z 为 [batch, blocks·latents, d_z]；返回 (词 logits, 先验均值, 先验对数方差)。"""
        batch, n = ids.shape
        slots = torch.arange(self.blocks * self.latents, device=ids.device)
        content = torch.cat((
            self.token(ids) + self.position(torch.arange(n, device=ids.device)),
            self.latent_in(z.to(self.latent_in.weight.dtype)) + self.latent_position(slots),
        ), 1)
        targets = torch.arange(prompt, n, device=ids.device)
        query = torch.cat((
            self.query + self.position(targets),
            self.query + self.latent_position(slots),
        ), 0).expand(batch, -1, -1)
        total = content.shape[1]
        rows = torch.arange(prompt, total, device=ids.device)
        content_mask = visible | torch.eye(total, dtype=torch.bool, device=ids.device)
        mask = torch.cat((content_mask, visible[rows]), 0)
        for layer in self.layers:
            content, query = layer(content, query, mask)
        query = self.norm(query)
        words = n - prompt
        prior = self.prior_output(query[:, words:]).float()
        prior = prior.view(batch, -1, 2, self.latent_dim)
        return self.output(query[:, :words]), prior[:, :, 0], prior[:, :, 1].clamp(-8, 8)


class GatedLeftAttention(nn.Module):
    """读左边块的门控注意力（类似 Flamingo 的 gated cross-attention）。

    输出乘 tanh(α)·σ(w·x)：α 初始为 0，训练开始时完全不读左边；σ 项随内容变化，
    让模型能按当前位置决定信多少左边。key/value 只来自内容流。
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, content, query, mask):
        batch, n, width = content.shape
        x = torch.cat((content, query), 1)
        h = self.norm(x)
        q, k, v = (
            self.qkv(h).view(batch, x.shape[1], 3, self.heads, width // self.heads)
            .transpose(1, 3).unbind(2)
        )
        value = F.scaled_dot_product_attention(q, k[:, :, :n], v[:, :, :n], attn_mask=mask)
        update = self.out(value.transpose(1, 2).reshape(batch, x.shape[1], width))
        x = x + torch.tanh(self.alpha) * torch.sigmoid(self.gate(h)) * update
        return x[:, :n], x[:, n:]


class WindowLM(nn.Module):
    """CALM 式：每块一个强 z 决定内容，小解码器读 z_t 和窗口内已写的词管表面衔接。

    编码器是块内双向的小 Transformer，只读本块真实文字，输出对角高斯后验；
    先验固定为 N(0, I)（z 由外部采样器给出，这里只测解码器）。
    z_t 只进入第 t 块的查询流：直接加到查询上，并作为一个只给本块查询读的内容位置；
    词的内容流不读 z，所以 z_t 不会经由第 t 块的词流到别的块。
    """

    def __init__(self, vocab_size, length, blocks, block_size, latent_dim,
                 dim, layers, heads, ff_dim, encoder_layers=2, gate=False):
        super().__init__()
        self.blocks, self.block_size, self.latent_dim = blocks, block_size, latent_dim
        self.gate = gate
        if gate:
            self.left_layers = nn.ModuleList(GatedLeftAttention(dim, heads) for _ in range(layers))
            self.sink = nn.Parameter(torch.zeros(dim))
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(length, dim)
        self.latent_in = nn.Linear(latent_dim, dim, bias=False)
        self.latent_token = nn.Parameter(torch.zeros(dim))
        self.query = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList(TwoStreamLayer(dim, heads, ff_dim) for _ in range(layers))
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token.weight
        self.encoder_position = nn.Embedding(block_size, dim)
        self.encoder_layers = nn.ModuleList(
            TwoStreamLayer(dim, heads, ff_dim) for _ in range(encoder_layers))
        self.encoder_norm = RMSNorm(dim)
        self.posterior = nn.Linear(block_size * dim, 2 * latent_dim)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, ids, prompt):
        """返回后验 (均值, 对数方差)，形状都是 [batch, blocks, d_z]；每块只读本块文字。"""
        batch = ids.shape[0]
        words = ids[:, prompt:].reshape(batch * self.blocks, self.block_size)
        x = self.token(words) + self.encoder_position.weight
        empty = x[:, :0]
        full = torch.ones(self.block_size, self.block_size, dtype=torch.bool, device=ids.device)
        for layer in self.encoder_layers:
            x, _ = layer(x, empty, full)
        stats = self.posterior(self.encoder_norm(x).flatten(1)).float()
        stats = stats.view(batch, self.blocks, 2, self.latent_dim)
        return stats[:, :, 0], stats[:, :, 1].clamp(-8, 8)

    def forward(self, ids, visible, prompt, z):
        """visible 为词之间的 [N, N] 可见矩阵（已含窗口）；z 为 [batch, blocks, d_z]。"""
        batch, n = ids.shape
        device = ids.device
        latent = self.latent_in(z.to(self.latent_in.weight.dtype))
        parts = [self.token(ids) + self.position(torch.arange(n, device=device)),
                 latent + self.latent_token]
        if self.gate:
            parts.append(self.sink.expand(batch, 1, -1))
        content = torch.cat(parts, 1)
        targets = torch.arange(prompt, n, device=device)
        block = (targets - prompt) // self.block_size
        query = self.query + self.position(targets) + latent[:, block]
        total = content.shape[1]
        own = torch.zeros(len(targets), total - n, dtype=torch.bool, device=device)
        own[torch.arange(len(targets), device=device), block] = True
        eye = torch.eye(total, dtype=torch.bool, device=device)

        def stream_mask(words):
            """词之间的可见矩阵 → 内容流加查询流的完整 mask。"""
            content_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            content_mask[:n, :n] = words
            return torch.cat((content_mask | eye, torch.cat((words[targets], own), 1)), 0)

        if not self.gate:
            mask = stream_mask(visible)
            for layer in self.layers:
                content, query = layer(content, query, mask)
            return self.output(self.norm(query))
        # gate：本块的词（和 z）走原注意力，左边块的词另走一路带门控的注意力。
        # 两路的可见集合不相交、并起来正好是 visible；sink 让左边一个词都看不到的行也有可读位置。
        position = torch.arange(n, device=device)
        word_block = torch.where(position >= prompt, (position - prompt) // self.block_size, -1)
        same = word_block[:, None] == word_block[None, :]
        local = stream_mask(visible & same)
        left = torch.zeros(total + len(targets), total, dtype=torch.bool, device=device)
        left[:n, :n] = visible & ~same
        left[total:, :n] = (visible & ~same)[targets]
        left[:, -1] = True
        for layer, gated in zip(self.layers, self.left_layers):
            content, query = gated(content, query, left)
            content, query = layer(content, query, local)
        return self.output(self.norm(query))
