"""双流 Transformer（XLNet 式）：内容流按生成时间因果，预测来自只含位置的查询流。

每个词的预测只读到调度允许它看到的词；PlanLM 在此基础上给每块加 K 个离散计划码。
"""


import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.model import RMSNorm, SwiGLU


class TwoStreamLayer(nn.Module):
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff = SwiGLU(dim, ff_dim)

    def forward(self, content, query, mask):
        batch, n, width = content.shape
        x = torch.cat((content, query), 1)
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, x.shape[1], 3, self.heads, width // self.heads)
            .transpose(1, 3)
            .unbind(2)
        )
        # 两个流的 key/value 都只来自内容流；查询流不被任何位置读取。
        value = F.scaled_dot_product_attention(q, k[:, :, :n], v[:, :, :n], attn_mask=mask)
        x = x + self.out(value.transpose(1, 2).reshape(batch, x.shape[1], width))
        x = x + self.ff(self.norm2(x))
        return x[:, :n], x[:, n:]


class TwoStreamLM(nn.Module):
    def __init__(self, vocab_size, length, dim, layers, heads, ff_dim):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(length, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList(
            TwoStreamLayer(dim, heads, ff_dim) for _ in range(layers)
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token.weight
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, visible, targets):
        """visible 为 [N, N]；targets 是要预测的位置下标，返回其 logits。"""
        n = ids.shape[1]
        positions = torch.arange(n, device=ids.device)
        content = self.token(ids) + self.position(positions)
        query = (self.query + self.position(targets)).expand(ids.shape[0], -1, -1)
        # 内容流位置 k 额外看到自己；查询流只看到调度允许的词，看不到自己。
        content_mask = visible | torch.eye(n, dtype=torch.bool, device=ids.device)
        mask = torch.cat((content_mask, visible[targets]), 0)
        for layer in self.layers:
            content, query = layer(content, query, mask)
        return self.output(self.norm(query))


class PlanLM(nn.Module):
    def __init__(self, vocab_size, length, blocks, block_size, codes, choices,
                 dim, layers, heads, ff_dim, encoder_dim=64):
        super().__init__()
        self.blocks, self.block_size = blocks, block_size
        self.codes, self.choices = codes, choices
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(length, dim)
        self.plan_position = nn.Embedding(blocks * codes, dim)
        self.codebook = nn.Parameter(torch.zeros(choices, dim))
        self.query = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList(TwoStreamLayer(dim, heads, ff_dim) for _ in range(layers))
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token.weight
        self.plan_output = nn.Linear(dim, choices, bias=False)
        # 计划编码器：与 ChunkEncoder 同式的块内 MLP，只读本块真实文字。
        self.encoder_token = nn.Embedding(vocab_size, encoder_dim)
        self.encoder = nn.Sequential(
            nn.Linear(block_size * encoder_dim, dim), nn.GELU(),
            nn.Linear(dim, codes * choices),
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.codebook, std=0.02)

    def encode(self, ids, prompt):
        """返回计划码的直通 one-hot [batch, blocks·codes, choices] 与码号。"""
        words = ids[:, prompt:].reshape(ids.shape[0], self.blocks, -1)
        logits = self.encoder(self.encoder_token(words).flatten(2)).float()
        logits = logits.view(ids.shape[0], self.blocks * self.codes, self.choices)
        soft = logits.softmax(-1)
        index = logits.argmax(-1)
        hard = F.one_hot(index, self.choices).to(soft.dtype)
        return hard + soft - soft.detach(), index

    def forward(self, ids, visible, prompt):
        """返回 (词 logits [batch, 生成词数, vocab], 码 logits, 直通码, 码号)。"""
        batch, n = ids.shape
        plan, index = self.encode(ids, prompt)
        slots = torch.arange(self.blocks * self.codes, device=ids.device)
        content = torch.cat((
            self.token(ids) + self.position(torch.arange(n, device=ids.device)),
            (plan.to(self.codebook.dtype) @ self.codebook) + self.plan_position(slots),
        ), 1)
        targets = torch.arange(prompt, n, device=ids.device)
        query = torch.cat((
            self.query + self.position(targets),
            self.query + self.plan_position(slots),
        ), 0).expand(batch, -1, -1)
        total = content.shape[1]
        rows = torch.arange(prompt, total, device=ids.device)
        content_mask = visible | torch.eye(total, dtype=torch.bool, device=ids.device)
        mask = torch.cat((content_mask, visible[rows]), 0)
        for layer in self.layers:
            content, query = layer(content, query, mask)
        query = self.norm(query)
        words = n - prompt
        return self.output(query[:, :words]), self.plan_output(query[:, words:]), plan, index
