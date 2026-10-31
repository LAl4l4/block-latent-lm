"""调度代价实验：只用 attention mask 表达"谁能看到谁"，与块级架构无关。

token 级双流 Transformer（XLNet 式）：内容流按生成时间因果，预测来自只含位置的
查询流，所以每个词的预测只读到调度允许它看到的词。各调度同预算从头训练，
自然语料 CE 之差就是该调度在信息上的代价下界（任何满足同一可见关系的架构都
不可能更好，只会因结构限制更差）。

序列 = prompt（完全已知，内部普通因果）+ blocks 个 L 词生成块。调度：
  causal              普通从左到右（上界）
  delayed             第 t 块只看到 ≤t-2 块与本块已生成前缀（现有延迟反馈的条件）
  forward --stagger s 块内正序，相邻块错开 s 步开始
  reverse --stagger s 块内倒序（先生成第 L 个词），相邻块错开 s 步开始
  reverse --stagger s --segment g  按段倒序：块切成长 g 的段，段间倒序、段内正序
                                  （g=1 即逐词倒序）
  landing             PSF 式：所有块同时正序开始，第 2 个词起额外读右侧块首词
s=0 时所有块同时开始（完全并行）。除 landing 外，所有调度都不读右侧的块。生成位置 i 能看到 k，当且仅当 k 生成得更早且所在块
不在 i 右边；reverse 下丢掉的词对距离一律 ≥ L+s，见
docs/idea-reverse-staggered-decoding.md。
"""

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from latent_ar.control import GracefulStop
from latent_ar.model import RMSNorm, SwiGLU
from train import append_record, precision_context

SCHEDULES = ("causal", "delayed", "forward", "reverse", "landing")


def visibility(schedule, prompt, blocks, block_size, stagger=None, segment=1):
    """返回 [N, N] 布尔矩阵 V：V[i, k] 表示生成位置 i 时可以读到位置 k 的词。"""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if prompt < 1 or blocks < 1 or block_size < 1:
        raise ValueError("prompt, blocks and block_size must be positive")
    n = prompt + blocks * block_size
    position = torch.arange(n)
    generated = position >= prompt
    block = torch.where(generated, (position - prompt) // block_size, -1)
    slot = torch.where(generated, (position - prompt) % block_size, 0)
    if schedule == "delayed":
        same = block[:, None] == block[None, :]
        allowed = (block[None, :] <= block[:, None] - 2) | (
            same & (slot[None, :] < slot[:, None])
        )
    else:
        if schedule == "causal":
            order, step = slot, block_size
        elif schedule == "landing":
            order, step = slot, 0
        else:
            if stagger is None or not 0 <= stagger <= block_size:
                raise ValueError("stagger must be in [0, block_size]")
            if schedule == "forward":
                order = slot
            else:
                if segment < 1 or block_size % segment:
                    raise ValueError("segment must divide block_size")
                # 段间倒序：最后一段最先写，块头所在的第一段最后写；段内仍从左到右。
                last = block_size // segment - 1
                order = (last - slot // segment) * segment + slot % segment
            step = stagger
        clock = block * step + order
        allowed = (clock[None, :] < clock[:, None]) & (block[None, :] <= block[:, None])
        if schedule == "landing":
            # 着陆目标：右侧块首词最先生成，本块其余词都能读到它。
            allowed |= (
                (block[None, :] == block[:, None] + 1)
                & (slot[None, :] == 0)
                & (slot[:, None] >= 1)
            )
    rows, keys = generated[:, None], generated[None, :]
    # prompt 完全已知：生成位置都能读；prompt 内部普通因果，读不到任何生成位置。
    allowed = torch.where(rows & ~keys, True, allowed)
    allowed = torch.where(~rows, ~keys & (position[None, :] < position[:, None]), allowed)
    return allowed


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


class Windows:
    """在拼接后的故事 token 流上取定长窗口；故事之间已有 EOS 分隔。"""

    def __init__(self, path, length):
        self.tokens = np.memmap(path, mode="r", dtype="<u4")
        self.length = length
        if len(self.tokens) <= length:
            raise ValueError("Token stream shorter than one window")

    def batch(self, starts):
        return torch.from_numpy(
            np.stack([self.tokens[s : s + self.length] for s in starts]).astype(np.int64)
        )

    def random_starts(self, rng, count):
        return [rng.randrange(len(self.tokens) - self.length) for _ in range(count)]


def arm_name(args):
    mix = f"mix{round(args.mix_causal * 100)}" if args.mix_causal else ""
    if args.schedule in ("causal", "delayed", "landing"):
        return f"{args.schedule}{mix}-s{args.seed}"
    if args.schedule == "reverse" and args.segment > 1:
        return f"reverse{args.stagger}g{args.segment}{mix}-s{args.seed}"
    return f"{args.schedule}{args.stagger}{mix}-s{args.seed}"


@torch.inference_mode()
def evaluate(model, data, visible, targets, args, device, autocast):
    was = model.training
    model.eval()
    starts = data.random_starts(random.Random(1042), args.eval_windows)
    blocks, size = args.blocks, args.block_size
    nll = torch.zeros(blocks, size, dtype=torch.float64)
    count = 0
    for begin in range(0, len(starts), args.eval_batch_size):
        ids = data.batch(starts[begin : begin + args.eval_batch_size]).to(device)
        with autocast():
            logits = model(ids, visible, targets)
        values = F.cross_entropy(
            logits.float().flatten(0, 1), ids[:, targets].flatten(), reduction="none"
        ).view(ids.shape[0], blocks, size)
        nll += values.sum(0).double().cpu()
        count += ids.shape[0]
    model.train(was)
    per = nll / count
    steady = per[2:]  # 第 0、1 个生成块仍直接读 prompt，不算稳态
    return {
        "ce": float(per.mean()),
        "steady_ce": float(steady.mean()),
        "steady_ce_by_slot": steady.mean(0).tolist(),
        "steady_first_ce": float(steady[:, 0].mean()),
        "ce_by_block": per.mean(1).tolist(),
        "windows": count,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", choices=SCHEDULES, required=True)
    p.add_argument("--stagger", type=int)
    p.add_argument("--segment", type=int, default=1, help="reverse：段长，段内正序")
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--out", type=Path, default=Path("runs/schedule-mask"))
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--blocks", type=int, default=7)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=768)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-windows", type=int, default=4096)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--mix-causal", type=float, default=0.0,
                   help="每步以此概率改用 causal mask 训练（同一模型），评估仍用目标调度")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.schedule in ("forward", "reverse") and args.stagger is None:
        p.error("forward/reverse 需要 --stagger")
    if args.segment != 1 and args.schedule != "reverse":
        p.error("只有 reverse 使用 --segment")
    if args.schedule in ("causal", "delayed", "landing") and args.stagger is not None:
        p.error("causal/delayed/landing 不使用 --stagger")
    run_dir = args.out / arm_name(args)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"必须使用新的实验目录：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    autocast = precision_context(device, "bf16")
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    length = args.prompt + args.blocks * args.block_size
    train = Windows(args.data / "train.bin", length)
    dev = Windows(args.data / "dev.bin", length)
    visible = visibility(
        args.schedule, args.prompt, args.blocks, args.block_size, args.stagger,
        args.segment,
    ).to(device)
    causal = visibility("causal", args.prompt, args.blocks, args.block_size).to(device)
    mixer = random.Random(args.seed + 1)
    targets = torch.arange(args.prompt, length, device=device)
    model = TwoStreamLM(
        metadata["vocab_size"], length, args.dim, args.layers, args.heads, args.ff_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
        fused=device.type == "cuda",
    )
    config = vars(args) | {
        "data": str(args.data),
        "out": str(args.out),
        "parameters": sum(x.numel() for x in model.parameters()),
        "visible_pairs": int(visible[args.prompt :].sum()),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)

    model.train()
    with GracefulStop() as stop:
        for step in range(1, args.steps + 1):
            begin = time.perf_counter()
            ids = train.batch(train.random_starts(rng, args.batch_size)).to(device)
            with autocast():
                mask = causal if mixer.random() < args.mix_causal else visible
                logits = model(ids, mask, targets)
                loss = F.cross_entropy(
                    logits.float().flatten(0, 1), ids[:, targets].flatten()
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            progress = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * (
                step / args.warmup
                if step <= args.warmup
                else 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if step % 100 == 0 or step == 1:
                record = {
                    "kind": "train", "step": step, "loss": float(loss),
                    "lr": lr, "grad_norm": float(norm),
                    "seconds": time.perf_counter() - begin,
                }
                append_record(run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps or stop.requested:
                record = {"kind": "dev", "step": step} | evaluate(
                    model, dev, visible, targets, args, device, autocast
                )
                append_record(run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if stop.requested:
                break
    (run_dir / "final.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    torch.save({"config": config, "model": model.state_dict()}, run_dir / "model.pt")


if __name__ == "__main__":
    main()
