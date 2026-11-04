"""CALM 式强 z 加小窗口解码器：z 决定内容时，倒序错开还能不能改善衔接处？

每块一个连续 z（d_z 维，后验只读本块真实文字，先验固定 N(0, I)）。小解码器读
z_t 和窗口内（本块与左边 --window 块）按调度已经写出的词。z 的信息量用 free bits
控制在 --capacity nats/块：批平均 KL 低于这个数的部分不罚，高于的部分按 1 罚，
于是 KL 停在 capacity 附近。这里只测解码器，z 一律取后验，不训练 z 的采样器。

调度除 visibility 的那些外，还有 zonly：解码器一个词都不读，只靠 z。
--gate 让左边块的词走单独的门控注意力，--robust 训练时制造 z 与已写的词不一致（见 corrupt）。
"""


import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from seam.common import GracefulStop
from seam.data import Windows
from seam.models import WindowLM
from seam.schedules import PLAN_SCHEDULES, window_visibility
from seam.common import append_record, precision_context


def corrupt(ids, z, prompt, block_size, vocab_size):
    """抗噪训练：让解码器见到"z 与已写的词对不上"的情况；预测目标仍是真实文字。

    - z：每块以 0.3 的概率加随机方向噪声，长度为 U(0, 0.5) × 本批两随机块 z 的平均距离；
    - 词：每块以 0.15 的概率整块换成批内另一条序列的同一块（通顺但与 z 不符），
      其余生成词以 0.1 的概率换成随机词。prompt 不动。
    编码器始终读真实文字，这里只改解码器的输入。
    """
    batch, blocks = z.shape[:2]
    flat = z.detach().flatten(0, 1)
    unit = (flat - flat[torch.randperm(len(flat), device=z.device)]).norm(dim=-1).mean()
    direction = torch.randn_like(z)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    scale = torch.rand(batch, blocks, 1, device=z.device) * 0.5 * unit
    hit = torch.rand(batch, blocks, 1, device=z.device) < 0.3
    z = z + torch.where(hit, scale * direction, 0)
    words = ids[:, prompt:].view(batch, blocks, block_size)
    swapped = words[torch.randperm(batch, device=ids.device)]
    swap = torch.rand(batch, blocks, 1, device=ids.device) < 0.15
    words = torch.where(swap, swapped, words)
    random_words = torch.randint_like(words, 0, vocab_size)
    words = torch.where(torch.rand(words.shape, device=ids.device) < 0.1, random_words, words)
    inputs = ids.clone()
    inputs[:, prompt:] = words.flatten(1)
    return inputs, z


def losses(model, ids, visible, prompt, generator=None, robust=False):
    mean_q, logvar_q = model.encode(ids, prompt)
    noise = torch.randn(mean_q.shape, device=ids.device, generator=generator)
    z = mean_q + (0.5 * logvar_q).exp() * noise
    inputs = ids
    if robust:
        inputs, z = corrupt(ids, z, prompt, model.block_size, model.token.num_embeddings)
    logits = model(inputs, visible, prompt, z)
    word_nll = F.cross_entropy(
        logits.float().flatten(0, 1), ids[:, prompt:].flatten(), reduction="none"
    ).view(ids.shape[0], model.blocks, model.block_size)
    # KL(q ‖ N(0, I))，逐维。
    kl = 0.5 * (mean_q.square() + logvar_q.exp() - 1 - logvar_q)
    return word_nll, kl


@torch.inference_mode()
def evaluate(model, data, visible, args, device, autocast):
    was = model.training
    model.eval()
    starts = data.random_starts(random.Random(1042), args.eval_windows)
    generator = torch.Generator(device=device).manual_seed(1042)
    word = torch.zeros(args.blocks, args.block_size, dtype=torch.float64)
    kl = torch.zeros(args.blocks, args.latent_dim, dtype=torch.float64)
    hits = torch.zeros(args.blocks, args.block_size, dtype=torch.float64)
    count = 0
    for begin in range(0, len(starts), args.eval_batch_size):
        ids = data.batch(starts[begin : begin + args.eval_batch_size]).to(device)
        with autocast():
            word_nll, kl_nats = losses(model, ids, visible, args.prompt, generator)
            mean_q, _ = model.encode(ids, args.prompt)
            logits = model(ids, visible, args.prompt, mean_q)
        target = ids[:, args.prompt:].view(-1, args.blocks, args.block_size)
        hit = logits.argmax(-1).view_as(target) == target
        word += word_nll.sum(0).double().cpu()
        kl += kl_nats.sum(0).double().cpu()
        hits += hit.sum(0).double().cpu()
        count += ids.shape[0]
    model.train(was)
    word, kl, hits = word / count, kl / count, hits / count
    steady_word, steady_hits = word[2:], hits[2:]
    return {
        "steady_word_ce": float(steady_word.mean()),
        "steady_kl_per_block": float(kl[2:].sum(-1).mean()),
        "steady_ce_by_slot": steady_word.mean(0).tolist(),
        # 用后验均值、teacher forcing 时的 argmax 准确率：z 加上已写的词能还原多少。
        "steady_accuracy": float(steady_hits.mean()),
        "steady_accuracy_by_slot": steady_hits.mean(0).tolist(),
        "active_dims": int((kl[2:].mean(0) > 0.01).sum()),
        "windows": count,
    } | ({"gate_tanh_alpha": [float(torch.tanh(g.alpha)) for g in model.left_layers]}
         if model.gate else {})


def arm_name(args):
    extra = ("-gate" if args.gate else "") + ("-robust" if args.robust else "")
    tail = f"cap{args.capacity:g}-w{args.window}{extra}-s{args.seed}"
    if args.schedule in ("causal", "zonly"):
        return f"{args.schedule}-{tail}"
    if args.schedule == "reverse" and args.segment > 1:
        return f"reverse{args.stagger}g{args.segment}-{tail}"
    return f"{args.schedule}{args.stagger}-{tail}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", choices=PLAN_SCHEDULES + ("zonly",), required=True)
    p.add_argument("--stagger", type=int)
    p.add_argument("--segment", type=int, default=1)
    p.add_argument("--window", type=int, default=2, help="解码器读左边几块已写的词")
    p.add_argument("--capacity", type=float, required=True, help="z 的信息量目标（nats/块）")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--gate", action="store_true", help="左边块走单独的门控注意力（α 初始为 0）")
    p.add_argument("--robust", action="store_true", help="训练时给 z 加噪、替换输入词，见 corrupt")
    p.add_argument("--kl-warmup", type=int, default=1000)
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--out", type=Path, default=Path("runs/schedule-window"))
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--blocks", type=int, default=7)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=2, help="解码器层数")
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=768)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-windows", type=int, default=4096)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.schedule in ("causal", "zonly") and args.stagger is not None:
        p.error(f"{args.schedule} 不使用 --stagger")
    if args.schedule not in ("causal", "zonly") and args.stagger is None:
        p.error("forward/reverse/meet 需要 --stagger")
    if args.segment != 1 and args.schedule != "reverse":
        p.error("只有 reverse 使用 --segment")
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
    visible = window_visibility(
        args.schedule, args.prompt, args.blocks, args.block_size, args.stagger,
        args.segment, args.window,
    ).to(device)
    model = WindowLM(
        metadata["vocab_size"], length, args.blocks, args.block_size, args.latent_dim,
        args.dim, args.layers, args.heads, args.ff_dim, args.encoder_layers, args.gate,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
        fused=device.type == "cuda",
    )
    config = vars(args) | {
        "data": str(args.data),
        "out": str(args.out),
        "parameters": sum(x.numel() for x in model.parameters()),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)

    model.train()
    with GracefulStop() as stop:
        for step in range(1, args.steps + 1):
            begin = time.perf_counter()
            ids = train.batch(train.random_starts(rng, args.batch_size)).to(device)
            beta = min(1.0, step / args.kl_warmup) if args.kl_warmup > 0 else 1.0
            with autocast():
                word_nll, kl = losses(model, ids, visible, args.prompt, robust=args.robust)
                kl_per_block = kl.sum(-1).mean()
                # 块的批平均 KL 低于 capacity 的部分不罚：z 至少装 capacity nats，又不白装更多。
                penalty = kl.sum(-1).mean(0).clamp(min=args.capacity).mean()
                loss = word_nll.mean() + beta * penalty / args.block_size
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
                    "word_ce": float(word_nll.mean()),
                    "kl_per_block": float(kl_per_block), "beta": beta,
                    "lr": lr, "grad_norm": float(norm),
                    "seconds": time.perf_counter() - begin,
                }
                append_record(run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps or stop.requested:
                record = {"kind": "dev", "step": step} | evaluate(
                    model, dev, visible, args, device, autocast
                )
                append_record(run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if stop.requested:
                break
    (run_dir / "final.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    torch.save({"config": config, "model": model.state_dict()}, run_dir / "model.pt")


if __name__ == "__main__":
    main()
