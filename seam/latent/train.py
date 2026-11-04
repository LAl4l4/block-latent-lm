"""带连续块级潜变量的调度代价实验：潜变量能否救回跳跃词、消掉排序代价？

在 train_plan 的基础上，把离散计划码换成连续潜变量（VAE）：
  - 每块 K 个潜变量槽，各 d_z 维；后验 q(z|块) 是对角高斯，只读本块真实文字；
  - 潜变量槽的生成时刻与计划码相同（plan_visibility）：在该块第一个词之前，
    能读到该块起步前已经写完的词、prompt 和更早的潜变量；
  - 先验 p(z|可见内容) 由潜变量槽的查询流给出，也是对角高斯。

词 CE + KL(q‖p) 是 −log p(x) 的合法上界（ELBO），可直接和无潜变量 causal 的 CE 比较。
训练时 KL 权重在 --kl-warmup 步内从 0 升到 1，以免后验一开始就塌缩到先验；
评估一律用权重 1，所以报告的总 CE 始终是合法上界。
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
from seam.models import LatentLM
from seam.schedules import PLAN_SCHEDULES, plan_visibility
from seam.common import append_record, precision_context


def gaussian_kl(mean_q, logvar_q, mean_p, logvar_p):
    """逐维 KL(N(mean_q, e^logvar_q) ‖ N(mean_p, e^logvar_p))。"""
    return 0.5 * (
        logvar_p - logvar_q
        + (logvar_q.exp() + (mean_q - mean_p) ** 2) / logvar_p.exp()
        - 1
    )


def losses(model, ids, visible, prompt, generator=None):
    mean_q, logvar_q = model.encode(ids, prompt)
    noise = torch.randn(mean_q.shape, device=ids.device, generator=generator)
    z = mean_q + (0.5 * logvar_q).exp() * noise
    word_logits, mean_p, logvar_p = model(ids, visible, prompt, z)
    word_nll = F.cross_entropy(
        word_logits.float().flatten(0, 1), ids[:, prompt:].flatten(), reduction="none"
    ).view(ids.shape[0], model.blocks, model.block_size)
    kl = gaussian_kl(mean_q, logvar_q, mean_p, logvar_p)
    return word_nll, kl.view(ids.shape[0], model.blocks, model.latents, model.latent_dim)


@torch.inference_mode()
def evaluate(model, data, visible, args, device, autocast):
    was = model.training
    model.eval()
    starts = data.random_starts(random.Random(1042), args.eval_windows)
    generator = torch.Generator(device=device).manual_seed(1042)
    word = torch.zeros(args.blocks, args.block_size, dtype=torch.float64)
    kl = torch.zeros(args.blocks, args.latents, args.latent_dim, dtype=torch.float64)
    count = 0
    for begin in range(0, len(starts), args.eval_batch_size):
        ids = data.batch(starts[begin : begin + args.eval_batch_size]).to(device)
        with autocast():
            word_nll, kl_nats = losses(model, ids, visible, args.prompt, generator)
        word += word_nll.sum(0).double().cpu()
        kl += kl_nats.sum(0).double().cpu()
        count += ids.shape[0]
    model.train(was)
    word, kl = word / count, kl / count
    steady_word, steady_kl = word[2:], kl[2:]
    per_dim = steady_kl.mean(0).flatten()
    return {
        "steady_ce": float(steady_word.mean() + steady_kl.sum((-1, -2)).mean() / args.block_size),
        "steady_word_ce": float(steady_word.mean()),
        "steady_kl_per_block": float(steady_kl.sum((-1, -2)).mean()),
        "steady_ce_by_slot": steady_word.mean(0).tolist(),
        "steady_first_ce": float(steady_word[:, 0].mean()),
        # KL 超过 0.01 nat 的维度数：太少说明后验塌缩，潜变量没被用上。
        "active_dims": int((per_dim > 0.01).sum()),
        "windows": count,
    }


def arm_name(args):
    latent = f"latent{args.latents}x{args.latent_dim}"
    if args.free_bits > 0:
        latent += f"fb{args.free_bits:g}"
    if args.schedule == "causal":
        return f"causal-{latent}-s{args.seed}"
    if args.schedule == "reverse" and args.segment > 1:
        return f"reverse{args.stagger}g{args.segment}-{latent}-s{args.seed}"
    return f"{args.schedule}{args.stagger}-{latent}-s{args.seed}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", choices=PLAN_SCHEDULES, required=True)
    p.add_argument("--stagger", type=int)
    p.add_argument("--segment", type=int, default=1)
    p.add_argument("--latents", type=int, default=4, help="每块潜变量槽个数 K")
    p.add_argument("--latent-dim", type=int, default=16, help="每个潜变量槽的维数 d_z")
    p.add_argument("--kl-warmup", type=int, default=1000, help="KL 权重从 0 升到 1 的步数")
    p.add_argument("--free-bits", type=float, default=0.0,
                   help="每块 KL 的免罚额度 λ（nats，均分到 K 个槽）；0 表示不用")
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--out", type=Path, default=Path("runs/schedule-latent"))
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.schedule != "causal" and args.stagger is None:
        p.error("forward/reverse 需要 --stagger")
    if args.schedule == "causal" and args.stagger is not None:
        p.error("causal 不使用 --stagger")
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
    visible = plan_visibility(
        args.schedule, args.prompt, args.blocks, args.block_size, args.latents,
        args.stagger, args.segment,
    ).to(device)
    model = LatentLM(
        metadata["vocab_size"], length, args.blocks, args.block_size, args.latents,
        args.latent_dim, args.dim, args.layers, args.heads, args.ff_dim,
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
                word_nll, kl = losses(model, ids, visible, args.prompt)
                # 每词平均：词 CE + β·KL 摊到块内各词。
                kl_per_block = kl.sum((-1, -2)).mean()
                # free bits：每个潜变量槽的批平均 KL 低于 λ/K 的部分不罚，防止后验塌缩。
                # 只改训练目标；评估仍用真实 KL，报告的总 CE 仍是合法上界。
                kl_slot = kl.sum(-1).mean(0).clamp(min=args.free_bits / args.latents)
                penalty = kl_slot.sum(-1).mean() if args.free_bits > 0 else kl_per_block
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
