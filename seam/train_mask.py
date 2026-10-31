"""调度代价实验：只用 attention mask 表达"谁能看到谁"，与块级架构无关。

token 级双流 Transformer（XLNet 式）：内容流按生成时间因果，预测来自只含位置的
查询流，所以每个词的预测只读到调度允许它看到的词。各调度同预算从头训练，
自然语料 CE 之差就是该调度在信息上的代价下界（任何满足同一可见关系的架构都
不可能更好，只会因结构限制更差）。

调度定义见 seam/schedules.py。
"""


import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from latent_ar.control import GracefulStop
from seam.data import Windows
from seam.models import TwoStreamLM
from seam.schedules import SCHEDULES, visibility
from train import append_record, precision_context


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
    if args.schedule in ("forward", "reverse", "meet", "alternate") and args.stagger is None:
        p.error("forward/reverse/meet/alternate 需要 --stagger")
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
