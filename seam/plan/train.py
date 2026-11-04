"""带块级计划的调度代价实验：块内顺序在有计划时还重要吗？

在 schedule_mask_study 的双流模型上，给每个生成块加 K 个离散计划码：
  - 编码器读取该块的真实文字，输出 K 个 V 选 1 的码（argmax，直通梯度）；
  - 计划码在该块第一个词之前生成：它能读到的，是该块起步前已经写完的词、
    prompt 和更早块的计划；该块及右侧块的所有词都能读到它；
  - 模型同时预测计划码（先验）和词。

码是文字的确定函数，所以 码 NLL + 词 CE 是 −log p(x) 的合法上界，可直接和
无计划 causal 的自然语料 CE 比较。每块计划最多含 K·ln V nat 信息。比较各调度
在同一计划预算下的"总代价"和"只算词的代价"，就能看出排序代价有多少被计划吸收。
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
from seam.models import PlanLM
from seam.schedules import PLAN_SCHEDULES, plan_visibility
from seam.common import append_record, precision_context


def losses(model, ids, visible, prompt):
    word_logits, plan_logits, plan, index = model(ids, visible, prompt)
    word_nll = F.cross_entropy(
        word_logits.float().flatten(0, 1), ids[:, prompt:].flatten(), reduction="none"
    ).view(ids.shape[0], model.blocks, model.block_size)
    # 码 NLL 只训练先验：若也回传给编码器，训练早期会把码推成常数（塌缩）。
    # 编码器只由词 CE 训练；评估时码是确定的，总 NLL 仍是合法上界。
    plan_nll = -(plan.detach() * plan_logits.float().log_softmax(-1)).sum(-1)
    return word_nll, plan_nll.view(ids.shape[0], model.blocks, model.codes), index


@torch.inference_mode()
def evaluate(model, data, visible, args, device, autocast):
    was = model.training
    model.eval()
    starts = data.random_starts(random.Random(1042), args.eval_windows)
    word = torch.zeros(args.blocks, args.block_size, dtype=torch.float64)
    plan = torch.zeros(args.blocks, args.codes, dtype=torch.float64)
    usage = torch.zeros(args.codes, args.choices, dtype=torch.float64)
    count = 0
    for begin in range(0, len(starts), args.eval_batch_size):
        ids = data.batch(starts[begin : begin + args.eval_batch_size]).to(device)
        with autocast():
            word_nll, plan_nll, index = losses(model, ids, visible, args.prompt)
        word += word_nll.sum(0).double().cpu()
        plan += plan_nll.sum(0).double().cpu()
        codes = index.view(ids.shape[0], args.blocks, args.codes)[:, 2:]
        usage += F.one_hot(codes, args.choices).sum((0, 1)).double().cpu()
        count += ids.shape[0]
    model.train(was)
    word, plan = word / count, plan / count
    steady_word, steady_plan = word[2:], plan[2:]
    p = usage / usage.sum(-1, keepdim=True)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return {
        "steady_ce": float(steady_word.mean() + steady_plan.sum(-1).mean() / args.block_size),
        "steady_word_ce": float(steady_word.mean()),
        "steady_plan_nll_per_block": float(steady_plan.sum(-1).mean()),
        "steady_ce_by_slot": steady_word.mean(0).tolist(),
        "steady_first_ce": float(steady_word[:, 0].mean()),
        "code_usage_entropy": entropy.tolist(),
        "plan_capacity_nats": args.codes * math.log(args.choices),
        "windows": count,
    }


def arm_name(args):
    plan = f"plan{args.codes}x{args.choices}"
    if args.schedule == "causal":
        return f"causal-{plan}-s{args.seed}"
    if args.schedule == "reverse" and args.segment > 1:
        return f"reverse{args.stagger}g{args.segment}-{plan}-s{args.seed}"
    return f"{args.schedule}{args.stagger}-{plan}-s{args.seed}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", choices=PLAN_SCHEDULES, required=True)
    p.add_argument("--stagger", type=int)
    p.add_argument("--segment", type=int, default=1)
    p.add_argument("--codes", type=int, default=4, help="每块计划码个数 K")
    p.add_argument("--choices", type=int, default=16, help="每个码的取值数 V")
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--out", type=Path, default=Path("runs/schedule-plan"))
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
        args.schedule, args.prompt, args.blocks, args.block_size, args.codes,
        args.stagger, args.segment,
    ).to(device)
    model = PlanLM(
        metadata["vocab_size"], length, args.blocks, args.block_size, args.codes,
        args.choices, args.dim, args.layers, args.heads, args.ff_dim,
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
            with autocast():
                word_nll, plan_nll, _ = losses(model, ids, visible, args.prompt)
                # 每词平均的总 NLL：词 CE + 计划码 NLL 摊到块内各词。
                loss = word_nll.mean() + plan_nll.sum(-1).mean() / args.block_size
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
                    "plan_nll_per_block": float(plan_nll.sum(-1).mean()),
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
