"""固定验证窗口，测量块内 token 头对主干上下文的依赖。"""

import argparse
import json
import hashlib
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from download_data import sha256
from latent_ar.checkpoints import (
    CALM_TOKEN_ARCHITECTURE,
    checkpoint_architecture,
    load_model,
)
from latent_ar.data import StoryDataset, collate
from train import precision_context, select_device, to_device


def block_inputs(model, batch):
    contexts, means, _, valid = model.teacher_contexts(**batch)
    labels = batch["labels"]
    padded = F.pad(
        labels, (0, means.shape[1] * model.cfg.chunk_size - labels.shape[1]), value=-100
    )
    blocks = padded.reshape(*valid.shape, model.cfg.chunk_size)
    previous = torch.full_like(blocks, model.cfg.pad_id)
    previous[:, :, 0] = model.cfg.bos_id
    previous[:, :, 1:] = blocks[:, :, :-1].masked_fill(
        blocks[:, :, :-1] == -100, model.cfg.pad_id
    )
    return contexts, blocks, previous, valid


def shuffled_contexts(contexts, valid, generator):
    """只在相同目标块序号且双方有效的行之间打乱；单行位置不计分。"""
    shuffled = contexts.clone()
    eligible = torch.zeros_like(valid)
    for position in range(valid.shape[1]):
        rows = valid[:, position].nonzero(as_tuple=True)[0]
        if rows.numel() < 2:
            continue
        order = rows[torch.randperm(rows.numel(), generator=generator).to(rows.device)]
        shuffled[order, position] = contexts[order.roll(1), position]
        eligible[rows, position] = True
    return shuffled, eligible


def score_blocks(model, contexts, blocks, previous, eligible):
    logits = model.latent_head(contexts, previous)
    losses = F.cross_entropy(
        logits.float().flatten(0, 2),
        blocks.flatten(),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(blocks)
    mask = blocks.ne(-100) & eligible[..., None]
    return (losses * mask).sum((0, 1)), mask.sum((0, 1))


def gradient_norm(module):
    terms = [
        p.grad.detach().float().square().sum()
        for p in module.parameters()
        if p.grad is not None
    ]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def parameter_change(model, source):
    current = model.state_dict()
    result = {}
    for group in ("history_embedding", "history_compressor", "backbone", "codec"):
        names = [name for name in current if name.startswith(group + ".")]
        delta = base = count = 0
        for name in names:
            if name not in source or source[name].shape != current[name].shape:
                raise ValueError(f"初始检查点缺少匹配参数：{name}")
            a, b = (
                current[name].detach().float().cpu(),
                source[name].detach().float().cpu(),
            )
            delta += float((a - b).square().sum())
            base += float(b.square().sum())
            count += a.numel()
        result[group] = {
            "parameters": count,
            "delta_rms": (delta / count) ** 0.5,
            "initial_rms": (base / count) ** 0.5,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--data", default=Path("data/prepared"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.windows < 2 or args.batch_size < 2:
        parser.error("窗口和批量均至少为 2")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_architecture(saved) != CALM_TOKEN_ARCHITECTURE:
        raise ValueError("检查点不是 CALM token 头")
    manifest_hash = sha256(args.data / "metadata.json")
    fingerprints = {f"{s}.{ext}":sha256(args.data/f"{s}.{ext}") for s in ("train","dev") for ext in ("bin","offsets.npy")}
    fingerprints["metadata"] = manifest_hash
    content_hash = hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
    if saved["dataset_hash"] not in {manifest_hash,content_hash}:
        raise ValueError("验证集元数据与训练集不匹配")
    device = select_device(args.device)
    saved.pop("optimizer",None)
    model = load_model(saved, device).eval()
    tr = saved["config"]["training"]
    dev = StoryDataset(
        args.data,
        "dev",
        tr["min_prompt_tokens"],
        tr["max_prompt_tokens"],
        tr["max_target_tokens"],
        model.cfg.pad_id,
    )
    if args.windows > dev.size:
        raise ValueError("请求的验证窗口数超过数据集")
    baseline = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=True
    )
    baseline.pop("optimizer",None)
    changes = parameter_change(model, baseline["model"])
    del baseline
    generator = torch.Generator().manual_seed(args.seed)
    autocast = precision_context(device, tr["precision"])
    totals = {
        name: torch.zeros(model.cfg.chunk_size, dtype=torch.float64)
        for name in ("matched", "shuffled", "zero", "count")
    }
    first_batch = None
    with torch.no_grad():
        for start in range(0, args.windows, args.batch_size):
            examples = [
                dev.example(i, random.Random(args.seed + i))
                for i in range(start, min(start + args.batch_size, args.windows))
            ]
            batch = to_device(collate(examples, model.cfg.pad_id), device)
            if first_batch is None:
                first_batch = batch
            with autocast():
                contexts, blocks, previous, valid = block_inputs(model, batch)
                shuffled, eligible = shuffled_contexts(contexts, valid, generator)
                for name, h in (
                    ("matched", contexts),
                    ("shuffled", shuffled),
                    ("zero", torch.zeros_like(contexts)),
                ):
                    loss, count = score_blocks(model, h, blocks, previous, eligible)
                    totals[name] += loss.double().cpu()
                totals["count"] += count.double().cpu()
    # cuDNN GRU 反向需要 training 模式；本头只有一层、无dropout。
    # CalmModel.train仍让冻结VAE保持eval，不改变上面的验证分数。
    model.train()
    model.zero_grad(set_to_none=True)
    with autocast():
        contexts, blocks, previous, valid = block_inputs(model, first_batch)
        sums, counts = score_blocks(model, contexts, blocks, previous, valid)
        loss = sums.sum() / counts.sum()
    loss.backward()
    gradients = {
        name: gradient_norm(getattr(model, name))
        for name in (
            "history_embedding",
            "history_compressor",
            "backbone",
            "latent_head",
        )
    }
    gradients["vae_has_gradient"] = any(
        p.grad is not None for p in model.codec.parameters()
    )
    count = totals["count"]
    positions = []
    for i in range(model.cfg.chunk_size):
        n = int(count[i])
        values = {
            name: float(totals[name][i] / n) if n else None
            for name in ("matched", "shuffled", "zero")
        }
        positions.append({"slot": i + 1, "tokens": n, **values})
    n = int(count.sum())
    if not n:
        raise ValueError("没有可配对的有效块")
    report = {
        "checkpoint": str(args.checkpoint),
        "initial_checkpoint": str(args.initial_checkpoint),
        "windows": args.windows,
        "seed": args.seed,
        "paired_tokens": n,
        "cross_entropy": {
            name: float(totals[name].sum() / n)
            for name in ("matched", "shuffled", "zero")
        },
        "per_slot": positions,
        "gradient_norms": gradients,
        "parameter_change": changes,
        "note": "块内真实前词 teacher forcing；打乱只配对同块序号的有效行；参数变化不代表质量。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
