"""打印样本原文，标出块边界和评委眼里的断点，用于定性阅读。

prompt、采样 seed 和批次划分与 sample_eval 相同，所以前几条样本与它评分的样本逐词一致。
每块之间插入 ‖；评委 NLL 超过阈值的词用 ⟦⟧ 括起来，并统计这些词落在块内哪个位置。
"""


import argparse
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from common.data import Windows
from stage2.mask.sample_eval import judge, load, sample
from common.training import precision_context


def render(tokenizer, ids, nll, prompt, size, threshold):
    """逐块解码；高 NLL 的词单独解码后加括号。"""
    blocks = []
    for b in range(len(nll)):
        words = ids[prompt + b * size : prompt + (b + 1) * size]
        pieces, run = [], []
        for slot, word in enumerate(words):
            if nll[b][slot] > threshold:
                if run:
                    pieces.append(tokenizer.decode(run))
                    run = []
                pieces.append("⟦" + tokenizer.decode([word]) + "⟧")
            else:
                run.append(word)
        if run:
            pieces.append(tokenizer.decode(run))
        blocks.append("".join(pieces))
    return " ‖ ".join(blocks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/schedule-mask"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--judge", default="causal-s43")
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--prompts", type=int, default=512, help="与 sample_eval 相同，保证样本一致")
    p.add_argument("--show", type=int, default=6, help="打印前几条")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2042)
    p.add_argument("--threshold", type=float, default=8.0, help="评委 NLL 超过它算断点")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    autocast = precision_context(device, "bf16")
    judge_model, judge_visible, jc = load(args.root / args.judge, device)
    prompt, blocks, size = jc["prompt"], jc["blocks"], jc["block_size"]
    dev = Windows(args.data / "dev.bin", prompt + blocks * size)
    real = dev.batch(dev.random_starts(random.Random(args.seed), args.prompts)).to(device)
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    results = {"threshold": args.threshold, "prompts": [], "runs": {}}
    for i in range(args.show):
        results["prompts"].append(tokenizer.decode(real[i, :prompt].tolist()))
    for name in args.runs:
        model, visible, _ = load(args.root / name, device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        parts = []
        for i in range(0, args.prompts, args.batch_size):
            ids, _ = sample(model, visible, real[i : i + args.batch_size, :prompt],
                            prompt, args.temperature, generator, autocast)
            parts.append(ids)
        ids = torch.cat(parts)
        nll = torch.cat([
            judge(judge_model, judge_visible, ids[i : i + args.batch_size],
                  prompt, blocks, size, autocast)
            for i in range(0, len(ids), args.batch_size)
        ]).float()
        breaks = nll[:, 2:] > args.threshold  # 稳态块
        results["runs"][name] = {
            "breaks_per_sample": float(breaks.sum((1, 2)).float().mean()),
            "break_share_by_slot": (breaks.sum((0, 1)) / breaks.sum().clamp(min=1)).tolist(),
            "examples": [
                render(tokenizer, ids[i].tolist(), nll[i].tolist(), prompt, size, args.threshold)
                for i in range(args.show)
            ],
        }
        print(name, json.dumps({k: v for k, v in results["runs"][name].items()
                                if k != "examples"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
