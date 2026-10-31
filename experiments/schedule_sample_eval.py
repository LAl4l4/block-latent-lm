"""按各调度自身的生成顺序采样续写，再用 causal 模型当评委打分。

CE 是似然指标；本脚本直接看样本：同一批 dev prompt，每个调度按自己的可见关系
逐轮并行采样（一轮内互不可见的位置同时采样），轮数就是该调度的串行步数。
评委是同预算训练的 causal 模型，报告样本在评委下的 CE（总体、按块内位置、
衔接处首词）和重复 4-gram 比例；同时给出真实续写和 causal 自身样本作参照。
"""

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from experiments.schedule_mask_study import TwoStreamLM, Windows, visibility
from train import precision_context


def load(run, device):
    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    c = saved["config"]
    length = c["prompt"] + c["blocks"] * c["block_size"]
    model = TwoStreamLM(
        saved["model"]["token.weight"].shape[0], length,
        c["dim"], c["layers"], c["heads"], c["ff_dim"],
    )
    model.load_state_dict(saved["model"])
    visible = visibility(
        c["schedule"], c["prompt"], c["blocks"], c["block_size"], c.get("stagger"),
        c.get("segment", 1),
    )
    return model.to(device).eval(), visible.to(device), c


def rounds(visible, prompt):
    """每个生成位置最早能采样的轮次：所依赖的生成位置都已在更早轮次完成。"""
    n = visible.shape[0]
    depth = torch.zeros(n, dtype=torch.long, device=visible.device)
    generated = torch.arange(n, device=visible.device) >= prompt
    needs = visible & generated[None, :]
    for _ in range(n):
        update = torch.where(needs, depth[None, :] + 1, 0).max(1).values
        update = torch.where(generated, update, 0)
        if torch.equal(update, depth):
            break
        depth = update
    return depth[prompt:]


@torch.inference_mode()
def sample(model, visible, prompts, prompt, temperature, generator, autocast):
    targets = torch.arange(prompt, visible.shape[0], device=visible.device)
    order = rounds(visible, prompt)
    ids = torch.zeros(prompts.shape[0], visible.shape[0], dtype=torch.long,
                      device=visible.device)
    ids[:, :prompt] = prompts
    for step in range(int(order.max()) + 1):
        with autocast():
            logits = model(ids, visible, targets).float()
        chosen = (order == step).nonzero(as_tuple=True)[0]
        probs = (logits[:, chosen] / temperature).softmax(-1)
        draws = torch.multinomial(probs.flatten(0, 1), 1, generator=generator)
        ids[:, targets[chosen]] = draws.view(ids.shape[0], -1)
    return ids, int(order.max()) + 1


@torch.inference_mode()
def judge(model, visible, ids, prompt, blocks, size, autocast):
    targets = torch.arange(prompt, visible.shape[0], device=visible.device)
    with autocast():
        logits = model(ids, visible, targets).float()
    nll = F.cross_entropy(
        logits.flatten(0, 1), ids[:, targets].flatten(), reduction="none"
    ).view(ids.shape[0], blocks, size)
    return nll


def repeated_4grams(ids):
    total = repeated = 0
    for row in ids.tolist():
        grams = [tuple(row[i : i + 4]) for i in range(len(row) - 3)]
        total += len(grams)
        repeated += len(grams) - len(set(grams))
    return repeated / max(1, total)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/schedule-mask"))
    p.add_argument("--runs", nargs="+", required=True, help="要评估的运行目录名")
    p.add_argument("--judge", default="causal-s42")
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--prompts", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2042)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    autocast = precision_context(device, "bf16")
    judge_model, judge_visible, jc = load(args.root / args.judge, device)
    if jc["schedule"] != "causal":
        raise ValueError("评委必须是 causal 运行")
    prompt, blocks, size = jc["prompt"], jc["blocks"], jc["block_size"]
    length = prompt + blocks * size
    dev = Windows(args.data / "dev.bin", length)
    real = dev.batch(dev.random_starts(random.Random(args.seed), args.prompts)).to(device)
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))

    def score(ids):
        nll = torch.cat([
            judge(judge_model, judge_visible, ids[i : i + args.batch_size],
                  prompt, blocks, size, autocast)
            for i in range(0, len(ids), args.batch_size)
        ])
        steady = nll[:, 2:]
        return {
            "judge_ce": float(nll.mean()),
            "judge_steady_ce": float(steady.mean()),
            "judge_steady_first_ce": float(steady[:, :, 0].mean()),
            "judge_steady_by_slot": steady.mean((0, 1)).tolist(),
            "repeated_4gram": repeated_4grams(ids[:, prompt:]),
        }

    results = {"reference_real_text": score(real)}
    for name in args.runs:
        model, visible, c = load(args.root / name, device)
        if (c["prompt"], c["blocks"], c["block_size"]) != (prompt, blocks, size):
            raise ValueError(f"{name} 的序列布局与评委不同")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        parts = []
        for i in range(0, args.prompts, args.batch_size):
            ids, steps = sample(model, visible, real[i : i + args.batch_size, :prompt],
                                prompt, args.temperature, generator, autocast)
            parts.append(ids)
        ids = torch.cat(parts)
        results[name] = score(ids) | {
            "serial_steps": steps,
            "tokens_per_step": blocks * size / steps,
            "examples": [
                {
                    "prompt": tokenizer.decode(ids[i, :prompt].tolist()),
                    "continuation": tokenizer.decode(ids[i, prompt:].tolist()),
                }
                for i in range(4)
            ],
        }
        print(name, json.dumps({k: v for k, v in results[name].items()
                                if k not in ("examples", "judge_steady_by_slot")}),
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
