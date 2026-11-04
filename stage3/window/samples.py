"""用后验 z（可加噪声）让窗口解码器按调度采样整段续写，由独立 causal 评委逐槽打分。

z 取编码器读真实续写的后验均值，相当于假设 z 采样器完美；再加随机方向的噪声
模拟采样器误差，噪声大小 = r × 本批两随机块 z 的平均距离。每个调度按自己的
可见关系逐轮并行采样（一轮内互不可见的词同时采），zonly 一轮采完全部词。
报告评委 CE（总体、按块内位置）和与真实续写逐词相同的比例。
"""


import argparse
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from common.data import Windows
from stage3.models import WindowLM
from stage2.mask.sample_eval import judge, load
from stage2.schedules import rounds
from stage3.schedules import window_visibility
from common.training import precision_context


def load_window(run, device):
    checkpoint = torch.load(run / "model.pt", map_location=device, weights_only=False)
    c = checkpoint["config"]
    metadata = json.loads((Path(c["data"]) / "metadata.json").read_text(encoding="utf-8"))
    length = c["prompt"] + c["blocks"] * c["block_size"]
    model = WindowLM(
        metadata["vocab_size"], length, c["blocks"], c["block_size"], c["latent_dim"],
        c["dim"], c["layers"], c["heads"], c["ff_dim"], c["encoder_layers"],
        c.get("gate", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    visible = window_visibility(
        c["schedule"], c["prompt"], c["blocks"], c["block_size"], c.get("stagger"),
        c.get("segment", 1), c["window"],
    ).to(device)
    return model.eval(), visible, c


@torch.inference_mode()
def sample(model, visible, prompt_ids, prompt, z, temperature, generator, fill=0):
    """按调度逐轮采样；fill 是未生成位置的占位值，可见关系保证它不被读到。"""
    order = rounds(visible, prompt)
    ids = torch.full((prompt_ids.shape[0], visible.shape[0]), fill, dtype=torch.long,
                     device=prompt_ids.device)
    ids[:, :prompt] = prompt_ids
    for step in range(int(order.max()) + 1):
        chosen = (order == step).nonzero(as_tuple=True)[0]
        logits = model(ids, visible, prompt, z)[:, chosen].float()
        draws = torch.multinomial((logits / temperature).softmax(-1).flatten(0, 1), 1,
                                  generator=generator)
        ids[:, prompt + chosen] = draws.view(ids.shape[0], -1)
    return ids


def noisy(z, r, generator):
    """z + 随机方向噪声，长度 r × 本批两随机块 z 的平均距离。"""
    if r == 0:
        return z
    flat = z.flatten(0, 1)
    other = flat[torch.randperm(len(flat), device=z.device, generator=generator)]
    unit = (flat - other).norm(dim=-1).mean()
    d = torch.randn(z.shape, device=z.device, generator=generator)
    return z + d / d.norm(dim=-1, keepdim=True) * r * unit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/schedule-window"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    p.add_argument("--judge", type=Path, default=Path("runs/schedule-mask/causal-s43"))
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--prompts", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--show", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    autocast = precision_context(device, "bf16")
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    judge_model, judge_visible, jc = load(args.judge, device)
    if jc["schedule"] != "causal":
        raise ValueError("评委必须是 causal 运行")
    prompt, blocks, size = jc["prompt"], jc["blocks"], jc["block_size"]
    dev = Windows(args.data / "dev.bin", prompt + blocks * size)
    real = dev.batch(dev.random_starts(random.Random(args.seed), args.prompts)).to(device)

    def score(ids):
        nll = torch.cat([
            judge(judge_model, judge_visible, ids[i : i + args.batch_size],
                  prompt, blocks, size, autocast)
            for i in range(0, len(ids), args.batch_size)
        ]).float()
        same = (ids[:, prompt:] == real[:, prompt:]).float().view(-1, blocks, size)
        return {
            "judge_ce": float(nll.mean()),
            "judge_steady_by_slot": nll[:, 2:].mean((0, 1)).tolist(),
            "agreement": float(same[:, 2:].mean()),
            "agreement_by_slot": same[:, 2:].mean((0, 1)).tolist(),
        }

    results = {"reference_real_text": score(real)}
    for name in args.runs:
        model, visible, c = load_window(args.root / name, device)
        if (c["prompt"], c["blocks"], c["block_size"]) != (prompt, blocks, size):
            raise ValueError(f"{name} 的序列布局与评委不同")
        for r in args.noise:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            parts = []
            for i in range(0, args.prompts, args.batch_size):
                chunk = real[i : i + args.batch_size]
                with autocast():
                    mean_q, _ = model.encode(chunk, prompt)
                    z = noisy(mean_q.float(), r, generator)
                    parts.append(sample(model, visible, chunk[:, :prompt], prompt, z,
                                        args.temperature, generator))
            ids = torch.cat(parts)
            label = f"{name} noise{r:g}"
            results[label] = score(ids) | {"examples": [
                {"real": tokenizer.decode(real[i, prompt:].tolist()),
                 "sample": tokenizer.decode(ids[i, prompt:].tolist())}
                for i in range(args.show)
            ]}
            s = results[label]["judge_steady_by_slot"]
            print(f"{label:40} judge {results[label]['judge_ce']:.2f}  s0 {s[0]:.2f}  "
                  f"s8 {s[8]:.2f}  s15 {s[15]:.2f}  agree {results[label]['agreement']:.1%}",
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
