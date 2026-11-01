"""只重新采样一块：上下文全用真实文本，看衔接处本身接得好不好。

整段采样时，样本一偏离真实文本，真实 z（后验读真实续写）就与已写出的上下文矛盾，
所以"真实 z 整段采样"不是合法的上限。这里改为：其余块全是真实文本，只对第 t 块
（默认 t = 2..5）按各调度自己的生成顺序重新采样，再由独立 causal 评委逐槽打分。

z 的取法：
  - prior：第 t 块的潜变量由先验 p(z|真实上下文) 采样，其余块的 z 取后验；
  - oracle：所有块的 z 都取后验 q(z|真实文本) 的一次采样，只采词；
  - 无潜变量的 mask 模型作为对照（--mask-runs）。
"""


import argparse
import json
from pathlib import Path
import random

import torch

from seam.data import Windows
from seam.latent_samples import load_latent
from seam.plan_samples import generation_order
from seam.sample_eval import judge, load
from seam.schedules import rounds
from train import precision_context


@torch.inference_mode()
def latent_block(model, visible, c, ids, t, mode, generator):
    """把 ids 的第 t 块按调度顺序重新采样；mode 为 "prior" 或 "oracle"。"""
    prompt, size, latents = c["prompt"], c["block_size"], c["latents"]
    ids = ids.clone()
    mean_q, logvar_q = model.encode(ids, prompt)
    noise = torch.randn(mean_q.shape, device=ids.device, generator=generator)
    z = (mean_q + (0.5 * logvar_q).exp() * noise).float()
    for kind, index in generation_order(c | {"codes": latents}):
        if kind == "plan" and (mode != "prior" or index // latents != t):
            continue
        if kind == "word" and index // size != t:
            continue
        word_logits, mean_p, logvar_p = model(ids, visible, prompt, z)
        if kind == "plan":
            noise = torch.randn(mean_p[:, index].shape, device=ids.device, generator=generator)
            z[:, index] = mean_p[:, index].float() + (0.5 * logvar_p[:, index].float()).exp() * noise
        else:
            ids[:, prompt + index] = torch.multinomial(
                word_logits[:, index].float().softmax(-1), 1, generator=generator)[:, 0]
    return ids


@torch.inference_mode()
def mask_block(model, visible, c, ids, t, generator):
    """无潜变量的 mask 模型：同样只重新采样第 t 块，一轮内互不可见的词同时采样。"""
    prompt, size = c["prompt"], c["block_size"]
    n = visible.shape[0] - prompt
    ids = ids.clone()
    targets = torch.arange(prompt, visible.shape[0], device=ids.device)
    order = rounds(visible, prompt)
    in_block = torch.arange(n, device=ids.device) // size == t
    for step in range(int(order.max()) + 1):
        chosen = ((order == step) & in_block).nonzero(as_tuple=True)[0]
        if len(chosen) == 0:
            continue
        probs = model(ids, visible, targets)[:, chosen].float().softmax(-1)
        draws = torch.multinomial(probs.flatten(0, 1), 1, generator=generator)
        ids[:, prompt + chosen] = draws.view(ids.shape[0], -1)
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-root", type=Path, default=Path("runs/schedule-latent"))
    p.add_argument("--latent-runs", nargs="*", default=[])
    p.add_argument("--mask-root", type=Path, default=Path("runs/schedule-mask"))
    p.add_argument("--mask-runs", nargs="*", default=[])
    p.add_argument("--judge", type=Path, default=Path("runs/schedule-mask/causal-s43"))
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--resample", type=int, nargs="+", default=[2, 3, 4, 5],
                   help="逐个重新采样的块下标")
    p.add_argument("--prompts", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    autocast = precision_context(device, "bf16")
    judge_model, judge_visible, jc = load(args.judge, device)
    if jc["schedule"] != "causal":
        raise ValueError("评委必须是 causal 运行")
    prompt, blocks, size = jc["prompt"], jc["blocks"], jc["block_size"]
    dev = Windows(args.data / "dev.bin", prompt + blocks * size)
    real = dev.batch(dev.random_starts(random.Random(args.seed), args.prompts)).to(device)

    def score(make):
        """每个 t 重新采样一次，返回第 t 块各槽的评委 NLL（对 prompt 和 t 取平均）。"""
        per = []
        for t in args.resample:
            out = torch.cat([make(real[i : i + args.batch_size], t)
                             for i in range(0, args.prompts, args.batch_size)])
            nll = torch.cat([
                judge(judge_model, judge_visible, out[i : i + args.batch_size],
                      prompt, blocks, size, autocast)
                for i in range(0, args.prompts, args.batch_size)
            ]).float()
            per.append(nll[:, t])
        return torch.cat(per).mean(0)

    rows = {"real text": score(lambda ids, t: ids)}
    for name in args.mask_runs:
        model, visible, c = load(args.mask_root / name, device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        with autocast():
            rows[f"{name} no-z"] = score(
                lambda ids, t: mask_block(model, visible, c, ids, t, generator))
    for name in args.latent_runs:
        model, visible, c = load_latent(args.latent_root / name, device)
        if (c["prompt"], c["blocks"], c["block_size"]) != (prompt, blocks, size):
            raise ValueError(f"{name} 的序列布局与评委不同")
        for mode in ("prior", "oracle"):
            generator = torch.Generator(device=device).manual_seed(args.seed)
            with autocast():
                rows[f"{name} z-{mode}"] = score(
                    lambda ids, t: latent_block(model, visible, c, ids, t, mode, generator))

    slots = sorted({0, 1, size // 2 - 1, size // 2, size // 2 + 1, size - 1})
    print(f"重新采样块 {args.resample}，{args.prompts} 个 prompt，评委 {args.judge.name}")
    width = max(len(k) for k in rows) + 2
    print(f"{'':{width}}{'mean':>6}" + "".join(f"{'s' + str(j):>6}" for j in slots))
    for k, s in rows.items():
        print(f"{k:{width}}{float(s.mean()):6.2f}" + "".join(f"{float(s[j]):6.2f}" for j in slots))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {k: {"mean": float(s.mean()), "by_slot": s.tolist()} for k, s in rows.items()},
            indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
