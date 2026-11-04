"""从带连续潜变量的调度模型采样，并用独立 causal 评委逐槽打分。

按生成时刻逐项采样：每块先由先验 p(z|可见内容) 采出 K 个潜变量，再按调度顺序采词。
可见关系保证每次预测只读到已经生成的内容，所以未生成位置可以先填占位值。
报告评委 CE（总体、按块内位置），并给出若干条完整续写供阅读。
"""


import argparse
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from seam.data import Windows
from seam.models import LatentLM
from seam.plan.samples import generation_order
from seam.mask.sample_eval import judge, load
from seam.schedules import plan_visibility
from seam.common import precision_context


def load_latent(run, device):
    checkpoint = torch.load(run / "model.pt", map_location=device, weights_only=False)
    c = checkpoint["config"]
    metadata = json.loads((Path(c["data"]) / "metadata.json").read_text(encoding="utf-8"))
    length = c["prompt"] + c["blocks"] * c["block_size"]
    model = LatentLM(
        metadata["vocab_size"], length, c["blocks"], c["block_size"], c["latents"],
        c["latent_dim"], c["dim"], c["layers"], c["heads"], c["ff_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    visible = plan_visibility(
        c["schedule"], c["prompt"], c["blocks"], c["block_size"], c["latents"],
        c.get("stagger"), c.get("segment", 1),
    ).to(device)
    return model.eval(), visible, c


@torch.inference_mode()
def sample(model, visible, c, prompt_ids, temperature, generator, fill=0, oracle=None):
    """返回 (词 [batch, N], 潜变量 [batch, blocks·K, d_z])；fill 是未生成位置的占位值。

    oracle 给定时（编码器读真实续写得到的 z），潜变量不再从先验采样，只采词。
    """
    batch, prompt = prompt_ids.shape[0], c["prompt"]
    n = prompt + c["blocks"] * c["block_size"]
    ids = torch.full((batch, n), fill, dtype=torch.long, device=prompt_ids.device)
    ids[:, :prompt] = prompt_ids
    z = torch.full((batch, c["blocks"] * c["latents"], c["latent_dim"]), float(fill),
                   device=ids.device)
    if oracle is not None:
        z = oracle.float().clone()
    for kind, index in generation_order(c | {"codes": c["latents"]}):
        if kind == "plan" and oracle is not None:
            continue
        word_logits, mean_p, logvar_p = model(ids, visible, prompt, z)
        if kind == "plan":
            noise = torch.randn(mean_p[:, index].shape, device=ids.device, generator=generator)
            # 温度只作用于词；潜变量按先验原样采样。
            z[:, index] = mean_p[:, index].float() + (0.5 * logvar_p[:, index].float()).exp() * noise
        else:
            logits = word_logits[:, index].float() / temperature
            ids[:, prompt + index] = torch.multinomial(
                logits.softmax(-1), 1, generator=generator)[:, 0]
    return ids, z


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/schedule-latent"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--judge", type=Path, default=Path("runs/schedule-mask/causal-s43"))
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--prompts", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--show", type=int, default=6, help="保存多少条完整续写")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--oracle", action="store_true",
                   help="z 取后验 q(z|真实续写) 的一次采样，只采词：上限，等于假设 z 采样器完美")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    autocast = precision_context(device, "bf16")
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    judge_model, judge_visible, jc = load(args.judge, device)
    if jc["schedule"] != "causal":
        raise ValueError("评委必须是 causal 运行")
    results = {}
    for name in args.runs:
        model, visible, c = load_latent(args.root / name, device)
        prompt, blocks, size = c["prompt"], c["blocks"], c["block_size"]
        if (prompt, blocks, size) != (jc["prompt"], jc["blocks"], jc["block_size"]):
            raise ValueError(f"{name} 的序列布局与评委不同")
        dev = Windows(args.data / "dev.bin", prompt + blocks * size)
        real = dev.batch(dev.random_starts(random.Random(args.seed), args.prompts)).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        parts = []
        for i in range(0, args.prompts, args.batch_size):
            chunk = real[i : i + args.batch_size]
            oracle = None
            if args.oracle:
                with autocast():
                    mean_q, logvar_q = model.encode(chunk, prompt)
                noise = torch.randn(mean_q.shape, device=device, generator=generator)
                oracle = mean_q + (0.5 * logvar_q).exp() * noise
            with autocast():
                parts.append(sample(model, visible, c, chunk[:, :prompt],
                                    args.temperature, generator, oracle=oracle)[0])
        ids = torch.cat(parts)
        nll = torch.cat([
            judge(judge_model, judge_visible, ids[i : i + args.batch_size],
                  prompt, blocks, size, autocast)
            for i in range(0, args.prompts, args.batch_size)
        ]).float()
        steady = nll[:, 2:].mean((0, 1))
        results[name] = {
            "judge_ce": float(nll.mean()),
            "judge_steady_by_slot": steady.tolist(),
            "examples": [
                {"prompt": tokenizer.decode(real[i, :prompt].tolist()),
                 "continuation": tokenizer.decode(ids[i, prompt:].tolist())}
                for i in range(args.show)
            ],
        }
        print(name, round(float(nll.mean()), 3), "slot0", round(float(steady[0]), 2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
