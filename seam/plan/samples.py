"""从带计划的调度模型采样，按块打印生成结果（阅读顺序），并用独立 causal 评委打分。

按生成时刻逐项采样：每块先由先验采出 K 个计划码，再按调度顺序采词。
可见关系保证每次预测只读到已经生成的内容，所以未生成位置可以先填占位值。
"""


import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from seam.data import Windows
from seam.models import PlanLM
from seam.mask.sample_eval import judge, load
from seam.schedules import plan_visibility, token_clock
from seam.common import precision_context


def load_plan(run, device):
    checkpoint = torch.load(run / "model.pt", map_location=device, weights_only=False)
    c = checkpoint["config"]
    metadata = json.loads((Path(c["data"]) / "metadata.json").read_text(encoding="utf-8"))
    length = c["prompt"] + c["blocks"] * c["block_size"]
    model = PlanLM(
        metadata["vocab_size"], length, c["blocks"], c["block_size"], c["codes"],
        c["choices"], c["dim"], c["layers"], c["heads"], c["ff_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    visible = plan_visibility(
        c["schedule"], c["prompt"], c["blocks"], c["block_size"], c["codes"],
        c.get("stagger"), c.get("segment", 1),
    ).to(device)
    return model.eval(), visible, c


def generation_order(c):
    """返回按生成时刻排序的 (类型, 下标) 列表：('plan', 码序号) 或 ('word', 生成词序号)。"""
    clock, starts = token_clock(
        c["schedule"], c["prompt"], c["blocks"], c["block_size"],
        c.get("stagger"), c.get("segment", 1),
    )
    codes = c["codes"]
    items = [(float(t), "word", i) for i, t in enumerate(clock.tolist())]
    for j in range(c["blocks"] * codes):
        block, k = divmod(j, codes)
        items.append((float(starts[block]) - 1 + (k + 1) / (codes + 1), "plan", j))
    return [(kind, index) for _, kind, index in sorted(items)]


@torch.inference_mode()
def sample(model, visible, c, prompt_ids, temperature, generator):
    batch, prompt = prompt_ids.shape[0], c["prompt"]
    n = prompt + c["blocks"] * c["block_size"]
    ids = torch.zeros(batch, n, dtype=torch.long, device=prompt_ids.device)
    ids[:, :prompt] = prompt_ids
    codes = torch.zeros(batch, c["blocks"] * c["codes"], dtype=torch.long, device=ids.device)
    model.encode = lambda ids_, prompt_: (F.one_hot(codes, c["choices"]).float(), codes)
    for kind, index in generation_order(c):
        word_logits, plan_logits, _, _ = model(ids, visible, prompt)
        logits = (plan_logits if kind == "plan" else word_logits)[:, index].float()
        choice = torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator)[:, 0]
        if kind == "plan":
            codes[:, index] = choice
        else:
            ids[:, prompt + index] = choice
    return ids, codes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/schedule-plan"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--judge", type=Path, default=Path("runs/schedule-mask-wide/causal-s42"))
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--prompts", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2042)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    autocast = precision_context(device, "bf16")
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    judge_model, judge_visible, jc = load(args.judge, device)
    results = {}
    for name in args.runs:
        model, visible, c = load_plan(args.root / name, device)
        length = c["prompt"] + c["blocks"] * c["block_size"]
        dev = Windows(args.data / "dev.bin", length)
        real = dev.batch(dev.random_starts(random.Random(args.seed), args.prompts)).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        with autocast():
            ids, codes = sample(model, visible, c, real[:, : c["prompt"]], args.temperature, generator)
            nll = judge(judge_model, judge_visible, ids, c["prompt"], c["blocks"], c["block_size"], autocast)
        size = c["block_size"]
        examples = []
        for i in range(args.prompts):
            blocks = []
            for b in range(c["blocks"]):
                begin = c["prompt"] + b * size
                blocks.append({
                    "text": tokenizer.decode(ids[i, begin : begin + size].tolist()),
                    "codes": codes[i, b * c["codes"] : (b + 1) * c["codes"]].tolist(),
                    "judge_ce": float(nll[i, b].mean()),
                    "judge_first_ce": float(nll[i, b, 0]),
                })
            examples.append({
                "prompt": tokenizer.decode(real[i, : c["prompt"]].tolist()),
                "real": tokenizer.decode(real[i, c["prompt"] :].tolist()),
                "blocks": blocks,
            })
        results[name] = {"judge_ce": float(nll.mean()), "examples": examples}
        print(name, round(float(nll.mean()), 3), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
