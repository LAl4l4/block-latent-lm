"""比较原始 latent 序列与该序列的均值，使用相同随机数生成短续写。

这里先给定 prompt，取出它生成的一组 latent，随机主干固定 latent seed，
再比较原始序列和“将 16 个向量平均后重复 16 次”的序列。
这是一项离线诊断；均值包含完整 latent 段，不是流式推理调度方案。
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from latent_ar.data import StoryDataset
from latent_ar.checkpoints import (
    BLOCK_ARCHITECTURE,
    CODEC_ARCHITECTURE,
    PRIOR_ARCHITECTURE,
    FLOW_ARCHITECTURE,
    ENERGY_ARCHITECTURE,
    MIXTURE_ARCHITECTURE,
    ENERGY_TEXT_ARCHITECTURE,
    DENOISING_ARCHITECTURE,
    TOKEN_HEAD_ARCHITECTURE,
    checkpoint_architecture,
    load_model,
)
from train import precision_context, select_device


def mean_generated_latents(memory, generated_count):
    """只平均生成部分，保留 prompt latent、数量和位置不变。"""
    if not 1 <= generated_count <= memory.shape[1]:
        raise ValueError("Invalid generated latent count")

    result = memory.clone()
    generated = memory[:, -generated_count:]
    average = generated.float().mean(dim=1, keepdim=True).to(memory.dtype)
    result[:, -generated_count:] = average.expand(-1, generated_count, -1)
    return result


@contextmanager
def use_fixed_rollout(model, rollout):
    """复用正式 generate()，只替换它读取的 latent，不另写一套采样器。"""
    original = model.rollout
    model.rollout = lambda *args, **kwargs: rollout
    try:
        yield
    finally:
        model.rollout = original


def trim_eos(ids, eos_id):
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


def identical_prefix(left, right):
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def select_prompts(args, training, tokenizer, cfg):
    if args.prompt:
        prompts = []
        for text in args.prompt:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids:
                raise ValueError("Prompt must contain text")
            prompts.append(ids[-training["max_prompt_tokens"] :])
        return prompts

    # 默认取固定 dev 窗口，使用模型训练覆盖过的故事续写形式。
    dataset = StoryDataset(
        args.data,
        "dev",
        training["min_prompt_tokens"],
        training["max_prompt_tokens"],
        args.max_new_tokens,
        cfg.pad_id,
    )
    prompts = []
    for index in range(min(args.prompt_count, dataset.size)):
        ids, _ = dataset.example(index, random.Random(training["seed"] + 10000 + index))
        prompts.append(ids.tolist())
    return prompts


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/latent-mean-comparison")
    )
    parser.add_argument(
        "--prompt", action="append", help="可重复传入；省略则取固定 dev 窗口"
    )
    parser.add_argument("--prompt-count", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    training = checkpoint["config"]["training"]
    architecture = checkpoint_architecture(checkpoint)
    if architecture == CODEC_ARCHITECTURE:
        parser.error("Codec 没有语言 prior，请使用 V3 prior 检查点")
    if args.temperature is None:
        args.temperature = (
            0.0
            if architecture
            in {
                BLOCK_ARCHITECTURE,
                PRIOR_ARCHITECTURE,
                FLOW_ARCHITECTURE,
                ENERGY_ARCHITECTURE,
                DENOISING_ARCHITECTURE,
                TOKEN_HEAD_ARCHITECTURE,
            }
            else (
                1.0
                if architecture in {MIXTURE_ARCHITECTURE, ENERGY_TEXT_ARCHITECTURE}
                else 0.8
            )
        )
    if not 1 <= args.max_new_tokens <= training["max_target_tokens"]:
        parser.error("比较长度必须在该模型的训练续写长度内")
    if args.prompt_count < 1:
        parser.error("prompt-count must be positive")
    if checkpoint["config"].get("ablation", "full") != "full":
        parser.error("This comparison requires a model trained with generated latents")

    torch.set_num_threads(4)
    device = select_device(args.device)
    autocast = precision_context(device, training["precision"])
    tokenizer = Tokenizer.from_file(str(args.checkpoint.parent / "tokenizer.json"))
    model = load_model(checkpoint, device).eval()
    cfg = model.cfg
    prompts = select_prompts(args, training, tokenizer, cfg)
    steps = math.ceil(args.max_new_tokens / cfg.chunk_size)
    results = []
    latent_statistics = []

    for index, ids in enumerate(prompts):
        prompt = torch.tensor([ids], device=device)
        lengths = torch.tensor([len(ids)], device=device)
        with autocast():
            if architecture in {
                FLOW_ARCHITECTURE,
                ENERGY_ARCHITECTURE,
                MIXTURE_ARCHITECTURE,
                ENERGY_TEXT_ARCHITECTURE,
                DENOISING_ARCHITECTURE,
                TOKEN_HEAD_ARCHITECTURE,
            }:
                original = model.rollout(
                    prompt,
                    lengths,
                    steps,
                    generator=torch.Generator(device=device).manual_seed(
                        training["seed"] + index
                    ),
                )
            else:
                original = model.rollout(prompt, lengths, steps)
            if architecture in {
                BLOCK_ARCHITECTURE,
                PRIOR_ARCHITECTURE,
                FLOW_ARCHITECTURE,
                ENERGY_ARCHITECTURE,
                MIXTURE_ARCHITECTURE,
                ENERGY_TEXT_ARCHITECTURE,
                DENOISING_ARCHITECTURE,
                TOKEN_HEAD_ARCHITECTURE,
            }:
                memory = original
                mean_rollout = mean_generated_latents(memory, steps)
            else:
                memory, positions, valid, _ = original
                averaged = mean_generated_latents(memory, steps)
                mean_rollout = (
                    averaged,
                    positions,
                    valid,
                    list(averaged[:, -steps:].unbind(1)),
                )

        # 只描述向量变化幅度，不把它当作生成质量分数。
        generated = memory[:, -steps:].float()
        mean = generated.mean(dim=1, keepdim=True)
        cosine = F.cosine_similarity(generated, mean.expand_as(generated), dim=-1)
        ratio = ((generated - mean).square().mean() / generated.square().mean()).sqrt()
        latent_statistics.append(
            {
                "prompt_index": index,
                "mean_cosine_to_average": float(cosine.mean()),
                "relative_temporal_rms": float(ratio),
            }
        )

        for seed in args.seeds:
            outputs = {}
            for label, rollout in [("original", original), ("mean", mean_rollout)]:
                # 每个条件都重新建立同一种子的生成器，避免随机数流不同。
                generator = torch.Generator(device=device).manual_seed(seed)
                with use_fixed_rollout(model, rollout), autocast():
                    output = model.generate(
                        prompt,
                        lengths,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        generator=generator,
                    )
                raw = output[0].tolist()
                content = trim_eos(raw, cfg.eos_id)
                outputs[label] = {
                    "token_ids": raw,
                    "text": tokenizer.decode(content),
                    "content_tokens": len(content),
                    "stopped_by_eos": cfg.eos_id in raw,
                }
            left = outputs["original"]["token_ids"]
            right = outputs["mean"]["token_ids"]
            results.append(
                {
                    "prompt_index": index,
                    "prompt": tokenizer.decode(ids),
                    "seed": seed,
                    **outputs,
                    "identical_output": left == right,
                    "identical_prefix_tokens": identical_prefix(left, right),
                }
            )
            print(
                f"[比较] prompt {index + 1}/{len(prompts)} | seed {seed} | 相同输出：{left == right}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint["step"],
        "architecture": architecture,
        "max_new_tokens": args.max_new_tokens,
        "latent_steps": steps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seeds": args.seeds,
        "prompt_count": len(prompts),
        "pair_count": len(results),
        "identical_pairs": sum(row["identical_output"] for row in results),
        "first_token_different_pairs": sum(
            row["identical_prefix_tokens"] == 0 for row in results
        ),
        "mean_original_content_tokens": sum(
            row["original"]["content_tokens"] for row in results
        )
        / len(results),
        "mean_averaged_content_tokens": sum(
            row["mean"]["content_tokens"] for row in results
        )
        / len(results),
        "mean_definition": "Per-prompt temporal mean over the complete generated latent segment; broadcast to all generated positions. Prompt latents unchanged.",
        "limitations": "Offline perturbation diagnostic, not a separately trained no-latent baseline or a causal quality estimate.",
        "latent_statistics": latent_statistics,
    }
    (args.output / "results.json").write_text(
        json.dumps(
            {"metadata": metadata, "pairs": results}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 原始 latent 与均值 latent 的短文本生成对比",
        "",
        f"检查点：step {checkpoint['step']}；{len(prompts)} 个 prompt × {len(args.seeds)} 个随机种子；最多 {args.max_new_tokens} token。",
        "",
        "均值定义：对每个 prompt 的完整生成 latent 序列取时间均值，再重复到全部生成位置。保留原 prompt latent。两边采样设置和随机种子完全相同；不强制最低生成长度。",
        "",
        "这项离线诊断比较逐步变化的 latent 与其均值；均值仍包含该 prompt 的信息，不能等同于没有 latent，也不能据此单独判断主干的质量收益。",
        "",
        f"完全相同的输出：{metadata['identical_pairs']} / {len(results)}。",
        "",
        f"第一个 token 即不同：{metadata['first_token_different_pairs']} / {len(results)}；平均续写长度：原始 {metadata['mean_original_content_tokens']:.2f} token，均值 {metadata['mean_averaged_content_tokens']:.2f} token。",
        "",
    ]
    for row in results:
        lines += [
            f"## Prompt {row['prompt_index'] + 1} / seed {row['seed']}",
            "",
            "**输入**",
            "",
            row["prompt"],
            "",
            "**原始 latent**",
            "",
            row["original"]["text"] or "（直接 EOS）",
            "",
            "**均值 latent**",
            "",
            row["mean"]["text"] or "（直接 EOS）",
            "",
            f"相同前缀：{row['identical_prefix_tokens']} token；输出长度：{row['original']['content_tokens']} / {row['mean']['content_tokens']} token。",
            "",
        ]
    (args.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] {args.output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
