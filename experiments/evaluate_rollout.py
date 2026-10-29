"""阶段三：比较真实 latent 历史与自主回灌，检查表示漂移和短文本输出。

真实历史分支只用于诊断，不是实际生成。主干直接预测连续 latent；
比较两个分支的漂移及短续写，不把 latent 距离当作生成质量分数。
"""

import argparse
import json
from pathlib import Path

import torch

from download_data import sha256
from latent_ar.checkpoints import (
    PRIOR_ARCHITECTURE,
    ENERGY_ARCHITECTURE,
    MIXTURE_ARCHITECTURE,
    ENERGY_TEXT_ARCHITECTURE,
    DENOISING_ARCHITECTURE,
    TOKEN_HEAD_ARCHITECTURE,
    TRAINABLE_PRIOR_ARCHITECTURES,
    checkpoint_architecture,
    load_model,
)
from latent_ar.data import StoryDataset
from train import precision_context, select_device, to_device
from tokenizers import Tokenizer


def trim(ids, eos):
    return ids[: ids.index(eos)] if eos in ids else ids


def repetition(ids):
    triples = list(zip(ids, ids[1:], ids[2:]))
    return 0.0 if not triples else 1 - len(set(triples)) / len(triples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("runs/rollout-check-v3"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint_architecture(checkpoint)
    if architecture not in TRAINABLE_PRIOR_ARCHITECTURES | {ENERGY_TEXT_ARCHITECTURE}:
        parser.error("阶段三需要 train_prior.py 生成的 AR/Energy prior 检查点")
    if checkpoint["dataset_hash"] != sha256(args.data / "metadata.json"):
        parser.error("检查点与数据不匹配")
    training = checkpoint["config"]["training"]
    if (
        not 1 <= args.max_new_tokens <= training["max_target_tokens"]
        or args.samples < 1
    ):
        parser.error("样本数需为正，输出长度必须在训练范围内")

    torch.set_num_threads(4)
    device = select_device(args.device)
    model = load_model(checkpoint, device).eval()
    autocast = precision_context(device, training["precision"])
    tokenizer = Tokenizer.from_file(str(args.checkpoint.parent / "tokenizer.json"))
    dataset = StoryDataset(
        args.data,
        "dev",
        training["min_prompt_tokens"],
        training["max_prompt_tokens"],
        args.max_new_tokens,
        model.cfg.pad_id,
    )
    batch = to_device(
        next(dataset.evaluation_batches(min(args.samples, dataset.size), 1, args.seed)),
        device,
    )
    length = batch["labels"].shape[1]
    with autocast():
        contexts, targets, valid = model.teacher_contexts(
            batch["prompt"], batch["prompt_lengths"], batch["labels"]
        )
        oracle_logits = model.decode(targets, length)
    token_valid = batch["labels"] != -100
    oracle_accuracy = float(
        ((oracle_logits.argmax(-1) == batch["labels"]) & token_valid).sum()
        / token_valid.sum()
    )
    pairs = []
    position_metrics = []
    energy = architecture in {
        ENERGY_ARCHITECTURE,
        ENERGY_TEXT_ARCHITECTURE,
        DENOISING_ARCHITECTURE,
        TOKEN_HEAD_ARCHITECTURE,
    }
    noise = None
    if energy:
        noise = model.draw_noise(
            targets.shape[:2],
            device,
            torch.Generator(device=device).manual_seed(args.seed),
        )
    if architecture == MIXTURE_ARCHITECTURE:
        noise = torch.rand(
            (*targets.shape[:2], 1),
            device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
    with autocast():
        teacher = model.sample_latent(contexts, noise=noise)
        free = model.rollout(
            batch["prompt"], batch["prompt_lengths"], targets.shape[1], noise=noise
        )
        teacher_ids = model.decode(teacher, length).float()
        free_ids = model.decode(free, length).float()
        # 与正式生成保持相同的特殊 token 屏蔽。
        teacher_ids[..., model.cfg.pad_id] = -torch.inf
        teacher_ids[..., model.cfg.bos_id] = -torch.inf
        free_ids[..., model.cfg.pad_id] = -torch.inf
        free_ids[..., model.cfg.bos_id] = -torch.inf
        teacher_ids = teacher_ids.argmax(-1)
        free_ids = free_ids.argmax(-1)

    normalized_teacher = model.normalize(teacher)
    normalized_free = model.normalize(free)
    normalized_target = model.normalize(targets)
    for index in range(targets.shape[1]):
        mask = valid[:, index]
        if not bool(mask.any()):
            continue
        position_metrics.append(
            {
                "latent_step": index + 1,
                "oracle_rms": float(
                    normalized_target[mask, index].square().mean().sqrt()
                ),
                "teacher_rms": float(
                    normalized_teacher[mask, index].square().mean().sqrt()
                ),
                "free_rms": float(normalized_free[mask, index].square().mean().sqrt()),
                "teacher_free_distance": float(
                    (normalized_teacher[mask, index] - normalized_free[mask, index])
                    .square()
                    .mean()
                    .sqrt()
                ),
                "free_finite": bool(torch.isfinite(free[:, index]).all()),
            }
        )
    for row in range(batch["prompt"].shape[0]):
        original = teacher_ids[row].tolist()
        generated = free_ids[row].tolist()
        original_content = trim(original, model.cfg.eos_id)
        content = trim(generated, model.cfg.eos_id)
        prompt_length = int(batch["prompt_lengths"][row])
        pairs.append(
            {
                "sample": row,
                "prompt": tokenizer.decode(
                    batch["prompt"][row, :prompt_length].tolist()
                ),
                "teacher_history_text": tokenizer.decode(original_content),
                "free_rollout_text": tokenizer.decode(content),
                "free_token_ids": content,
                "free_length": len(content),
                "free_eos": model.cfg.eos_id in generated,
                "free_repeated_trigram_fraction": repetition(content),
            }
        )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step": checkpoint["step"],
        "samples": batch["prompt"].shape[0],
        "seed": args.seed,
        "latent_prediction": architecture,
        "max_new_tokens": length,
        "oracle_codec_token_accuracy": oracle_accuracy,
        "all_free_latents_finite": all(row["free_finite"] for row in position_metrics),
        "position_metrics": position_metrics,
        "pairs": pairs,
        "notes": [
            "Teacher-history branch uses true target latent prefixes; not an inference result.",
            "Both branches use the same per-position noise when stochastic; the first prediction must agree.",
            "Latent distances and RMS diagnose drift; they are not semantic quality scores.",
            "No generated text is fed back to the prior.",
            "All planned latent positions are measured, including positions after generated EOS.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 自主 latent 回灌检查",
        "",
        f"检查点 step {checkpoint['step']}；codec 在本批真实目标上的还原准确率 {oracle_accuracy:.2%}。",
        "",
        "真实历史分支读取正确的前序 target latent，仅用于诊断；自主分支只读取 prompt 与自身预测的 latent。差异不等于生成质量下降，需人工检查下列短续写。",
        "",
        f"全部自主 latent 为有限值：{report['all_free_latents_finite']}。",
        "",
    ]
    for row in pairs:
        lines += [
            f"## 样本 {row['sample'] + 1}",
            "",
            "**输入**",
            "",
            row["prompt"],
            "",
            "**真实 latent 历史（非实际推理）**",
            "",
            row["teacher_history_text"] or "（EOS）",
            "",
            "**自主回灌**",
            "",
            row["free_rollout_text"] or "（EOS）",
            "",
        ]
    (args.output / "rollout.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[检查] codec 还原 {oracle_accuracy:.2%} | 自主 latent 有限：{report['all_free_latents_finite']}"
    )
    print(f"[结果] {args.output / 'rollout.md'}")


if __name__ == "__main__":
    main()
