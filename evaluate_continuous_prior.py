"""固定 dev 上评价真正的下一 latent 预测与自由回灌，不使用带噪目标作推理输入。"""

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.checkpoints import load_model
from latent_ar.data import StoryDataset
from latent_ar.model import ModelConfig
from token_ar_baseline import TokenAR
from train import precision_context, select_device, to_device


def trim(ids, eos):
    return ids[: ids.index(eos)] if eos in ids else ids


@torch.inference_mode()
def evaluate(path, batch, rater, autocast, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["dataset_hash"] != sha256(Path("data/prepared/metadata.json")):
        raise ValueError("Dataset mismatch")
    model = load_model(checkpoint, device).eval()
    labels = batch["labels"]
    with autocast():
        context, targets, valid = model.teacher_contexts(
            batch["prompt"], batch["prompt_lengths"], labels
        )
    normalized = model.normalize(targets)
    output = {
        "checkpoint": str(path),
        "step": checkpoint["step"],
        "zero_predictor_mse": float(normalized[valid].square().mean()),
        "seeds": [],
    }
    tokenizer = Tokenizer.from_file("data/prepared/tokenizer.json")
    for seed in [42, 43, 44]:
        noise = (
            model.draw_noise(
                targets.shape[:2],
                device,
                torch.Generator(device=device).manual_seed(seed),
            )
            if hasattr(model, "draw_noise")
            else None
        )
        with autocast():
            teacher = model.sample_latent(context, noise=noise)
            wrong = model.sample_latent(context.roll(1, 0), noise=noise)
            free = model.rollout(
                batch["prompt"], batch["prompt_lengths"], targets.shape[1], noise=noise
            )
            teacher_logits = model.decode(teacher, labels.shape[1]).float()
            free_logits = model.decode(free, labels.shape[1]).float()
        teacher_mse = float(
            (model.normalize(teacher)[valid] - normalized[valid]).square().mean()
        )
        wrong_mse = float(
            (model.normalize(wrong)[valid] - normalized[valid]).square().mean()
        )
        first_distance = float((teacher[:, 0].float() - free[:, 0].float()).abs().max())
        for logits in [teacher_logits, free_logits]:
            logits[..., model.cfg.pad_id] = -torch.inf
            logits[..., model.cfg.bos_id] = -torch.inf
        predicted = free_logits.argmax(-1)
        texts = []
        score, token_count, repeat = 0.0, 0, 0.0
        # 本地普通 LM 仅作流畅性代理；不当作语义质量或唯一选模标准。
        for row in range(len(labels)):
            all_ids = predicted[row].tolist()
            ids = trim(all_ids, model.cfg.eos_id)
            triples = list(zip(ids, ids[1:], ids[2:]))
            repeat += (1 - len(set(triples)) / len(triples)) if triples else 0.0
            scored = ids + ([model.cfg.eos_id] if model.cfg.eos_id in all_ids else [])
            prompt = batch["prompt"][row, : batch["prompt_lengths"][row]].tolist()
            sequence = torch.tensor([prompt + scored], device=device)
            with autocast():
                logits, _ = rater(
                    sequence[:, :-1],
                    torch.ones_like(sequence[:, :-1], dtype=torch.bool),
                )
            start = len(prompt) - 1
            score += float(
                F.cross_entropy(
                    logits[:, start:].float().flatten(0, 1),
                    sequence[:, len(prompt) :].flatten(),
                    reduction="sum",
                )
            )
            token_count += len(scored)
            texts.append(
                {
                    "prompt": tokenizer.decode(prompt),
                    "continuation": tokenizer.decode(ids),
                    "length": len(ids),
                    "eos": model.cfg.eos_id in all_ids,
                }
            )
        output["seeds"].append(
            {
                "seed": seed,
                "teacher_latent_mse": teacher_mse,
                "teacher_standardized_rms": float(
                    model.normalize(teacher)[valid].square().mean().sqrt()
                ),
                "shuffled_latent_mse": wrong_mse,
                "context_mse_advantage": wrong_mse - teacher_mse,
                "first_block_max_distance": first_distance,
                "teacher_sample_reference_ce": float(
                    F.cross_entropy(
                        teacher_logits.flatten(0, 1),
                        labels.flatten(),
                        ignore_index=-100,
                    )
                ),
                "free_sample_reference_ce": float(
                    F.cross_entropy(
                        free_logits.flatten(0, 1), labels.flatten(), ignore_index=-100
                    )
                ),
                "free_rater_ce": score / token_count,
                "free_repeat_trigram_fraction": repeat / len(labels),
                "free_mean_length": sum(x["length"] for x in texts) / len(texts),
                "texts": texts,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("samples must be at least 2")
    torch.set_num_threads(4)
    device = select_device("auto")
    autocast = precision_context(device, args.precision)
    source = torch.load(
        "runs/token-ar-control/last.pt", map_location="cpu", weights_only=True
    )
    if source["dataset_hash"] != sha256(Path("data/prepared/metadata.json")):
        raise ValueError("Rater dataset mismatch")
    rater = TokenAR(ModelConfig(**source["model_config"])).to(device).eval()
    rater.load_state_dict(source["model"], strict=True)
    dataset = StoryDataset(Path("data/prepared"), "dev", 16, 128, 64, 0)
    batch = to_device(dataset.batch(random.Random(30042), args.samples), device)
    reports = []
    for path in args.checkpoints:
        report = evaluate(path, batch, rater, autocast, device)
        reports.append(report)
        print(
            json.dumps(
                {
                    **report,
                    "seeds": [
                        {k: v for k, v in row.items() if k != "texts"}
                        for row in report["seeds"]
                    ],
                }
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "data_seed": 30042,
                "samples": args.samples,
                "precision": args.precision,
                "notes": [
                    "All prediction branches use history and independent initial noise only; never noisy current targets.",
                    "Single-sample reference CE is diagnostic, not model PPL.",
                    "The local token LM score is only a fluency proxy; all unfiltered texts are included.",
                ],
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
