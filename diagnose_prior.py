"""在固定 dev 窗口上区分几何评分、文本分布和回灌误差。"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from latent_ar.checkpoints import load_model
from download_data import sha256
from latent_ar.data import StoryDataset
from latent_ar.energy import energy_statistics
from train import precision_context, select_device, to_device


@torch.inference_mode()
def diagnose(model, batch, autocast, seed=1042):
    device = batch["prompt"].device
    rng = torch.Generator(device=device).manual_seed(seed)
    labels = batch["labels"]
    with autocast():
        contexts, targets, valid = model.teacher_contexts(
            batch["prompt"], batch["prompt_lengths"], labels
        )
        noise = model.draw_noise(
            (*valid.shape, model.cfg.energy_candidates), device, rng
        )
        samples = model.latent_head(contexts[:, :, None], noise).float()
        wrong = model.latent_head(contexts.roll(1, 0)[:, :, None], noise).float()
    normalized = model.normalize(targets)
    # 无条件经验分布：只用当前 dev 批次构造诊断基线，不用于训练。
    bank = normalized[valid]
    indices = torch.randint(len(bank), samples.shape[:-1], device=device, generator=rng)
    empirical = bank[indices]
    gaussian = torch.randn(samples.shape, device=device, generator=rng)
    results = {}
    for name, values in [
        ("model", samples),
        ("shuffled_context", wrong),
        ("unconditional_empirical", empirical),
        ("standard_gaussian", gaussian),
        ("zero_point", torch.zeros_like(samples)),
    ]:
        stats = energy_statistics(values, normalized, valid)
        results[name] = {
            key: float(stats[key]) / stats["blocks"]
            for key in ("energy_sum", "attraction_sum", "spread_sum")
        }
    token_valid = labels != -100
    for name, latent in [
        ("oracle", targets),
        ("model", model.denormalize(samples[:, :, 0])),
    ]:
        with autocast():
            logits = model.decode(latent, labels.shape[1]).float()
        prediction = logits.argmax(-1)
        probability = logits.softmax(-1)
        metrics = {
            "reference_ce": float(
                F.cross_entropy(
                    logits.flatten(0, 1), labels.flatten(), ignore_index=-100
                )
            ),
            "reference_accuracy": float(
                ((prediction == labels) & token_valid).sum() / token_valid.sum()
            ),
            "mean_max_probability": float(
                probability.max(-1).values[token_valid].mean()
            ),
        }
        results.setdefault(name, {}).update(metrics)
    # 所有输出位置使用相同噪声；首块相同是训练/推理接线的校验。
    with autocast():
        teacher = model.sample_latent(contexts, noise=noise[:, :, 0])
        free = model.rollout(
            batch["prompt"],
            batch["prompt_lengths"],
            targets.shape[1],
            noise=noise[:, :, 0],
        )
        free_logits = model.decode(free, labels.shape[1]).float()
    distance = (
        (model.normalize(teacher) - model.normalize(free)).square().mean(-1).sqrt()
    )
    results["rollout"] = {
        "first_block_max_distance": float(distance[:, 0].max()),
        "mean_teacher_free_distance": float(distance[valid].mean()),
        "reference_ce": float(
            F.cross_entropy(
                free_logits.flatten(0, 1), labels.flatten(), ignore_index=-100
            )
        ),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/energy-diagnosis.json")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["dataset_hash"] != sha256(Path("data/prepared/metadata.json")):
        raise ValueError("Checkpoint dataset mismatch")
    model = load_model(checkpoint, device).eval()
    training = checkpoint["config"]["training"]
    dataset = StoryDataset(
        Path("data/prepared"),
        "dev",
        training["min_prompt_tokens"],
        training["max_prompt_tokens"],
        64,
        model.cfg.pad_id,
    )
    batch = to_device(dataset.batch(random.Random(10042), args.batch_size), device)
    results = {
        "checkpoint": str(args.checkpoint),
        "step": checkpoint["step"],
        "batch_size": args.batch_size,
        "metrics": diagnose(
            model, batch, precision_context(device, training["precision"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
