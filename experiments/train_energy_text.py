"""独立文字评分微调实验；原 Energy/混合 NLL 训练入口与断点保持各自语义。"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch

from download_data import sha256
from latent_ar.checkpoints import ENERGY_ARCHITECTURE, ENERGY_TEXT_ARCHITECTURE
from latent_ar.control import GracefulStop
from latent_ar.data import StoryDataset
from latent_ar.energy_text import EnergyTextConfig, LatentEnergyTextModel
from latent_ar.staged_training import aggregate, save_stage
from train import append_record, precision_context, select_device, to_device


def metrics(totals):
    return {
        "energy": totals["energy_sum"] / totals["blocks"],
        "text_brier_loss": totals["text_brier_sum"] / totals["blocks"],
        "sample_token_ce": totals["sample_ce_sum"] / totals["tokens"],
    }


@torch.inference_mode()
def evaluate(model, dataset, autocast, device):
    model.eval()
    batch = to_device(dataset.batch(random.Random(10042), 128), device)
    with autocast():
        stats = model.prior_statistics(
            batch["prompt"],
            batch["prompt_lengths"],
            batch["labels"],
            generator=torch.Generator(device=device).manual_seed(10043),
        )
    totals = {}
    aggregate(totals, stats)
    model.train()
    return metrics(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", type=Path)
    group.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--steps", type=int, default=1000, help="本实验累计完成的优化步数"
    )
    parser.add_argument("--text-weight", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("steps and learning-rate must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data_rng = random.Random(42)
    device = select_device("auto")
    source_path = args.resume or args.init
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    expected = ENERGY_TEXT_ARCHITECTURE if args.resume else ENERGY_ARCHITECTURE
    if source["architecture"] != expected:
        raise ValueError(
            "Use --init for an Energy checkpoint or --resume for this experiment"
        )
    dataset_hash = sha256(Path("data/prepared/metadata.json"))
    if source["dataset_hash"] != dataset_hash:
        raise ValueError("Dataset/tokenizer mismatch")
    cfg_dict = dict(source["model_config"])
    if not args.resume:
        cfg_dict.update(text_score_weight=args.text_weight, text_score_blocks=64)
    model = LatentEnergyTextModel(EnergyTextConfig(**cfg_dict)).to(device)
    model.load_state_dict(source["model"], strict=True)
    config = deepcopy(source["config"])
    config.update(prior_type="energy_text", model=asdict(model.cfg))
    training = config["training"]
    if not args.resume:
        training.update(learning_rate=args.learning_rate)
    autocast = precision_context(device, training["precision"])
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=training["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and training["precision"] == "fp16"
    )
    first, tokens_seen, blocks_seen = 1, 0, 0
    if args.resume:
        optimizer.load_state_dict(source["optimizer"])
        scaler.load_state_dict(source["scaler"])
        data_rng.setstate(source["train_rng"])
        if source["device_type"] != device.type:
            raise ValueError("Exact resume requires the same device type")
        torch.set_rng_state(source["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in source["cuda_rng"]])
        first = source["step"] + 1
        tokens_seen, blocks_seen = source["tokens_seen"], source["blocks_seen"]
    if args.steps < first:
        raise ValueError("No remaining steps")
    run_dir = args.run_dir or (
        args.resume.parent if args.resume else Path("runs/prior-energy-text-v5-check")
    )
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Use a new experiment directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = source.get("codec_source")
    config.setdefault(
        "text_score_init",
        {
            "checkpoint": str(source_path.resolve()),
            "step": source["step"],
            "sha256": sha256(source_path),
        },
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "tokenizer.json").write_bytes(
        (source_path.parent / "tokenizer.json").read_bytes()
    )
    datasets = {
        split: StoryDataset(
            Path("data/prepared"),
            split,
            training["min_prompt_tokens"],
            training["max_prompt_tokens"],
            training["max_target_tokens"],
            model.cfg.pad_id,
        )
        for split in ["train", "dev"]
    }
    if not args.resume:
        initial = evaluate(model, datasets["dev"], autocast, device)
        append_record(run_dir / "metrics.jsonl", {"kind": "dev", "step": 0, **initial})
        print(f"[dev 0] {initial}", flush=True)
    validation = source.get("validation") if args.resume else None
    model.train()
    with GracefulStop() as stop:
        for step in range(first, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batches = [
                to_device(
                    datasets["train"].batch(data_rng, training["batch_size"]), device
                )
                for _ in range(training["grad_accum"])
            ]
            blocks = sum(
                int(
                    (
                        ((b["labels"] != -100).sum(-1) + model.cfg.chunk_size - 1)
                        // model.cfg.chunk_size
                    ).sum()
                )
                for b in batches
            )
            totals = {}
            for batch in batches:
                with autocast():
                    stats = model.prior_statistics(
                        batch["prompt"], batch["prompt_lengths"], batch["labels"]
                    )
                    loss = (
                        stats["energy_sum"]
                        + model.cfg.text_score_weight * stats["text_brier_sum"]
                    ) / blocks
                scaler.scale(loss).backward()
                aggregate(totals, stats)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            # 本实验从成熟 prior 微调，只做 50 步 warmup，随后保持固定 LR。
            for group in optimizer.param_groups:
                group["lr"] = training["learning_rate"] * min(1.0, step / 50)
            scaler.step(optimizer)
            scaler.update()
            tokens_seen += totals["tokens"]
            blocks_seen += totals["blocks"]
            values = metrics(totals)
            append_record(
                run_dir / "metrics.jsonl",
                {"kind": "train", "step": step, "grad_norm": float(norm), **values},
            )
            if step % 10 == 0 or step == first:
                print(f"[train {step}] {values}", flush=True)
            if (step % 250 == 0 or step == args.steps) and not stop.requested:
                validation = {
                    "step": step,
                    **evaluate(model, datasets["dev"], autocast, device),
                }
                append_record(run_dir / "metrics.jsonl", {"kind": "dev", **validation})
                print(f"[dev] {validation}", flush=True)
            if step % 100 == 0 or step == args.steps or stop.requested:
                save_stage(
                    run_dir / "last.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    config,
                    dataset_hash,
                    data_rng,
                    tokens_seen,
                    blocks_seen,
                    validation,
                    provenance,
                )
            if stop.requested:
                break
    print(f"[saved] {run_dir / 'last.pt'} step {step}", flush=True)


if __name__ == "__main__":
    main()
