"""在相同起点、数据和更新步数下比较 latent 主干的训练目标。

所有实验写入独立目录；保存完整断点，可用 train_prior.py 按记录的目标继续。
额外使用已有 V1 小模型辅助检查生成文本的语言概率，同时保留文本与重复指标。
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.checkpoints import ENERGY_ARCHITECTURE, load_model
from latent_ar.data import StoryDataset
from latent_ar.staged_training import aggregate, readable_metrics, save_stage
from train import (
    append_record,
    learning_rate,
    precision_context,
    select_device,
    to_device,
)


def repeated_trigrams(ids):
    grams = list(zip(ids, ids[1:], ids[2:]))
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


@contextmanager
def canonical_feedback(model):
    """只用于诊断：将反馈向量解码/重编码，输出当前块仍使用原向量。

    该路径等待 decoder，不满足异步目标，不能作为已完成的正式方案。
    """

    def canonicalize(module, args):
        if len(args) < 4 or args[3] is None:
            return
        latent, positions, valid, caches = args
        ids = model.codec.decode(latent).argmax(-1)
        lengths = torch.full((ids.shape[0],), ids.shape[1], device=ids.device)
        encoded, _ = model.codec.encode(ids, lengths)
        return encoded, positions, valid, caches

    hook = model.backbone.register_forward_pre_hook(canonicalize)
    try:
        yield
    finally:
        hook.remove()


@torch.inference_mode()
def reference_score(rater, prompt, lengths, sequences):
    width = max(map(len, sequences))
    labels = torch.full((len(sequences), width), -100, device=prompt.device)
    for row, ids in enumerate(sequences):
        labels[row, : len(ids)] = torch.tensor(ids, device=prompt.device)
    inputs = torch.full_like(labels, rater.cfg.pad_id)
    inputs[:, 0] = rater.cfg.bos_id
    inputs[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, rater.cfg.pad_id)
    logits = rater(prompt, lengths, inputs)
    loss = F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
        reduction="sum",
    )
    return float(loss), int((labels != -100).sum())


@torch.inference_mode()
def evaluate(model, dev, tokenizer, rater, device, autocast, project_feedback=False):
    model.eval()
    totals = {}
    rng = random.Random(10042)
    generator = torch.Generator(device=device).manual_seed(10043)
    for _ in range(4):
        batch = to_device(dev.batch(rng, 16), device)
        with autocast():
            options = {"generator": generator}
            if model.architecture == ENERGY_ARCHITECTURE:
                options["distribution_diagnostics"] = True
            stats = model.prior_statistics(
                batch["prompt"],
                batch["prompt_lengths"],
                batch["labels"],
                **options,
            )
        aggregate(totals, stats)
    metrics = readable_metrics("prior", totals)

    batch = to_device(next(dev.evaluation_batches(8, 1, 10042)), device)
    samples = []
    score_sum, score_tokens = 0.0, 0
    for seed in [42, 43, 44]:
        with autocast(), (
            canonical_feedback(model) if project_feedback else nullcontext()
        ):
            generated = model.generate(
                batch["prompt"],
                batch["prompt_lengths"],
                max_new_tokens=64,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
        sequences = []
        for row in range(len(generated)):
            ids = generated[row].tolist()
            ended = model.cfg.eos_id in ids
            if ended:
                ids = ids[: ids.index(model.cfg.eos_id) + 1]
            sequences.append(ids)
            content = ids[:-1] if ended else ids
            length = int(batch["prompt_lengths"][row])
            samples.append(
                {
                    "prompt_index": row,
                    "seed": seed,
                    "prompt": tokenizer.decode(batch["prompt"][row, :length].tolist()),
                    "reference": tokenizer.decode(
                        batch["labels"][row, batch["labels"][row] != -100].tolist()
                    ),
                    "text": tokenizer.decode(content),
                    "token_ids": content,
                    "length": len(content),
                    "eos": ended,
                    "repeated_trigram_fraction": repeated_trigrams(content),
                }
            )
        with autocast():
            value, count = reference_score(
                rater, batch["prompt"], batch["prompt_lengths"], sequences
            )
        score_sum += value
        score_tokens += count
    metrics.update(
        {
            "free_reference_lm_ce": score_sum / score_tokens,
            "free_mean_length": sum(row["length"] for row in samples) / len(samples),
            "free_early_eos_fraction": sum(
                row["eos"] and row["length"] < 8 for row in samples
            )
            / len(samples),
            "free_repeated_trigram_fraction": sum(
                row["repeated_trigram_fraction"] for row in samples
            )
            / len(samples),
            "free_cross_seed_unique_fraction": sum(
                len(
                    {
                        tuple(row["token_ids"])
                        for row in samples
                        if row["prompt_index"] == index
                    }
                )
                / 3
                for index in range(8)
            )
            / 8,
        }
    )
    model.train()
    return metrics, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("runs/prior-energy-v4-check/last.pt")
    )
    parser.add_argument("--rater", type=Path, default=Path("runs/train-b128/last.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/objective-ablation-v1")
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=[
            "energy",
            "token_ce",
            "block_mixture",
            "block_mixture_cycle",
            "token_head",
        ],
        default=["energy", "token_ce", "block_mixture"],
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-seed", type=int, default=1729)
    parser.add_argument("--noise-seed", type=int, default=2718)
    parser.add_argument("--cycle-weight", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_every) < 1:
        parser.error("Experiment sizes must be positive")

    torch.set_num_threads(4)
    device = select_device(args.device)
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    source.pop("optimizer", None)
    if source["architecture"] != ENERGY_ARCHITECTURE:
        parser.error("This ablation requires the same Energy-head starting checkpoint")
    rater_checkpoint = torch.load(args.rater, map_location="cpu", weights_only=True)
    if source["dataset_hash"] != rater_checkpoint["dataset_hash"] or source[
        "dataset_hash"
    ] != sha256(args.data / "metadata.json"):
        parser.error("Source, reference model and prepared data must match")
    rater = load_model(rater_checkpoint, device).eval().requires_grad_(False)
    del rater_checkpoint
    tokenizer = Tokenizer.from_file(str(args.source.parent / "tokenizer.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if args.resume and report_path.exists()
        else {
            "source": str(args.source.resolve()),
            "source_step": source["step"],
            "source_sha256": sha256(args.source),
            "rater": str(args.rater.resolve()),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "notes": [
                "Matched source weights, data-window RNG and candidate RNG; equal updates, not equal FLOPs.",
                "All objectives use eight candidates; token CE averages the candidate CE, mixture NLL averages whole-block probabilities first.",
                "Reference LM CE is an auxiliary local-model fluency proxy, not a semantic score; inspect repetition, EOS and samples too.",
                "Mixture likelihood uses finite candidates and a softmax decoder, not exact argmax-generation likelihood.",
            ],
            "arms": {},
        }
    )
    autocast = precision_context(device, "bf16")
    report.setdefault("data_seed", args.data_seed)
    report.setdefault("noise_seed", args.noise_seed)
    report.setdefault("cycle_weight", args.cycle_weight)
    for objective in args.objectives:
        objective_type = (
            "block_mixture" if objective == "block_mixture_cycle" else objective
        )
        run_dir = args.output / objective
        previous = None
        if args.resume:
            previous = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=True
            )
            if (
                previous["model_config"].get("objective", "token_head")
                != objective_type
            ):
                raise ValueError("Objective mismatch")
            model = load_model(previous, device)
            config = deepcopy(previous["config"])
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"{run_dir}: use --resume or a new output")
            start = dict(source)
            start["model_config"] = {
                **source["model_config"],
                "objective": objective_type,
                "cycle_weight": (
                    args.cycle_weight if objective == "block_mixture_cycle" else 0.0
                ),
            }
            if objective == "token_head":
                from latent_ar.token_head import LatentTokenHeadModel, TokenHeadConfig

                allowed = {field.name for field in fields(TokenHeadConfig)}
                cfg = TokenHeadConfig(
                    **{
                        key: value
                        for key, value in source["model_config"].items()
                        if key in allowed
                    }
                )
                torch.manual_seed(42)
                model = LatentTokenHeadModel(cfg).to(device)
                state = {
                    key: value
                    for key, value in source["model"].items()
                    if not key.startswith("latent_head.")
                }
                missing, unexpected = model.load_state_dict(state, strict=False)
                if unexpected or any(
                    not key.startswith("latent_head.") for key in missing
                ):
                    raise ValueError("Only the new output head may differ")
            else:
                model = load_model(start, device)
            config = deepcopy(source["config"])
            config["name"] = run_dir.name
            config["model"] = asdict(model.cfg)
            config["training"].update(
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                max_steps=10000,
                warmup_steps=50,
                min_lr_ratio=1.0,
                learning_rate=0.0003,
                eval_batches=4,
                save_every=100,
            )
            config["ablation_source"] = {
                "checkpoint": report["source"],
                "step": source["step"],
                "sha256": report["source_sha256"],
            }
        model = model.to(device).train()
        train_cfg = config["training"]
        if objective == "token_head":
            config["prior_type"] = "token_head"
        if (
            train_cfg["batch_size"] != args.batch_size
            or train_cfg["grad_accum"] != args.grad_accum
        ):
            raise ValueError("Resume batch must match the experiment")
        train = StoryDataset(args.data, "train", 16, 128, 64, model.cfg.pad_id)
        dev = StoryDataset(args.data, "dev", 16, 128, 64, model.cfg.pad_id)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=0.0003,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=device.type == "cuda",
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        rng = random.Random(args.data_seed)
        torch.manual_seed(args.noise_seed)
        first_step, tokens_seen, blocks_seen = 1, 0, 0
        if previous:
            optimizer.load_state_dict(previous["optimizer"])
            rng.setstate(previous["train_rng"])
            torch.set_rng_state(previous["torch_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(
                    [value.cpu() for value in previous["cuda_rng"]]
                )
            first_step = previous["step"] + 1
            tokens_seen, blocks_seen = previous["tokens_seen"], previous["blocks_seen"]
            del previous
        run_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(run_dir / "tokenizer.json"))
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        records = report["arms"].setdefault(objective, [])
        if first_step == 1:
            metrics, samples = evaluate(model, dev, tokenizer, rater, device, autocast)
            records.append({"step": 0, **metrics})
            (run_dir / "samples-step-0.json").write_text(
                json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"arm": objective, "step": 0, **metrics}), flush=True)
        validation = None
        for step in range(first_step, args.steps + 1):
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batches = [
                train.batch(rng, args.batch_size) for _ in range(args.grad_accum)
            ]
            token_count = sum(int((b["labels"] != -100).sum()) for b in batches)
            block_count = sum(
                int((((b["labels"] != -100).sum(-1) + 3) // 4).sum()) for b in batches
            )
            cycle_count = sum(
                int(((b["labels"] != -100).sum(-1) // model.cfg.chunk_size).sum())
                for b in batches
            )
            totals = {}
            for raw in batches:
                batch = to_device(raw, device)
                with autocast():
                    stats = model.prior_statistics(
                        batch["prompt"], batch["prompt_lengths"], batch["labels"]
                    )
                    loss = stats["objective_sum"] / (
                        block_count if objective == "energy" else token_count
                    )
                    if "weighted_cycle_sum" in stats:
                        loss = loss + stats["weighted_cycle_sum"] / max(1, cycle_count)
                loss.backward()
                aggregate(totals, stats)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            lr = learning_rate(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tokens_seen += token_count
            blocks_seen += block_count
            record = {
                "kind": "train",
                "step": step,
                **readable_metrics("prior", totals),
                "seconds": time.perf_counter() - began,
                "grad_norm": float(norm),
            }
            append_record(run_dir / "metrics.jsonl", record)
            if step == first_step or step % 25 == 0:
                print(json.dumps({"arm": objective, **record}), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, samples = evaluate(
                    model, dev, tokenizer, rater, device, autocast
                )
                validation = {"step": step, **metrics}
                records.append(validation)
                (run_dir / f"samples-step-{step}.json").write_text(
                    json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    json.dumps({"arm": objective, "kind": "dev", **validation}),
                    flush=True,
                )
            if step % 100 == 0 or step == args.steps:
                save_stage(
                    run_dir / "last.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    config,
                    source["dataset_hash"],
                    rng,
                    tokens_seen,
                    blocks_seen,
                    validation,
                    source["codec_source"],
                )
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[结果] {report_path}", flush=True)


if __name__ == "__main__":
    main()
