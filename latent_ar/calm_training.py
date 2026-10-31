"""16-token VAE / CALM 训练入口，独立于历史确定性 codec 实验。

每次保存包含优化器、随机状态和数据来源；第一次 Ctrl+C 在更新完成后保存。
验证使用独立的样本与噪声 RNG，不改变后续训练抽样。
"""

import argparse
import hashlib
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.calm import CalmConfig, CalmModel
from latent_ar.vae import BlockVAE, VAEConfig
from latent_ar.checkpoints import load_model
from latent_ar.checkpoint_io import publish_checkpoint
from latent_ar.codec_data import CodecDataset
from latent_ar.control import GracefulStop
from latent_ar.data import StoryDataset
from train import (
    append_record,
    learning_rate,
    precision_context,
    select_device,
    to_device,
)


def dataset(stage, data, split, cfg, training):
    if stage == "vae":
        return CodecDataset(data, split, cfg.chunk_size, cfg.pad_id)
    return StoryDataset(
        data,
        split,
        training["min_prompt_tokens"],
        training["max_prompt_tokens"],
        training["max_target_tokens"],
        cfg.pad_id,
    )


def statistics(stage, model, batch, generator):
    if stage == "vae":
        return model.reconstruction_statistics(**batch, generator=generator)
    return model.prior_statistics(**batch, generator=generator)


def add(totals, values):
    for key, value in values.items():
        totals[key] = totals.get(key, 0) + (
            float(value.detach()) if isinstance(value, torch.Tensor) else value
        )


def metrics(stage, totals):
    if stage == "vae":
        return {
            "ce": totals["ce_sum"] / totals["tokens"],
            "kl_per_block": totals["kl_sum"] / totals["blocks"],
            "token_accuracy": totals["correct_tokens"] / totals["tokens"],
            "full_block_accuracy": totals["correct_full_blocks"]
            / max(1, totals["full_blocks"]),
            "tokens": totals["tokens"],
            "blocks": totals["blocks"],
        }
    return {
        key: totals[key + "_sum"] / totals["blocks"]
        for key in ("energy", "attraction", "spread")
    } | {"tokens": totals["tokens"], "blocks": totals["blocks"]}


def describe(stage, values):
    if stage == "vae":
        return (
            f"CE {values['ce']:.4f} | KL/块 {values['kl_per_block']:.2f} | "
            f"token {values['token_accuracy']:.2%} | 完整块 {values['full_block_accuracy']:.2%}"
        )
    return (
        f"Energy {values['energy']:.4f} | 距离 {values['attraction']:.4f} | "
        f"spread {values['spread']:.4f}"
    )


@torch.inference_mode()
def evaluate(stage, model, dev, training, device, autocast, blocks=None):
    was_training = model.training
    model.eval()
    rng = random.Random(1042)
    noise = torch.Generator(device=device).manual_seed(1043)
    totals, wrong = {}, {}
    prefix_width = min(4, model.cfg.chunk_size)
    brier_sum = torch.zeros(prefix_width, device=device)
    brier_count = torch.zeros(prefix_width, device=device)
    size = training.get("eval_batch_size", training["batch_size"])
    count = blocks or size * training["eval_batches"]
    try:
        for begin in range(0, count, size):
            batch = to_device(dev.batch(rng, min(size, count - begin)), device)
            state = noise.get_state()
            with autocast():
                add(totals, statistics(stage, model, batch, noise))
                if stage == "calm":
                    paired = torch.Generator(device=device)
                    paired.set_state(state)
                    add(
                        wrong,
                        model.prior_statistics(
                            **batch, shuffle_context=True, generator=paired
                        ),
                    )
                    # 两个独立预测的命中项减碰撞项；按块前缀分别评价1至4词。
                    contexts, mean, _, valid = model.teacher_contexts(**batch)
                    padded = F.pad(
                        batch["labels"],
                        (
                            0,
                            mean.shape[1] * model.cfg.chunk_size
                            - batch["labels"].shape[1],
                        ),
                        value=-100,
                    )
                    labels = padded.reshape(*valid.shape, model.cfg.chunk_size)[valid]
                    h = contexts[valid]
                    for offset in range(0, len(h), 16):
                        z1 = model.sample_latent(
                            h[offset : offset + 16], generator=noise
                        )
                        z2 = model.sample_latent(
                            h[offset : offset + 16], generator=noise
                        )
                        p1 = model.codec.decode(z1[:, None]).argmax(-1)[
                            :, :prefix_width
                        ]
                        p2 = model.codec.decode(z2[:, None]).argmax(-1)[
                            :, :prefix_width
                        ]
                        y = labels[offset : offset + 16, :prefix_width]
                        ok = y.ne(-100).long().cumprod(-1)
                        hit1 = p1.eq(y).long().cumprod(-1)
                        hit2 = p2.eq(y).long().cumprod(-1)
                        collision = p1.eq(p2).long().cumprod(-1)
                        brier_sum += ((hit1 + hit2 - collision) * ok).sum(0)
                        brier_count += ok.sum(0)
    finally:
        model.train(was_training)
    result = metrics(stage, totals)
    if stage == "calm":
        result["conditional_energy_advantage"] = (
            metrics(stage, wrong)["energy"] - result["energy"]
        )
        result["brier_prefix_reward"] = (brier_sum / brier_count.clamp_min(1)).tolist()
    return result


def run(stage):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(f"configs/{stage}-16.json"))
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--stop-after", type=int)
    for key in ("max-steps", "batch-size", "grad-accum", "eval-batches", "save-every"):
        parser.add_argument("--" + key, type=int)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if args.eval_only and not args.resume:
        parser.error("--eval-only requires --resume")
    saved = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume
        else None
    )
    config = (
        deepcopy(saved["config"])
        if saved
        else json.loads(args.config.read_text(encoding="utf-8"))
    )
    tr = config["training"]
    for key in ("max_steps", "batch_size", "grad_accum", "eval_batches", "save_every"):
        if getattr(args, key) is not None:
            tr[key] = getattr(args, key)
    for key in (
        "max_steps",
        "batch_size",
        "grad_accum",
        "eval_batches",
        "save_every",
        "eval_every",
        "log_every",
    ):
        if tr[key] < 1:
            raise ValueError(f"{key} must be positive")
    seed = tr["seed"]
    torch.set_num_threads(tr.get("cpu_threads", 4))
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = select_device(args.device)
    autocast = precision_context(device, tr["precision"])
    noise = torch.Generator(device=device).manual_seed(seed + 1)
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    # 同时核对数据实体，不能只相信没有变化的 metadata 文件。
    fingerprints = {}
    for split in ("train", "dev"):
        for suffix, key in (
            ("bin", "tokens_sha256"),
            ("offsets.npy", "offsets_sha256"),
        ):
            name = f"{split}.{suffix}"
            actual = sha256(args.data / name)
            expected = metadata.get("splits", {}).get(split, {}).get(key)
            if expected is not None and actual != expected:
                raise ValueError(f"Prepared data hash mismatch: {name}")
            fingerprints[name] = actual
    fingerprints["metadata"] = sha256(args.data / "metadata.json")
    data_hash = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode()
    ).hexdigest()
    if sha256(args.data / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match prepared data")
    provenance = None
    if saved:
        expected = BlockVAE.architecture if stage == "vae" else CalmModel.architecture
        if saved["architecture"] != expected or saved["dataset_hash"] != data_hash:
            raise ValueError("Resume stage or dataset mismatch")
        model = load_model(saved, device)
        provenance = saved.get("vae_source")
    else:
        values = dict(config["model"])
        values.update(
            vocab_size=metadata["vocab_size"],
            **{f"{k}_id": metadata["special_ids"][k] for k in ("pad", "bos", "eos")},
        )
        if stage == "vae":
            model = BlockVAE(VAEConfig(**values)).to(device)
        else:
            path = Path(config["vae_checkpoint"])
            source = torch.load(path, map_location="cpu", weights_only=True)
            if (
                source["architecture"] != BlockVAE.architecture
                or source["dataset_hash"] != data_hash
            ):
                raise ValueError("需要相同词表的 VAE 检查点；纯重建 codec 不能代替 VAE")
            # VAE结构由其检查点决定；只允许覆盖主干/生成头结构。
            merged = dict(source["model_config"])
            structural = (
                "chunk_size",
                "dim",
                "encoder_dim",
                "decoder_dim",
                "decoder_layers",
                "decoder_heads",
                "decoder_ff_dim",
                "vocab_size",
                "pad_id",
                "bos_id",
                "eos_id",
                "vae_kl_weight",
                "vae_kl_floor",
                "vae_input_dropout",
                "vae_latent_dropout",
            )
            for key in structural:
                if key in values and values[key] != merged[key]:
                    raise ValueError(f"Cannot override VAE field: {key}")
            merged.update(values)
            model = CalmModel(CalmConfig(**merged)).to(device)
            model.codec.load_state_dict(source["model"], strict=True)
            del source
            gate = config["quality_gate"]
            quality = evaluate(
                "vae",
                model.codec,
                dataset("vae", args.data, "dev", model.cfg, tr),
                tr,
                device,
                autocast,
                gate["eval_blocks"],
            )
            print("[VAE采样重建检查] " + describe("vae", quality), flush=True)
            if (
                quality["token_accuracy"] < gate["token_accuracy"]
                or quality["full_block_accuracy"] < gate["block_accuracy"]
            ):
                raise SystemExit("[VAE未达标] 请继续训练 VAE；尚未建立主干训练目录。")
            provenance = {
                "checkpoint": str(path.resolve()),
                "sha256": sha256(path),
                "quality": quality,
            }
    config["model"] = asdict(model.cfg)
    dev = dataset(stage, args.data, "dev", model.cfg, tr)
    if args.eval_only:
        print(
            json.dumps(
                evaluate(stage, model, dev, tr, device, autocast),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_dir = args.run_dir or (
        args.resume.parent if saved else Path("runs") / config["name"]
    )
    if not saved and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Use a new run directory or --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    tokenizer.save(str(run_dir / "tokenizer.json"))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=tr["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=tr["weight_decay"],
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and tr["precision"] == "fp16"
    )
    first, seen = 1, 0
    if saved:
        if saved["device_type"] != device.type:
            raise ValueError("精确续训需要保持相同设备类型")
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        rng.setstate(saved["train_rng"])
        torch.set_rng_state(saved["torch_rng"])
        noise.set_state(saved["noise_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first, seen = saved["step"] + 1, saved["tokens_seen"]
        del saved
    final = min(tr["max_steps"], args.stop_after or tr["max_steps"])
    if final < first:
        raise ValueError("No remaining updates")
    train = dataset(stage, args.data, "train", model.cfg, tr)
    print(
        f"[{stage}] 参数 {sum(p.numel() for p in model.parameters()):,} | batch {tr['batch_size']}×{tr['grad_accum']} | 块长 {model.cfg.chunk_size} | 每 {tr['save_every']} 步保存",
        flush=True,
    )
    if stage == "calm":
        print(
            f"[Energy] 每块 {model.cfg.energy_candidates} 个预测、{model.cfg.target_samples} 个后验目标；所有有效块计损失，训练不解码词表。",
            flush=True,
        )
    model.train()
    validation = None
    with GracefulStop() as stop:
        for step in range(first, final + 1):
            started = time.perf_counter()
            batches = [
                to_device(train.batch(rng, tr["batch_size"]), device)
                for _ in range(tr["grad_accum"])
            ]
            token_count = sum(int(b["labels"].ne(-100).sum()) for b in batches)
            block_count = sum(
                int(
                    (
                        (b["labels"].ne(-100).sum(-1) + model.cfg.chunk_size - 1)
                        // model.cfg.chunk_size
                    ).sum()
                )
                for b in batches
            )
            optimizer.zero_grad(set_to_none=True)
            totals = {}
            for batch in batches:
                with autocast():
                    stats = statistics(stage, model, batch, noise)
                    # VAE使用块CE之和+KL，再按总有效token归一化；完整块对应官方损失除K。
                    loss = (
                        stats["loss_sum"] / token_count
                        if stage == "vae"
                        else stats["energy_sum"] / block_count
                    )
                scaler.scale(loss).backward()
                add(totals, stats)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                tr["grad_clip"],
                error_if_nonfinite=not scaler.is_enabled(),
            )
            if tr.get("lr_schedule") == "constant":
                lr = tr["learning_rate"] * min(1.0, step / max(1, tr["warmup_steps"]))
            else:
                lr = learning_rate(step, tr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            seen += token_count
            values = metrics(stage, totals)
            append_record(
                run_dir / "metrics.jsonl",
                {
                    "kind": "train",
                    "step": step,
                    **values,
                    "lr": lr,
                    "grad_norm": float(norm),
                    "step_seconds": elapsed,
                    "tokens_seen": seen,
                },
            )
            if step == first or step % tr["log_every"] == 0:
                print(
                    f"[{stage}] {step}/{tr['max_steps']} | {describe(stage, values)} | {elapsed:.2f} s/step",
                    flush=True,
                )
            if step % tr["eval_every"] == 0 or step == final:
                validation = evaluate(stage, model, dev, tr, device, autocast)
                append_record(
                    run_dir / "metrics.jsonl",
                    {"kind": "dev", "step": step, **validation},
                )
                print(f"[验证 {stage}] {describe(stage, validation)}", flush=True)
                if stage == "calm":
                    print(
                        f"[条件优势] {validation['conditional_energy_advantage']:+.4f} | Brier前缀奖励 {validation['brier_prefix_reward']}",
                        flush=True,
                    )
                    example = to_device(dev.batch(random.Random(2042), 1), device)
                    with autocast():
                        generated = model.generate(
                            example["prompt"],
                            example["prompt_lengths"],
                            max_new_tokens=64,
                            generator=torch.Generator(device=device).manual_seed(2043),
                        )
                    append_record(
                        run_dir / "samples.jsonl",
                        {
                            "step": step,
                            "prompt": tokenizer.decode(
                                example["prompt"][
                                    0, : int(example["prompt_lengths"][0])
                                ].tolist()
                            ),
                            "continuation": tokenizer.decode(generated[0].tolist()),
                        },
                    )
            if step % tr["save_every"] == 0 or step == final or stop.requested:
                checkpoint = {
                    "architecture": model.architecture,
                    "model_config": asdict(model.cfg),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "config": config,
                    "dataset_hash": data_hash,
                    "train_rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "noise_rng": noise.get_state(),
                    "device_type": device.type,
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                    ),
                    "tokens_seen": seen,
                    "validation": validation,
                    "vae_source": provenance,
                }
                temporary = run_dir / "last.tmp"
                torch.save(checkpoint, temporary)
                path = publish_checkpoint(temporary, run_dir / "last.pt", step)
                print(f"[保存] step {step} → {path}", flush=True)
            if stop.requested:
                break
