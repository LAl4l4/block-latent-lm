"""V2 训练：编码 prompt、自由展开 latent、用块内双向 decoder 计算 CE。

阅读顺序：main → train_one_step → evaluate / save_checkpoint。
模型结构在 latent_ar/model.py；数据切窗在 latent_ar/data.py。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.checkpoints import (
    BLOCK_ARCHITECTURE,
    CODEC_ARCHITECTURE,
    PRIOR_ARCHITECTURE,
    FLOW_ARCHITECTURE,
    ENERGY_ARCHITECTURE,
    checkpoint_architecture,
)
from latent_ar.control import GracefulStop
from latent_ar.data import StoryDataset
from latent_ar.model import LatentARModel, ModelConfig


def select_device(name: str) -> torch.device:
    """auto 优先用 GPU；也可以显式传入 cpu 或 cuda。"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def precision_context(device: torch.device, precision: str):
    """返回混合精度上下文的工厂函数；CPU 始终使用 FP32。"""
    if device.type != "cuda" or precision == "fp32":
        return nullcontext

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bf16; use fp16 or fp32")
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown precision: {precision}")

    return lambda: torch.autocast("cuda", dtype=dtype)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def append_record(path: Path, record: dict):
    """文件保留 JSONL，方便后续画曲线；终端则输出容易阅读的进度。"""
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def loss_sum(model, batch, use_latents):
    """只计算真实续写 token 的 CE 总和，padding 标签为 -100。"""
    logits = model(
        batch["prompt"],
        batch["prompt_lengths"],
        batch["labels"].shape[1],
        use_latents,
    )
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        batch["labels"].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


@torch.inference_mode()
def evaluate(model, dataset, batch_size, batches, seed, device, autocast, use_latents):
    """使用固定窗口评价；结束后恢复模型原来的训练/推理状态。"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    stories = 0

    try:
        for batch in dataset.evaluation_batches(batch_size, batches, seed):
            batch = to_device(batch, device)
            with autocast():
                loss = loss_sum(model, batch, use_latents)

            total_loss += float(loss)
            total_tokens += int((batch["labels"] != -100).sum())
            stories += batch["prompt"].shape[0]
    finally:
        model.train(was_training)

    if not total_tokens:
        raise RuntimeError("Empty evaluation set")

    ce = total_loss / total_tokens
    return {
        "ce": ce,
        "perplexity": math.exp(min(ce, 80)),
        "tokens": total_tokens,
        "stories": stories,
    }


def learning_rate(step: int, config: dict) -> float:
    """先线性预热，再余弦下降；step 指完成一次参数更新。"""
    total = config["max_steps"]
    warmup = min(config["warmup_steps"], max(0, total - 1))
    if step <= warmup:
        return config["learning_rate"] * step / max(1, warmup)

    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    ratio = config["min_lr_ratio"] + (1 - config["min_lr_ratio"]) * cosine
    return config["learning_rate"] * ratio


def save_checkpoint(
    path, model, optimizer, scaler, step, config, dataset_hash, train_rng, tokens_seen
):
    """保存完整续训状态，写完临时文件后才替换上一个可用断点。"""
    checkpoint = {
        "format_version": 2,
        "architecture": BLOCK_ARCHITECTURE,
        "model_config": asdict(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config,
        "dataset_hash": dataset_hash,
        "tokens_seen": tokens_seen,
        # 随机状态与优化器一样重要：恢复后继续原来的数据采样顺序。
        "train_rng": train_rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/overfit.json"))
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-steps", type=int, help="覆盖总更新步数，同时改变学习率计划"
    )
    parser.add_argument(
        "--stop-after", type=int, help="在指定步骤结束，保留原来的学习率计划"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument(
        "--save-every", type=int, help="自动保存间隔；默认配置为 100 步"
    )
    parser.add_argument("--ablation", choices=["full", "zero-latents", "no-latents"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--eval-batches", type=int, help="验证批次数；0 表示整个 split")
    args = parser.parse_args()

    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume CHECKPOINT")
    if args.split == "test" and not args.eval_only:
        parser.error("The test split is only available through explicit --eval-only")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    return args


def load_configuration(args):
    """恢复时先读取断点配置，再应用命令行明确指定的覆盖值。"""
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        architecture = checkpoint_architecture(checkpoint)
        if architecture in {
            CODEC_ARCHITECTURE,
            PRIOR_ARCHITECTURE,
            FLOW_ARCHITECTURE,
            ENERGY_ARCHITECTURE,
        }:
            entry = (
                "train_codec.py"
                if architecture == CODEC_ARCHITECTURE
                else "train_prior.py"
            )
            raise ValueError(f"该阶段请通过 {entry} --resume 恢复")
        if checkpoint_architecture(checkpoint) != BLOCK_ARCHITECTURE:
            raise ValueError(
                "旧版 AR decoder 检查点不能直接恢复为 V2。"
                "请用新实验目录重新训练；旧模型仍可通过 generate.py 生成。"
            )
        config = deepcopy(checkpoint["config"])
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    training = config["training"]
    training.setdefault("save_every", 100)
    override_keys = [
        "max_steps",
        "batch_size",
        "grad_accum",
        "save_every",
        "eval_batches",
    ]
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            training[key] = value

    if args.ablation:
        previous = config.get("ablation", "full")
        if checkpoint and args.ablation != previous:
            raise ValueError(
                "Train ablations in a separate run; do not change them on resume"
            )
        config["ablation"] = args.ablation
    config.setdefault("ablation", "full")

    positive_keys = [
        "max_steps",
        "batch_size",
        "grad_accum",
        "eval_every",
        "save_every",
        "log_every",
    ]
    for key in positive_keys:
        if training[key] < 1:
            raise ValueError(f"{key} must be positive")
    if training["eval_batches"] < 0:
        raise ValueError("eval_batches must be nonnegative")
    return config, checkpoint


def train_one_step(
    model,
    optimizer,
    scaler,
    dataset,
    train_rng,
    training,
    device,
    autocast,
    use_latents,
    step,
):
    """取若干小批次累积梯度，然后只更新一次参数。"""
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    batches = [
        dataset.batch(train_rng, training["batch_size"])
        for _ in range(training["grad_accum"])
    ]
    valid_tokens = sum(int((batch["labels"] != -100).sum()) for batch in batches)

    total_loss = 0.0
    for batch in batches:
        batch = to_device(batch, device)
        with autocast():
            value = loss_sum(model, batch, use_latents)

        total_loss += float(value.detach())
        # 各片段长度不同，按整次更新的有效 token 总数归一化。
        # 这里只 detach 用于日志的数值；主干回灌计算图一直保留。
        scaler.scale(value / valid_tokens).backward()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), training["grad_clip"], error_if_nonfinite=True
    )
    lr = learning_rate(step, training)
    for group in optimizer.param_groups:
        group["lr"] = lr
    scaler.step(optimizer)
    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "kind": "train",
        "step": step,
        "ce": total_loss / valid_tokens,
        "lr": lr,
        "grad_norm": float(grad_norm),
        "target_tokens": valid_tokens,
        "step_seconds": elapsed,
        "target_tokens_per_second": valid_tokens / elapsed,
    }


def evaluate_and_sample(
    model, dataset, training, device, autocast, use_latents, tokenizer, run_dir, step
):
    """dev 指标与固定 prompt 的自由生成，使用独立的采样随机数。"""
    seed = training["seed"]
    batches = training["eval_batches"] or None
    result = evaluate(
        model,
        dataset,
        training["batch_size"],
        batches,
        seed + 10000,
        device,
        autocast,
        use_latents,
    )
    append_record(run_dir / "metrics.jsonl", {"kind": "dev", "step": step, **result})
    print(
        f"[验证] step {step} | CE {result['ce']:.4f} | "
        f"PPL {result['perplexity']:.2f} | {result['stories']} 篇",
        flush=True,
    )

    prompt, _ = dataset.example(0, random.Random(seed + 10000))
    prompt_tensor = torch.tensor(prompt, device=device)[None]
    prompt_length = torch.tensor([len(prompt)], device=device)
    sample_rng = torch.Generator(device=device).manual_seed(seed + 20000)
    with autocast():
        generated = model.generate(
            prompt_tensor,
            prompt_length,
            max_new_tokens=training["max_target_tokens"],
            use_latents=use_latents,
            generator=sample_rng,
        )
    append_record(
        run_dir / "samples.jsonl",
        {
            "step": step,
            "prompt": tokenizer.decode(prompt.tolist()),
            "continuation": tokenizer.decode(generated[0].tolist()),
        },
    )


def main():
    args = parse_args()
    config, checkpoint = load_configuration(args)
    training = config["training"]
    seed = training["seed"]
    torch.set_num_threads(training.get("cpu_threads", 4))
    random.seed(seed)
    torch.manual_seed(seed)
    train_rng = random.Random(seed)
    device = select_device(args.device)
    autocast = precision_context(device, training["precision"])

    # 1. 确认数据和词表版本，防止用不兼容的数据恢复 checkpoint。
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    dataset_hash = sha256(args.data / "metadata.json")
    if sha256(args.data / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer does not match prepared data metadata")
    if checkpoint and checkpoint["dataset_hash"] != dataset_hash:
        raise RuntimeError("Checkpoint belongs to a different prepared dataset")

    # 2. 构建模型。恢复时沿用原结构；修改 batch 不会改变模型参数形状。
    model_dict = dict(config["model"])
    model_dict["vocab_size"] = metadata["vocab_size"]
    for name in ["pad", "bos", "eos"]:
        model_dict[f"{name}_id"] = metadata["special_ids"][name]
    cfg = ModelConfig(**model_dict)
    model = LatentARModel(cfg).to(device)
    if checkpoint:
        if asdict(cfg) != checkpoint["model_config"]:
            raise RuntimeError("Model configuration does not match checkpoint")
        model.load_state_dict(checkpoint["model"])

    use_latents = config["ablation"] == "full"
    if not use_latents:
        # V2 唯一内容入口是生成 latent。全零诊断不保留 prompt 旁路。
        model.encoder.requires_grad_(False)
        model.backbone.requires_grad_(False)
        model.latent_head.requires_grad_(False)

    parameters = sum(p.numel() for p in model.parameters())
    active_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"[模型] {gpu_name} | 参数 {parameters:,} | 可训练 {active_parameters:,}",
        flush=True,
    )
    print(
        f"[配置] batch {training['batch_size']} × 累计 {training['grad_accum']} "
        f"= {training['batch_size'] * training['grad_accum']} 个窗口/step | "
        f"每 {training['save_every']} 步自动保存",
        flush=True,
    )

    # 3. dev 用于训练检查；只有显式 --eval-only --split test 才读取 test。
    data_args = {
        "directory": args.data,
        "min_prompt": training["min_prompt_tokens"],
        "max_prompt": training["max_prompt_tokens"],
        "max_target": training["max_target_tokens"],
        "pad_id": cfg.pad_id,
    }
    evaluation_split = args.split if args.eval_only else "dev"
    dev = StoryDataset(split=evaluation_split, **data_args)
    if args.eval_only:
        result = evaluate(
            model,
            dev,
            training["batch_size"],
            training["eval_batches"] or None,
            seed + 10000,
            device,
            autocast,
            use_latents,
        )
        print(json.dumps({"split": args.split, "step": checkpoint["step"], **result}))
        return
    train_data = StoryDataset(
        split="train", limit=training.get("train_limit"), **data_args
    )

    # 4. 每次实验独立目录；恢复时默认继续写入原目录。
    default_run = args.resume.parent if args.resume else Path("runs") / config["name"]
    run_dir = args.run_dir or default_run
    run_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint and any(run_dir.iterdir()):
        raise FileExistsError(
            "Run directory is not empty; use a new --run-dir or --resume"
        )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    tokenizer.save(str(run_dir / "tokenizer.json"))

    # 5. 同时恢复优化器与随机状态，避免只有权重恢复、训练状态却重置。
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=training["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=training["weight_decay"],
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and training["precision"] == "fp16"
    )
    first_step = 1
    tokens_seen = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        train_rng.setstate(checkpoint["train_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng"]]
            )
        first_step = checkpoint["step"] + 1
        tokens_seen = checkpoint["tokens_seen"]
        print(
            f"[恢复] 已保存 step {first_step - 1}，从 step {first_step} 继续。",
            flush=True,
        )

    final_step = min(training["max_steps"], args.stop_after or training["max_steps"])
    if final_step < first_step:
        raise ValueError("No remaining steps; increase --max-steps/--stop-after")

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_saved_step = first_step - 1 if checkpoint else 0
    completed_step = first_step - 1

    def save_progress(step, reason):
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scaler,
            step,
            config,
            dataset_hash,
            train_rng,
            tokens_seen,
        )
        print(f"[保存] step {step}（{reason}）→ {run_dir / 'last.pt'}", flush=True)

    # 6. 每一步都先完成梯度累计与更新，再进入验证/保存等安全边界。
    # 第一次 Ctrl+C 不打断半个更新，而是在这里保存完整状态后退出。
    with GracefulStop() as stop:
        for step in range(first_step, final_step + 1):
            record = train_one_step(
                model,
                optimizer,
                scaler,
                train_data,
                train_rng,
                training,
                device,
                autocast,
                use_latents,
                step,
            )
            completed_step = step
            tokens_seen += record["target_tokens"]
            record["tokens_seen"] = tokens_seen
            append_record(run_dir / "metrics.jsonl", record)

            should_log = (
                step == first_step
                or step % training["log_every"] == 0
                or step == final_step
                or stop.requested
            )
            if should_log:
                print(
                    f"[训练] {step}/{training['max_steps']} | CE {record['ce']:.4f} | "
                    f"lr {record['lr']:.2e} | {record['step_seconds']:.2f} s/step | "
                    f"{record['target_tokens_per_second']:,.0f} token/s",
                    flush=True,
                )

            should_save = (
                step % training["save_every"] == 0
                or step == final_step
                or stop.requested
            )
            if should_save:
                reason = "中断" if stop.requested else "自动"
                save_progress(step, reason)
                last_saved_step = step

            # 先保存再验证，避免验证或生成期间退出时丢掉刚完成的更新。
            should_evaluate = step % training["eval_every"] == 0 or step == final_step
            if should_evaluate and not stop.requested:
                evaluate_and_sample(
                    model,
                    dev,
                    training,
                    device,
                    autocast,
                    use_latents,
                    tokenizer,
                    run_dir,
                    step,
                )

            # 信号也可能在验证期间到达；此时仍须保存当前完整步骤。
            if stop.requested:
                if last_saved_step != step:
                    save_progress(step, "中断")
                break

        interrupted = stop.requested

    summary = {
        "last_step": completed_step,
        "tokens_seen": tokens_seen,
        "interrupted": interrupted,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "checkpoint": str(run_dir / "last.pt"),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    status = "已安全中断" if interrupted else "训练结束"
    print(f"[结束] {status}，最新检查点为 step {completed_step}。", flush=True)


if __name__ == "__main__":
    main()
