"""codec / prior 两个独立训练阶段共用的保存、评价与运行流程。"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.checkpoints import (
    CODEC_ARCHITECTURE,
    PRIOR_ARCHITECTURE,
    ENERGY_ARCHITECTURE,
    MIXTURE_ARCHITECTURE,
    DENOISING_ARCHITECTURE,
    TOKEN_HEAD_ARCHITECTURE,
    TRAINABLE_PRIOR_ARCHITECTURES,
    checkpoint_architecture,
    load_model,
)
from latent_ar.codec import BlockCodec, reconstruction_statistics
from latent_ar.codec_data import CodecDataset
from latent_ar.control import GracefulStop
from latent_ar.checkpoint_io import publish_checkpoint
from latent_ar.data import StoryDataset
from latent_ar.model import ModelConfig
from latent_ar.staged import LatentPriorModel, StagedModelConfig
from latent_ar.energy import EnergyModelConfig, LatentEnergyModel
from latent_ar.mixture import MixtureModelConfig, LatentMixtureModel
from latent_ar.denoising import DenoisingConfig, LatentDenoisingModel
from latent_ar.token_head import TokenHeadConfig, LatentTokenHeadModel
from train import (
    append_record,
    learning_rate,
    precision_context,
    select_device,
    to_device,
)


class CodecQualityError(RuntimeError):
    """预期的阶段质量检查失败，与程序实现错误分开报告。"""


def parse_arguments(stage):
    parser = argparse.ArgumentParser(description=f"训练 {stage} 阶段")
    parser.add_argument("--config", type=Path, default=Path(f"configs/{stage}.json"))
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    for name in ["max-steps", "batch-size", "grad-accum", "save-every", "eval-batches"]:
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    if stage == "prior":
        parser.add_argument("--codec-checkpoint", type=Path)
        parser.add_argument(
            "--init-prior",
            type=Path,
            help="只继承旧 prior 的主干；新生成头和优化器从头开始",
        )
        parser.add_argument("--min-token-accuracy", type=float)
        parser.add_argument("--min-block-accuracy", type=float)
    args = parser.parse_args()
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if args.split == "test" and not args.eval_only:
        parser.error("Only explicit --eval-only can access test")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after must be positive")
    if args.resume and getattr(args, "init_prior", None):
        parser.error("--init-prior 用于新实验，不能与 --resume 同时使用")
    return args


def configuration(args, stage):
    checkpoint = None
    if args.resume:
        # 普通读取，不长期映射文件，避免 Windows 不能替换正在映射的断点。
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected = (
            {CODEC_ARCHITECTURE} if stage == "codec" else TRAINABLE_PRIOR_ARCHITECTURES
        )
        if checkpoint_architecture(checkpoint) not in expected:
            raise ValueError(f"该检查点不能用于 {stage} 阶段续训")
        config = deepcopy(checkpoint["config"])
        # 早期小头消融继承了 Energy 实验的配置名；以严格加载的架构纠正元数据。
        if checkpoint_architecture(checkpoint) == TOKEN_HEAD_ARCHITECTURE:
            config["prior_type"] = "token_head"
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    for key in ["max_steps", "batch_size", "grad_accum", "save_every", "eval_batches"]:
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    for key in [
        "max_steps",
        "batch_size",
        "grad_accum",
        "save_every",
        "eval_every",
        "eval_batches",
        "log_every",
    ]:
        if training[key] < 1:
            raise ValueError(f"{key} must be positive")
    if stage == "prior":
        if config.get("prior_type", "ar") not in {
            "ar",
            "energy",
            "mixture",
            "denoising",
            "token_head",
        }:
            raise ValueError("Unknown prior_type")
        if config.get("prior_type", "ar") in {"ar", "denoising"}:
            for key in ["latent_mse_weight", "token_ce_weight"]:
                training.setdefault(key, 1.0)
                if not 0 <= training[key] < float("inf"):
                    raise ValueError(f"{key} must be nonnegative and finite")
            if training["latent_mse_weight"] + training["token_ce_weight"] <= 0:
                raise ValueError("At least one prior loss weight must be positive")
        for argument, key in [
            ("min_token_accuracy", "token_accuracy"),
            ("min_block_accuracy", "block_accuracy"),
        ]:
            value = getattr(args, argument)
            if value is not None:
                config["quality_gate"][key] = value
        gate = config["quality_gate"]
        if not 0 <= gate["token_accuracy"] <= 1 or not 0 <= gate["block_accuracy"] <= 1:
            raise ValueError("Codec accuracy thresholds must be in [0, 1]")
        if gate["eval_blocks"] < 1 or config.get("latent_stats_blocks", 4096) < 1:
            raise ValueError("Quality/statistics sample counts must be positive")
    return config, checkpoint


def codec_dataset(data, split, cfg, limit=None):
    return CodecDataset(data, split, cfg.chunk_size, cfg.pad_id, limit)


def stage_dataset(stage, data, split, cfg, training, limit=None):
    if stage == "codec":
        return codec_dataset(data, split, cfg, limit)
    return StoryDataset(
        data,
        split,
        training["min_prompt_tokens"],
        training["max_prompt_tokens"],
        training["max_target_tokens"],
        cfg.pad_id,
        limit,
    )


def loss_and_statistics(
    stage,
    model,
    batch,
    generator=None,
    distribution_diagnostics=False,
    token_ce_grad=True,
):
    if stage == "codec":
        logits = model(batch["ids"], batch["lengths"])
        stats = reconstruction_statistics(logits, batch["labels"])
        stats["units"] = stats["tokens"]
        return stats
    options = (
        {"generator": generator, "distribution_diagnostics": distribution_diagnostics}
        if model.architecture == ENERGY_ARCHITECTURE
        else {}
    )
    if model.architecture in {MIXTURE_ARCHITECTURE, TOKEN_HEAD_ARCHITECTURE}:
        options["generator"] = generator
    if model.architecture in {PRIOR_ARCHITECTURE, DENOISING_ARCHITECTURE}:
        options["token_ce_grad"] = token_ce_grad
        if model.architecture == DENOISING_ARCHITECTURE:
            options["generator"] = generator
    return model.prior_statistics(
        batch["prompt"], batch["prompt_lengths"], batch["labels"], **options
    )


def aggregate(totals, stats):
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            value = float(value.detach())
        totals[key] = totals.get(key, 0) + value


def readable_metrics(stage, totals):
    if stage == "codec":
        return {
            "ce": totals["loss_sum"] / totals["tokens"],
            "token_accuracy": totals["correct_tokens"] / totals["tokens"],
            "block_accuracy": totals["correct_blocks"] / totals["blocks"],
            "full_block_accuracy": (
                totals["correct_full_blocks"] / totals["full_blocks"]
                if totals["full_blocks"]
                else 0.0
            ),
            "full_blocks": totals["full_blocks"],
            "tokens": totals["tokens"],
            "blocks": totals["blocks"],
        }
    if "text_nll_sum" in totals:
        return {
            "text_nll": totals["text_nll_sum"] / totals["tokens"],
            "tokens": totals["tokens"],
            "blocks": totals["blocks"],
            "scored_blocks": totals["scored_blocks"],
        }
    if "energy_sum" in totals:
        metrics = {
            "energy": totals["energy_sum"] / totals["blocks"],
            "attraction": totals["attraction_sum"] / totals["blocks"],
            "candidate_spread": totals["spread_sum"] / totals["blocks"],
            "sample_token_ce": totals["sample_ce_sum"] / totals["tokens"],
            "objective_loss": totals["objective_sum"] / totals["objective_count"],
            "tokens": totals["tokens"],
            "blocks": totals["blocks"],
        }
        if "mixture_block_nll_sum" in totals:
            metrics["mixture_token_ce"] = (
                totals["mixture_token_ce_sum"] / totals["tokens"]
            )
            metrics["mixture_block_nll"] = (
                totals["mixture_block_nll_sum"] / totals["blocks"]
            )
        if "cycle_sum" in totals:
            count = max(1, totals["cycle_blocks"])
            metrics["cycle_mse"] = totals["cycle_sum"] / count
            metrics["objective_loss"] += totals["weighted_cycle_sum"] / count
        return metrics
    return {
        "latent_mse": totals["latent_mse_sum"] / totals["blocks"],
        "token_ce": totals["token_ce_sum"] / totals["tokens"],
        "tokens": totals["tokens"],
        "blocks": totals["blocks"],
    }


@torch.inference_mode()
def evaluate_stage(stage, model, dataset, batch_size, batches, seed, device, autocast):
    was_training = model.training
    model.eval()
    data_rng = random.Random(seed)
    totals = {}
    wrong_totals = {}
    noise_rng = torch.Generator(device=device).manual_seed(seed + 1)
    energy = model.architecture == ENERGY_ARCHITECTURE
    try:
        for _ in range(batches):
            batch = to_device(dataset.batch(data_rng, batch_size), device)
            options = {}
            if energy or model.architecture == DENOISING_ARCHITECTURE:
                # 正确/错配条件使用同一批随机候选，验证也不消耗训练的全局 RNG。
                paired_rng = torch.Generator(device=device)
                paired_rng.set_state(noise_rng.get_state())
                options["generator"] = paired_rng
            with autocast():
                stats = loss_and_statistics(
                    stage, model, batch, noise_rng, distribution_diagnostics=energy
                )
                if stage == "prior" and batch_size > 1:
                    # 错配历史，检查主干是否实际使用条件；不是生成质量分数。
                    wrong_stats = model.prior_statistics(
                        batch["prompt"],
                        batch["prompt_lengths"],
                        batch["labels"],
                        shuffle_context=True,
                        **options,
                    )
                    aggregate(wrong_totals, wrong_stats)
            aggregate(totals, stats)
    finally:
        model.train(was_training)
    metrics = readable_metrics(stage, totals)
    if wrong_totals:
        wrong = readable_metrics(stage, wrong_totals)
        if energy:
            metrics["shuffled_context_energy"] = wrong["energy"]
            metrics["context_energy_advantage"] = wrong["energy"] - metrics["energy"]
        elif model.architecture in {MIXTURE_ARCHITECTURE, TOKEN_HEAD_ARCHITECTURE}:
            metrics["shuffled_context_text_nll"] = wrong["text_nll"]
            metrics["context_nll_advantage"] = wrong["text_nll"] - metrics["text_nll"]
        else:
            metrics["shuffled_context_latent_mse"] = wrong["latent_mse"]
            metrics["shuffled_context_token_ce"] = wrong["token_ce"]
            metrics["context_ce_advantage"] = wrong["token_ce"] - metrics["token_ce"]
    return metrics


@torch.inference_mode()
def codec_quality(codec, dataset, block_count, seed, device, autocast):
    """精确评价指定块数，包括最后不足一个 batch 的情况。"""
    was_training = codec.training
    codec.eval()
    rng = random.Random(seed)
    totals = {}
    try:
        for start in range(0, block_count, 128):
            batch = to_device(dataset.batch(rng, min(128, block_count - start)), device)
            with autocast():
                stats = loss_and_statistics("codec", codec, batch)
            aggregate(totals, stats)
    finally:
        codec.train(was_training)
    return readable_metrics("codec", totals)


def initialize_prior(args, config, metadata, device, autocast):
    source_path = args.codec_checkpoint or Path(config["codec_checkpoint"])
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if checkpoint_architecture(source) != CODEC_ARCHITECTURE:
        raise ValueError("--codec-checkpoint 必须来自 train_codec.py")
    if source["dataset_hash"] != sha256(args.data / "metadata.json"):
        raise ValueError("Codec 使用的数据或词表与当前 prior 数据不一致")

    model_dict = {**source["model_config"], **config["model"]}
    codec_fields = [
        "vocab_size",
        "pad_id",
        "bos_id",
        "eos_id",
        "chunk_size",
        "dim",
        "encoder_dim",
        "decoder_dim",
        "decoder_layers",
        "decoder_heads",
        "decoder_ff_dim",
    ]
    for field in codec_fields:
        if model_dict[field] != source["model_config"][field]:
            raise ValueError(f"Prior 不能修改已冻结 codec 的结构参数：{field}")
    energy = config.get("prior_type", "ar") == "energy"
    mixture = config.get("prior_type") == "mixture"
    cfg_type = (
        MixtureModelConfig
        if mixture
        else (EnergyModelConfig if energy else StagedModelConfig)
    )
    model_type = (
        LatentMixtureModel
        if mixture
        else (LatentEnergyModel if energy else LatentPriorModel)
    )
    if config.get("prior_type") == "denoising":
        cfg_type, model_type = DenoisingConfig, LatentDenoisingModel
    if config.get("prior_type") == "token_head":
        cfg_type, model_type = TokenHeadConfig, LatentTokenHeadModel
    cfg = cfg_type(**model_dict)
    model = model_type(cfg).to(device)
    model.codec.load_state_dict(source["model"], strict=True)

    # 先证明 codec 可以还原文字，再建立主干训练文件和优化器。
    gate = config["quality_gate"]
    dev = codec_dataset(args.data, "dev", cfg)
    quality = codec_quality(
        model.codec, dev, gate["eval_blocks"], 1042, device, autocast
    )
    print(
        f"[重建检查] token {quality['token_accuracy']:.2%} | "
        f"完整四 token 块 {quality['full_block_accuracy']:.2%} | {quality['blocks']} 块",
        flush=True,
    )
    if (
        quality["token_accuracy"] < gate["token_accuracy"]
        or quality["full_block_accuracy"] < gate["block_accuracy"]
    ):
        raise CodecQualityError(
            f"Codec 尚未达到重建要求（token≥{gate['token_accuracy']:.2%}，"
            f"完整四 token 块≥{gate['block_accuracy']:.2%}）。请先继续 train_codec.py；"
            "达到重建要求后再训练主干。"
        )

    # 坐标归一化统计只来自训练块，不使用 dev/test。
    train = codec_dataset(
        args.data, "train", cfg, config["training"].get("train_limit")
    )
    rng = random.Random(2042)
    bank = []
    count = config.get("latent_stats_blocks", 4096)
    for start in range(0, count, 128):
        batch = to_device(train.batch(rng, min(128, count - start)), device)
        with torch.no_grad(), autocast():
            latents, _ = model.codec.encode(batch["ids"], batch["lengths"])
        bank.append(latents[:, 0].float())
    model.set_latent_statistics(torch.cat(bank))
    config["model"] = asdict(cfg)
    provenance = {
        "checkpoint": str(source_path.resolve()),
        "step": source["step"],
        "sha256": sha256(source_path),
        "quality": quality,
        "latent_stats_blocks": count,
        "latent_stats_split": "train",
    }
    init_path = getattr(args, "init_prior", None)
    if init_path:
        provenance["backbone_init"] = initialize_backbone(
            model, init_path, sha256(args.data / "metadata.json")
        )
        config["init_prior"] = str(init_path.resolve())
    return model, provenance


def initialize_backbone(model, path, dataset_hash):
    """显式继承相同 codec 的历史主干；不把旧线性头/优化器冒充新实验。"""
    source = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint_architecture(source) not in TRAINABLE_PRIOR_ARCHITECTURES:
        raise ValueError("--init-prior requires a staged AR/Energy prior")
    if source["dataset_hash"] != dataset_hash:
        raise ValueError("Initial prior dataset/tokenizer does not match")
    state = source["model"]
    for key, value in model.codec.state_dict().items():
        if "codec." + key not in state or not torch.equal(
            value.cpu(), state["codec." + key]
        ):
            raise ValueError("Initial prior must use exactly the same frozen codec")
    backbone = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone, strict=True)
    model.latent_mean.copy_(state["latent_mean"])
    model.latent_std.copy_(state["latent_std"])
    print(
        f"[初始化] 继承 {path} 的主干（step {source['step']}）；生成头和优化器重新训练",
        flush=True,
    )
    return {
        "checkpoint": str(path.resolve()),
        "step": source["step"],
        "sha256": sha256(path),
    }


def save_stage(
    path,
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
):
    checkpoint = {
        "format_version": 3,
        "architecture": model.architecture,
        "model_config": asdict(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config,
        "dataset_hash": dataset_hash,
        "train_rng": data_rng.getstate(),
        "device_type": next(model.parameters()).device.type,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "tokens_seen": tokens_seen,
        "blocks_seen": blocks_seen,
        "validation": validation,
        "codec_source": provenance,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    return publish_checkpoint(temporary, path, step)


def describe(stage, values):
    if stage == "codec":
        return (
            f"CE {values['ce']:.4f} | token {values['token_accuracy']:.2%} | "
            f"四 token 块 {values['full_block_accuracy']:.2%}"
        )
    if "text_nll" in values:
        text = f"文字联合 NLL/token {values['text_nll']:.4f}"
        if "context_nll_advantage" in values:
            text += f" | 条件 NLL 优势 {values['context_nll_advantage']:+.4f}"
        return text
    if "energy" in values:
        text = f"Energy {values['energy']:.4f} | spread {values['candidate_spread']:.4f} | 样本 CE {values['sample_token_ce']:.4f}"
        if "context_energy_advantage" in values:
            text += f" | 条件 Energy 优势 {values['context_energy_advantage']:+.4f}"
        if "mixture_block_nll" in values:
            text += f" | 块混合 NLL {values['mixture_block_nll']:.4f}"
        return text
    text = f"latent MSE {values['latent_mse']:.4f} | token CE {values['token_ce']:.4f}"
    if "context_ce_advantage" in values:
        text += f" | 条件 CE 优势 {values['context_ce_advantage']:+.4f}"
    return text


@torch.inference_mode()
def save_sample(stage, model, batch, tokenizer, run_dir, step, device, autocast, seed):
    if stage == "codec":
        with autocast():
            prediction = model(batch["ids"][:1], batch["lengths"][:1]).argmax(-1)[0]
        length = int(batch["lengths"][0])
        record = {
            "step": step,
            "text": tokenizer.decode(batch["ids"][0, :length].tolist()),
            "reconstructed": tokenizer.decode(prediction[:length].tolist()),
        }
    else:
        with autocast():
            output = model.generate(
                batch["prompt"][:1],
                batch["prompt_lengths"][:1],
                max_new_tokens=batch["labels"].shape[1],
                generator=torch.Generator(device=device).manual_seed(seed),
                temperature=1.0 if model.architecture == MIXTURE_ARCHITECTURE else 0.0,
            )
        length = int(batch["prompt_lengths"][0])
        record = {
            "step": step,
            "prompt": tokenizer.decode(batch["prompt"][0, :length].tolist()),
            "continuation": tokenizer.decode(output[0].tolist()),
            "latent_prediction": model.architecture,
            "seed": seed,
        }
    append_record(run_dir / "samples.jsonl", record)


def run_stage(stage):
    args = parse_arguments(stage)
    config, checkpoint = configuration(args, stage)
    training = config["training"]
    seed = training["seed"]
    torch.set_num_threads(training.get("cpu_threads", 4))
    random.seed(seed)
    torch.manual_seed(seed)
    data_rng = random.Random(seed)
    device = select_device(args.device)
    autocast = precision_context(device, training["precision"])
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    dataset_hash = sha256(args.data / "metadata.json")
    if sha256(args.data / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match metadata")

    provenance = None
    if checkpoint:
        if checkpoint["dataset_hash"] != dataset_hash:
            raise ValueError("Checkpoint does not match prepared data")
        model = load_model(checkpoint, device)
        provenance = checkpoint.get("codec_source")
    elif stage == "codec":
        model_dict = dict(config["model"])
        model_dict["vocab_size"] = metadata["vocab_size"]
        for name in ["pad", "bos", "eos"]:
            model_dict[f"{name}_id"] = metadata["special_ids"][name]
        model = BlockCodec(ModelConfig(**model_dict)).to(device)
        config["model"] = asdict(model.cfg)
    else:
        model, provenance = initialize_prior(args, config, metadata, device, autocast)

    dev = stage_dataset(
        stage, args.data, args.split if args.eval_only else "dev", model.cfg, training
    )
    print(
        f"[{stage}] 参数 {sum(p.numel() for p in model.parameters()):,} | "
        f"可训练 {sum(p.numel() for p in model.parameters() if p.requires_grad):,} | "
        f"batch {training['batch_size']}×{training['grad_accum']} | "
        f"每 {training['save_every']} 步保存",
        flush=True,
    )
    if stage == "prior":
        objective = (
            f"{model.cfg.objective} | 每块 {model.cfg.energy_candidates} 个候选"
            if model.architecture == ENERGY_ARCHITECTURE
            else (
                "精确 latent 混合文字联合 NLL（训练均匀抽块，验证全量）"
                if model.architecture == MIXTURE_ARCHITECTURE
                else (
                    "主模型小输出头的块内联合文字 CE（decoder 冻结还原）"
                    if model.architecture == TOKEN_HEAD_ARCHITECTURE
                    else f"latent MSE × {training['latent_mse_weight']:g} + token CE × {training['token_ce_weight']:g}"
                )
            )
        )
        print(f"[目标] {objective}", flush=True)
        if model.architecture == DENOISING_ARCHITECTURE:
            print(
                "[去噪] 训练读取带噪目标，验证 loss 是去噪误差；自由生成从纯噪声开始，只回灌 latent。",
                flush=True,
            )
    if args.eval_only:
        metrics = evaluate_stage(
            stage,
            model,
            dev,
            training["batch_size"],
            training["eval_batches"],
            seed + 10000,
            device,
            autocast,
        )
        print(f"[{args.split}] {describe(stage, metrics)}", flush=True)
        return

    dataset = stage_dataset(
        stage, args.data, "train", model.cfg, training, training.get("train_limit")
    )
    run_dir = args.run_dir or (
        args.resume.parent if args.resume else Path("runs") / config["name"]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint and any(run_dir.iterdir()):
        raise FileExistsError("Use a new run directory or --resume")
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    tokenizer.save(str(run_dir / "tokenizer.json"))
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
    first_step, tokens_seen, blocks_seen = 1, 0, 0
    validation = None
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        data_rng.setstate(checkpoint["train_rng"])
        # 旧 codec 断点保存的是 noise_device；其 codec 权重和优化目标没有改变。
        if checkpoint.get("device_type", checkpoint.get("noise_device")) != device.type:
            raise ValueError(
                "Exact staged resume currently requires the same device type"
            )
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng"]]
            )
        first_step = checkpoint["step"] + 1
        tokens_seen, blocks_seen = checkpoint["tokens_seen"], checkpoint["blocks_seen"]
        validation = checkpoint.get("validation")
        print(f"[恢复] 从 step {first_step} 继续", flush=True)
    final_step = min(training["max_steps"], args.stop_after or training["max_steps"])
    if final_step < first_step:
        raise ValueError("No remaining steps; increase --max-steps/--stop-after")

    last_saved_path = args.resume
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    with GracefulStop() as stop:
        for step in range(first_step, final_step + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batches = [
                dataset.batch(data_rng, training["batch_size"])
                for _ in range(training["grad_accum"])
            ]
            token_count = sum(int((batch["labels"] != -100).sum()) for batch in batches)
            if stage == "prior":
                cycle_count = sum(
                    int(((b["labels"] != -100).sum(-1) // model.cfg.chunk_size).sum())
                    for b in batches
                )
                block_count = sum(
                    int(
                        (
                            (
                                (batch["labels"] != -100).sum(-1)
                                + model.cfg.chunk_size
                                - 1
                            )
                            // model.cfg.chunk_size
                        ).sum()
                    )
                    for batch in batches
                )
            totals = {}
            for batch in batches:
                batch = to_device(batch, device)
                with autocast():
                    stats = loss_and_statistics(
                        stage,
                        model,
                        batch,
                        token_ce_grad=training.get("token_ce_weight", 1.0) > 0,
                    )
                    if stage == "codec":
                        loss = stats["loss_sum"] / token_count
                    elif model.architecture == ENERGY_ARCHITECTURE:
                        denominator = (
                            block_count
                            if model.cfg.objective == "energy"
                            else token_count
                        )
                        loss = stats["objective_sum"] / denominator
                        if "weighted_cycle_sum" in stats:
                            loss = loss + stats["weighted_cycle_sum"] / max(
                                1, cycle_count
                            )
                    elif model.architecture in {
                        MIXTURE_ARCHITECTURE,
                        TOKEN_HEAD_ARCHITECTURE,
                    }:
                        loss = stats["text_nll_sum"] / token_count
                    else:
                        # 用整个累计 batch 的分母，尾块和 padding 不改变权重。
                        loss = (
                            training["latent_mse_weight"]
                            * stats["latent_mse_sum"]
                            / block_count
                            + training["token_ce_weight"]
                            * stats["token_ce_sum"]
                            / token_count
                        )
                scaler.scale(loss).backward()
                aggregate(totals, stats)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["grad_clip"], error_if_nonfinite=True
            )
            lr = learning_rate(step, training)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_start
            tokens_seen += totals["tokens"]
            blocks_seen += totals["blocks"]
            metrics = readable_metrics(stage, totals)
            record = {
                "kind": "train",
                "stage": stage,
                "step": step,
                **metrics,
                "lr": lr,
                "grad_norm": float(norm),
                "step_seconds": elapsed,
                "tokens_seen": tokens_seen,
                "blocks_seen": blocks_seen,
            }
            append_record(run_dir / "metrics.jsonl", record)
            if (
                step == first_step
                or step % training["log_every"] == 0
                or step == final_step
                or stop.requested
            ):
                print(
                    f"[{stage}] {step}/{training['max_steps']} | {describe(stage, metrics)} | {elapsed:.2f} s/step",
                    flush=True,
                )

            if (
                step % training["eval_every"] == 0 or step == final_step
            ) and not stop.requested:
                metrics = evaluate_stage(
                    stage,
                    model,
                    dev,
                    training["batch_size"],
                    training["eval_batches"],
                    seed + 10000,
                    device,
                    autocast,
                )
                validation = {"step": step, **metrics}
                append_record(
                    run_dir / "metrics.jsonl",
                    {"kind": "dev", "stage": stage, **validation},
                )
                print(f"[验证 {stage}] {describe(stage, metrics)}", flush=True)
                sample = to_device(dev.batch(random.Random(seed + 10000), 1), device)
                save_sample(
                    stage,
                    model,
                    sample,
                    tokenizer,
                    run_dir,
                    step,
                    device,
                    autocast,
                    seed + 20000,
                )
            if (
                step % training["save_every"] == 0
                or step == final_step
                or stop.requested
            ):
                last_saved_path = save_stage(
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
                print(f"[保存] step {step} → {last_saved_path}", flush=True)
            if stop.requested:
                break
        interrupted = stop.requested
    summary = {
        "stage": stage,
        "last_step": step,
        "tokens_seen": tokens_seen,
        "blocks_seen": blocks_seen,
        "interrupted": interrupted,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
        "checkpoint": str(last_saved_path),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[结束 {stage}] 最新检查点为 step {step}", flush=True)
