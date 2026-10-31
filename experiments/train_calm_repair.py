"""固定16-token和VAE，比较独立输出头；所有训练输出写入新目录。

flow使用多步连续生成，token对照使用局部GRU顺序选词。两者都不是原来
单次随机MLP的等价加速实现；本实验用于定位可训练的监督与采样接口。
"""

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from tokenizers import Tokenizer
from download_data import sha256

from latent_ar.checkpoints import load_model
from latent_ar.checkpoint_io import publish_checkpoint
from latent_ar.control import GracefulStop
from latent_ar.data import StoryDataset
from train import append_record, precision_context, to_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["flow", "token"], required=True)
    p.add_argument("--init", type=Path, default=Path("runs/calm-16/last.pt"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_every) < 1:
        p.error("训练参数必须为正")
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise FileExistsError("必须使用新的实验目录")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    data_rng = random.Random(args.seed)
    device = torch.device(args.device)
    noise = torch.Generator(device=device).manual_seed(args.seed + 1)
    source = torch.load(args.resume or args.init, map_location="cpu", weights_only=True)
    expected = (
        "latent_conditional_flow_v1"
        if args.method == "flow"
        else "latent_calm_token_v1"
    )
    if args.resume and source["architecture"] != expected:
        raise ValueError("续训方法与检查点架构不匹配")
    if not args.resume and source["architecture"] != "latent_calm_v1":
        raise ValueError("该对照要求从原 CALM 检查点继承主干")
    prepared = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    fingerprints = {
        f"{split}.{suffix}": sha256(args.data / f"{split}.{suffix}")
        for split in ("train", "dev")
        for suffix in ("bin", "offsets.npy")
    }
    fingerprints["metadata"] = sha256(args.data / "metadata.json")
    actual_hash = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode()
    ).hexdigest()
    if source["dataset_hash"] != actual_hash or prepared["tokenizer_sha256"] != sha256(
        args.data / "tokenizer.json"
    ):
        raise ValueError("数据或词表与初始化检查点不一致")
    if args.resume and source["device_type"] != device.type:
        raise ValueError("续训需要相同设备类型")
    config = dict(source["config"])
    initial_architecture = source["architecture"]
    if args.resume:
        model = load_model(source, device)
    else:
        values = dict(source["model_config"])
        if args.method == "flow":
            from latent_ar.conditional_flow import ConditionalFlowModel, FlowConfig

            model = ConditionalFlowModel(FlowConfig(**values)).to(device)
        else:
            from latent_ar.calm_token import CalmTokenConfig, CalmTokenModel

            model = CalmTokenModel(CalmTokenConfig(**values)).to(device)
        # 仅继承相同VAE、主干和输入压缩器；新头独立初始化。
        inherited = {
            k: v for k, v in source["model"].items() if not k.startswith("latent_head.")
        }
        missing, unexpected = model.load_state_dict(inherited, strict=False)
        assert not unexpected and all(k.startswith("latent_head.") for k in missing)
    config["model"] = asdict(model.cfg)
    config["repair_method"] = args.method
    config["init_checkpoint"] = str(args.init)
    config["training"].update(
        batch_size=args.batch_size, grad_accum=args.grad_accum, max_steps=args.steps
    )
    model.codec.eval()
    optimizer = torch.optim.AdamW(
        (x for x in model.parameters() if x.requires_grad),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device.type == "cuda",
    )
    autocast = precision_context(device, "bf16")
    first, seen = 1, 0
    if args.resume:
        optimizer.load_state_dict(source["optimizer"])
        data_rng.setstate(source["train_rng"])
        noise.set_state(source["noise_rng"])
        torch.set_rng_state(source["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(source["cuda_rng"])
        first, seen = source["step"] + 1, source["tokens_seen"]
    metadata = {k: source.get(k) for k in ("dataset_hash", "vae_source")}
    del source
    tr = config["training"]
    train = StoryDataset(
        args.data,
        "train",
        tr["min_prompt_tokens"],
        tr["max_prompt_tokens"],
        tr["max_target_tokens"],
        model.cfg.pad_id,
    )
    dev = StoryDataset(
        args.data,
        "dev",
        tr["min_prompt_tokens"],
        tr["max_prompt_tokens"],
        tr["max_target_tokens"],
        model.cfg.pad_id,
    )
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.run_dir / "tokenizer.json"))
    (args.run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    @torch.inference_mode()
    def evaluate(step):
        was = model.training
        model.eval()
        sums = {"loss": 0.0, "wrong": 0.0, "count": 0}
        rng = random.Random(1042)
        gen = torch.Generator(device=device).manual_seed(1043)
        for _ in range(8):
            batch = to_device(dev.batch(rng, 16), device)
            state = gen.get_state()
            with autocast():
                values = model.prior_statistics(**batch, generator=gen)
                paired = torch.Generator(device=device)
                paired.set_state(state)
                wrong = model.prior_statistics(
                    **batch, shuffle_context=True, generator=paired
                )
            denominator = (
                values["tokens"] if args.method == "token" else values["blocks"]
            )
            sums["count"] += denominator
            sums["loss"] += float(values["objective_sum"])
            sums["wrong"] += float(wrong["objective_sum"])
        record = {
            "kind": "dev",
            "step": step,
            "loss": sums["loss"] / sums["count"],
            "context_advantage": (sums["wrong"] - sums["loss"]) / sums["count"],
        }
        append_record(args.run_dir / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        for seed in (42, 43):
            text = "Once upon a time, Lily found a little box."
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            with autocast():
                output = model.generate(
                    torch.tensor([ids], device=device),
                    torch.tensor([len(ids)], device=device),
                    max_new_tokens=64,
                    generator=torch.Generator(device=device).manual_seed(seed),
                )
            append_record(
                args.run_dir / "samples.jsonl",
                {
                    "step": step,
                    "seed": seed,
                    "prompt": text,
                    "continuation": tokenizer.decode(output[0].tolist()),
                },
            )
        model.train(was)

    evaluate(first - 1)
    model.train()
    with GracefulStop() as stop:
        for step in range(first, args.steps + 1):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batches = [
                to_device(train.batch(data_rng, args.batch_size), device)
                for _ in range(args.grad_accum)
            ]
            tokens = sum(int(b["labels"].ne(-100).sum()) for b in batches)
            blocks = sum(
                int(
                    (
                        (b["labels"].ne(-100).sum(-1) + model.cfg.chunk_size - 1)
                        // model.cfg.chunk_size
                    ).sum()
                )
                for b in batches
            )
            loss_value = 0.0
            for batch in batches:
                with autocast():
                    values = model.prior_statistics(**batch, generator=noise)
                    loss = values["objective_sum"] / (
                        tokens if args.method == "token" else blocks
                    )
                loss.backward()
                loss_value += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            lr = 3e-4 * min(1.0, step / 100)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            seen += tokens
            record = {
                "kind": "train",
                "step": step,
                "loss": loss_value,
                "tokens": tokens,
                "blocks": blocks,
                "grad_norm": float(norm),
                "seconds": time.perf_counter() - begin,
            }
            append_record(args.run_dir / "metrics.jsonl", record)
            if step % 50 == 0 or step == first:
                print(json.dumps(record), flush=True)
            # 先保存再做生成，评估失败时也保留本次参数更新。
            if step % 100 == 0 or step == args.steps or stop.requested:
                checkpoint = {
                    "architecture": model.architecture,
                    "model_config": asdict(model.cfg),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                    "train_rng": data_rng.getstate(),
                    "noise_rng": noise.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                    ),
                    "device_type": device.type,
                    "tokens_seen": seen,
                    **metadata,
                }
                temporary = args.run_dir / "last.tmp"
                torch.save(checkpoint, temporary)
                publish_checkpoint(temporary, args.run_dir / "last.pt", step)
            if step % args.eval_every == 0 or step == args.steps:
                evaluate(step)
            if stop.requested:
                break


if __name__ == "__main__":
    main()
