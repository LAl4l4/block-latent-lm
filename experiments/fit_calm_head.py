"""冻结 CALM 主干，在固定少量块上检验生成头能否拟合已知目标。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from latent_ar.calm import posterior_energy_statistics
from latent_ar.checkpoints import CALM_ARCHITECTURE, checkpoint_architecture, load_model
from latent_ar.data import StoryDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, required=True, help="独立输出目录")
    parser.add_argument("--objective", choices=("energy", "mean_mse"), default="energy")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-head", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.blocks < 1 or args.learning_rate <= 0:
        parser.error("步数、块数和学习率均需为正")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("仅支持 CPU 或 CUDA")
    if device.type == "cpu":
        torch.set_num_threads(4)
    if (args.output / "report.json").exists() or (
        args.save_head and (args.output / "head.pt").exists()
    ):
        raise FileExistsError("输出报告或 head 检查点已存在，请选择新目录")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_architecture(source) != CALM_ARCHITECTURE:
        raise ValueError("需要 CALM 检查点")
    source_step = source["step"]
    training = source["config"]["training"]
    source.pop("optimizer", None)
    # load_model 构造独立模型；脚本绝不写回传入的检查点。
    model = load_model(source, device).eval()
    del source
    model.requires_grad_(False)
    model.latent_head.requires_grad_(True)
    cfg = model.cfg
    data = StoryDataset(
        args.data,
        args.split,
        training["min_prompt_tokens"],
        training["max_prompt_tokens"],
        training["max_target_tokens"],
        cfg.pad_id,
    )
    collected = [[], [], [], []]
    count = 0
    with torch.inference_mode():
        for batch in data.evaluation_batches(16, None, 10042):
            batch = {key: value.to(device) for key, value in batch.items()}
            h, mu, ls, valid = model.teacher_contexts(**batch)
            labels = F.pad(
                batch["labels"],
                (0, mu.shape[1] * cfg.chunk_size - batch["labels"].shape[1]),
                value=-100,
            )
            labels = labels.reshape(*mu.shape[:2], cfg.chunk_size)
            full = valid & labels.ne(-100).all(-1)
            for dest, value in zip(collected, (h, mu, ls, labels)):
                dest.append(value[full].detach().clone())
            count += int(full.sum())
            if count >= args.blocks:
                break
    if count < args.blocks:
        raise ValueError(f"只有 {count} 个完整块")
    # inference_mode 创建的张量不能作为需要梯度网络的输入，复制到普通模式。
    h, mu, ls, labels = (
        torch.cat(values)[: args.blocks].clone() for values in collected
    )
    noise_rng = torch.Generator(device=device).manual_seed(10043)
    evaluation_seed = 10044
    optimizer = torch.optim.AdamW(model.latent_head.parameters(), lr=args.learning_rate)
    use_amp = device.type == "cuda"

    def autocast():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else nullcontext()
        )

    @torch.inference_mode()
    def evaluate(step):
        fixed = torch.Generator(device=device).manual_seed(evaluation_seed)
        draw = model.draw_noise((args.blocks, cfg.energy_candidates), device, fixed)
        target_noise = torch.randn(
            (args.blocks, cfg.target_samples, cfg.dim), device=device, generator=fixed
        )
        with autocast():
            samples = model.latent_head(h[:, None], draw)
            targets = mu[:, None].float() + ls[:, None].float().exp() * target_noise
            score, attraction, spread = posterior_energy_statistics(samples, targets)
            # 每个 latent 对同一完整目标评分；不是可达语言似然。
            total_ce = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            exact = torch.zeros((), device=device)
            for begin in range(0, args.blocks, 4):
                z = samples[begin : begin + 4]
                y = labels[begin : begin + 4]
                b, n, _ = z.shape
                logits = model.codec.decode(z.reshape(b * n, 1, -1)).float()
                expanded = (
                    y[:, None, :]
                    .expand(b, n, cfg.chunk_size)
                    .reshape(b * n, cfg.chunk_size)
                )
                total_ce += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    expanded.reshape(-1),
                    reduction="sum",
                )
                # 原始 argmax 指标不屏蔽特殊 token。
                equal = logits.argmax(-1).eq(expanded)
                correct += equal.sum()
                exact += equal.all(-1).sum()
        samples_count = args.blocks * cfg.energy_candidates
        record = {
            "step": step,
            "energy": float(score / args.blocks),
            "attraction": float(attraction / args.blocks),
            "spread": float(spread / args.blocks),
            "sample_ce_per_token": float(total_ce / (samples_count * cfg.chunk_size)),
            "sample_token_accuracy": float(correct / (samples_count * cfg.chunk_size)),
            "sample_full_block_accuracy": float(exact / samples_count),
            "sample_norm_mean": float(samples.float().norm(dim=-1).mean()),
        }
        if args.objective == "mean_mse":
            with autocast():
                zero = torch.zeros(args.blocks, cfg.noise_dim, device=device)
                deterministic = model.latent_head(h, zero)
                record["zero_noise_mean_mse"] = float(
                    F.mse_loss(deterministic.float(), mu.float())
                )
                correct_zero, exact_zero = 0, 0
                for begin in range(0, args.blocks, 8):
                    logits = model.codec.decode(
                        deterministic[begin : begin + 8, None]
                    ).float()
                    matches = logits.argmax(-1).eq(labels[begin : begin + 8])
                    correct_zero += int(matches.sum())
                    exact_zero += int(matches.all(-1).sum())
                record["zero_noise_token_accuracy"] = correct_zero / (
                    args.blocks * cfg.chunk_size
                )
                record["zero_noise_full_block_accuracy"] = exact_zero / args.blocks
        return record

    result = [evaluate(0)]
    print(json.dumps(result[-1]), flush=True)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            if args.objective == "energy":
                eps = model.draw_noise(
                    (args.blocks, cfg.energy_candidates), device, noise_rng
                )
                y_noise = torch.randn(
                    (args.blocks, cfg.target_samples, cfg.dim),
                    device=device,
                    generator=noise_rng,
                )
                targets = mu[:, None].float() + ls[:, None].float().exp() * y_noise
                prediction = model.latent_head(h[:, None], eps)
                loss = posterior_energy_statistics(prediction, targets)[0] / args.blocks
            else:
                # 固定零噪声拟合已知后验均值，只测可达性，不是语言生成训练法。
                eps = torch.zeros(args.blocks, cfg.noise_dim, device=device)
                prediction = model.latent_head(h, eps)
                loss = F.mse_loss(prediction.float(), mu.float())
        loss.backward()
        optimizer.step()
        if step in {100, 300, args.steps}:
            values = evaluate(step)
            values["train_loss"] = float(loss.detach())
            result.append(values)
            print(json.dumps(values), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_checkpoint": str(args.checkpoint),
        "source_step": source_step,
        "objective": args.objective,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "blocks": args.blocks,
        "split": args.split,
        "device": str(device),
        "energy_candidates": cfg.energy_candidates,
        "posterior_samples": cfg.target_samples,
        "evaluations": result,
        "notes": [
            "固定块拟合不代表泛化，且已知目标不是实际未来预测。",
            "评估每次使用相同随机数；CE 与命中率针对已知参考块。",
            "mean_mse 训练用零噪声，但评估统一使用随机样本，与训练目标不同。",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.save_head:
        torch.save(
            {
                "head": model.latent_head.state_dict(),
                "source_step": source_step,
                "objective": args.objective,
                "steps": args.steps,
            },
            args.output / "head.pt",
        )


if __name__ == "__main__":
    main()
