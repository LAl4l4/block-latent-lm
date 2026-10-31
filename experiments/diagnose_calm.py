"""在 CPU 上对 CALM 检查点计算条件、无条件与真实后验参考 Energy。"""

import argparse
import json
from pathlib import Path
import random

import torch

from latent_ar.calm import posterior_energy_statistics
from latent_ar.checkpoints import CALM_ARCHITECTURE, checkpoint_architecture, load_model
from latent_ar.codec_data import CodecDataset
from latent_ar.data import StoryDataset


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--bank-blocks", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.blocks < 2 or args.bank_blocks < 1:
        parser.error("至少需要 2 个验证块和 1 个训练块")
    torch.set_num_threads(4)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_architecture(source) != CALM_ARCHITECTURE:
        raise ValueError("检查点不是 CALM")
    step = source["step"]
    training = source["config"]["training"]
    source.pop("optimizer", None)
    model = load_model(source, torch.device("cpu")).eval()
    del source
    cfg = model.cfg
    dataset = StoryDataset(
        args.data, "dev", training["min_prompt_tokens"],
        training["max_prompt_tokens"], training["max_target_tokens"], cfg.pad_id,
    )
    contexts, means, log_stds, total = [], [], [], 0
    # 固定验证窗口；有效块含真实短尾块，与训练指标的定义一致。
    for batch in dataset.evaluation_batches(16, None, 10042):
        h, mu, log_std, valid = model.teacher_contexts(**batch)
        contexts.append(h[valid])
        means.append(mu[valid])
        log_stds.append(log_std[valid])
        total += int(valid.sum())
        if total >= args.blocks:
            break
    if total < args.blocks:
        raise ValueError(f"验证集只有 {total} 个有效块")
    h, mu, log_std = (torch.cat(items)[:args.blocks] for items in (contexts, means, log_stds))

    # 独立训练集 bank 只保留完整块，不借用验证目标。
    bank = CodecDataset(args.data, "train", cfg.chunk_size, cfg.pad_id)
    rng = random.Random(10043)
    bank_mean, bank_log_std, bank_size = [], [], 0
    while bank_size < args.bank_blocks:
        batch = bank.batch(rng, 64)
        full = batch["lengths"] == cfg.chunk_size
        if not full.any():
            continue
        bm, bs, _ = model.codec.encode_posterior(batch["ids"][full], batch["lengths"][full])
        bank_mean.append(bm[:, 0])
        bank_log_std.append(bs[:, 0])
        bank_size += int(full.sum())
    bank_mean = torch.cat(bank_mean)[:args.bank_blocks]
    bank_log_std = torch.cat(bank_log_std)[:args.bank_blocks]

    # 每组共用同一批目标后验样本；两个模型分支共用预测噪声。
    noise = torch.Generator(device="cpu").manual_seed(10044)
    n, m, d = 8, 100, cfg.dim
    target_eps = torch.randn(args.blocks, m, d, generator=noise)
    targets = mu[:, None].float() + log_std[:, None].float().exp() * target_eps
    input_noise = model.draw_noise((args.blocks, n), torch.device("cpu"), noise)
    predicted = model.latent_head(h[:, None], input_noise)
    shuffled = model.latent_head(h.roll(1, 0)[:, None], input_noise)
    indices = torch.randint(args.bank_blocks, (args.blocks, n), generator=noise)
    bank_eps = torch.randn(args.blocks, n, d, generator=noise)
    unconditional = bank_mean[indices].float() + bank_log_std[indices].float().exp() * bank_eps
    oracle_eps = torch.randn(args.blocks, n, d, generator=noise)
    oracle = mu[:, None].float() + log_std[:, None].float().exp() * oracle_eps

    def score(samples):
        energy, attraction, spread = posterior_energy_statistics(samples, targets)
        return {"energy": float(energy / args.blocks),
                "attraction": float(attraction / args.blocks),
                "spread": float(spread / args.blocks)}

    sigma = log_std.float().exp()
    scores = {"conditional": score(predicted), "shuffled_history": score(shuffled),
              "unconditional_bank": score(unconditional), "oracle_posterior": score(oracle)}
    report = {
        "checkpoint": str(args.checkpoint), "step": step, "device": "cpu",
        "dev_blocks": args.blocks, "bank_blocks": args.bank_blocks,
        "prediction_samples": n, "target_samples": m, "scores": scores,
        "target_std_mean": float(sigma.mean()),
        "target_variance_sum_mean": float(sigma.square().sum(-1).mean()),
        "target_mean_norm_mean": float(mu.float().norm(dim=-1).mean()),
        "target_sample_norm_mean": float(targets.norm(dim=-1).mean()),
        "predicted_sample_norm_mean": float(predicted.float().norm(dim=-1).mean()),
        "notes": [
            "真实后验参考知道未来目标，不是语言模型可达到的下界。",
            "打乱条件按验证块循环移位；同一故事的其他块可能仍相邻。",
            "所有分数共享同一批目标样本；无条件分布来自独立训练集完整块。",
            "有限样本导致估计有方差；欧氏 Energy 数值依赖 latent 尺度。",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
