"""只读分析 CALM 后验、条件预测及 decoder 的关系。"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from latent_ar.checkpoints import CALM_ARCHITECTURE, checkpoint_architecture, load_model
from latent_ar.data import StoryDataset


def describe_vector(values):
    values = values.float().flatten()
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocks < 2:
        parser.error("至少需要两个块")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_architecture(checkpoint) != CALM_ARCHITECTURE:
        raise ValueError("检查点不是 CALM")
    step = checkpoint["step"]
    training = checkpoint["config"]["training"]
    checkpoint.pop("optimizer", None)
    model = load_model(checkpoint, torch.device("cpu")).eval()
    del checkpoint
    cfg = model.cfg
    data = StoryDataset(
        args.data,
        "dev",
        training["min_prompt_tokens"],
        training["max_prompt_tokens"],
        training["max_target_tokens"],
        cfg.pad_id,
    )
    contexts, means, log_stds, labels = [], [], [], []
    first_batch = None
    found = 0
    # 仅完整块；固定窗口保证多次调用比较同一批数据。
    for batch in data.evaluation_batches(16, None, 10042):
        if first_batch is None:
            first_batch = batch
        h, mu, ls, valid = model.teacher_contexts(**batch)
        y = F.pad(
            batch["labels"],
            (0, mu.shape[1] * cfg.chunk_size - batch["labels"].shape[1]),
            value=-100,
        )
        y = y.reshape(*mu.shape[:2], cfg.chunk_size)
        full = valid & (y != -100).all(-1)
        for dest, value in zip((contexts, means, log_stds, labels), (h, mu, ls, y)):
            dest.append(value[full])
        found += int(full.sum())
        if found >= args.blocks:
            break
    if found < args.blocks:
        raise ValueError(f"只有 {found} 个完整验证块")
    h, mu, ls, y = (
        torch.cat(value)[: args.blocks] for value in (contexts, means, log_stds, labels)
    )
    generator = torch.Generator(device="cpu").manual_seed(10044)
    posterior = mu + ls.exp() * torch.randn(mu.shape, generator=generator)
    noise1 = model.draw_noise((args.blocks,), torch.device("cpu"), generator)
    noise2 = model.draw_noise((args.blocks,), torch.device("cpu"), generator)
    predicted = model.sample_latent(h, noise=noise1)
    shuffled = model.sample_latent(h.roll(1, 0), noise=noise1)
    different_noise = model.sample_latent(h, noise=noise2)

    def decode(latents):
        tokens, entropy, maximum = [], [], []
        for i in range(0, len(latents), 8):
            logits = model.codec.decode(latents[i : i + 8, None]).float()
            log_probs = logits.log_softmax(-1)
            probs = log_probs.exp()
            entropy.append(-(probs * log_probs).sum(-1))
            maximum.append(probs.max(-1).values)
            # 与生成路径保持一致，只屏蔽 pad/bos；熵仍是未屏蔽分布。
            logits[..., cfg.pad_id] = -torch.inf
            logits[..., cfg.bos_id] = -torch.inf
            tokens.append(logits.argmax(-1))
        return torch.cat(tokens), torch.cat(entropy), torch.cat(maximum)

    decoded = {
        name: decode(z)
        for name, z in {
            "posterior_mean": mu,
            "posterior_sample": posterior,
            "conditioned": predicted,
            "shuffled_history": shuffled,
            "conditioned_other_noise": different_noise,
        }.items()
    }
    decoding = {}
    for name, (tokens, ent, max_prob) in decoded.items():
        matches = tokens == y
        decoding[name] = {
            "token_accuracy": float(matches.float().mean()),
            "full_block_accuracy": float(matches.all(-1).float().mean()),
            "mean_entropy_nats": float(ent.mean()),
            "mean_max_probability": float(max_prob.mean()),
        }
    signal_variance = mu.float().var(0, unbiased=False)
    noise_variance = (2 * ls.float()).exp().mean(0)
    centered_posterior = posterior.float() - posterior.float().mean(0)
    centered_predicted = predicted.float() - predicted.float().mean(0)
    # 只计算对角协方差和迹，避免将 64 样本估计出的完整矩阵视为可靠。
    covariance = {
        "posterior_sample_trace": float(centered_posterior.square().mean(0).sum()),
        "predicted_sample_trace": float(centered_predicted.square().mean(0).sum()),
        "posterior_sample_diagonal": describe_vector(
            centered_posterior.square().mean(0)
        ),
        "predicted_sample_diagonal": describe_vector(
            centered_predicted.square().mean(0)
        ),
        "posterior_norm": describe_vector(posterior.norm(dim=-1)),
        "predicted_norm": describe_vector(predicted.norm(dim=-1)),
        "sample_mean_distance": float((posterior.mean(0) - predicted.mean(0)).norm()),
    }
    context_change = (predicted - shuffled).float().norm(dim=-1)
    noise_change = (predicted - different_noise).float().norm(dim=-1)
    token_context_match = (
        (decoded["conditioned"][0] == decoded["shuffled_history"][0]).float().mean()
    )
    token_noise_match = (
        (decoded["conditioned"][0] == decoded["conditioned_other_noise"][0])
        .float()
        .mean()
    )

    # 首块 teacher 与 free 使用相同 prompt 和噪声；不涉及真实未来。
    teacher_h = model.teacher_contexts(**first_batch)[0][:, 0]
    seed = 2043
    explicit_noise = model.draw_noise(
        (len(teacher_h),), torch.device("cpu"), torch.Generator().manual_seed(seed)
    )
    teacher_z = model.sample_latent(teacher_h, noise=explicit_noise)
    free = model.generate(
        first_batch["prompt"],
        first_batch["prompt_lengths"],
        max_new_tokens=cfg.chunk_size,
        generator=torch.Generator().manual_seed(seed),
        return_trace=True,
    )
    first = free["records"][0]
    report = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "dev_full_blocks": args.blocks,
        "dev_seed": 10042,
        "sample_seed": 10044,
        "decoder": decoding,
        "posterior_std_by_dimension": describe_vector(ls.exp().mean(0)),
        "signal_variance_by_dimension": describe_vector(signal_variance),
        "posterior_variance_by_dimension": describe_vector(noise_variance),
        "snr_by_dimension": describe_vector(
            signal_variance / noise_variance.clamp_min(1e-12)
        ),
        "covariance": covariance,
        "same_noise_shuffled_context_latent_delta": describe_vector(context_change),
        "same_context_different_noise_latent_delta": describe_vector(noise_change),
        "same_noise_shuffled_context_token_agreement": float(token_context_match),
        "same_context_different_noise_token_agreement": float(token_noise_match),
        "teacher_free_first_context_max_abs_diff": float(
            (teacher_h - first["context"]).abs().max()
        ),
        "teacher_free_first_latent_max_abs_diff": float(
            (teacher_z - first["z"]).abs().max()
        ),
        "notes": [
            "固定 teacher-forced 历史，预测匹配真实未来的命中率只供诊断。",
            "打乱上下文可能仍选到同一故事的其他块；不是严格独立文档配对。",
            "方差、SNR、协方差只基于少量块；逐维 SNR 为均值变化方差除后验噪声方差。",
            "熵和最大概率未屏蔽特殊 token；准确率与生成一致，屏蔽 PAD/BOS。",
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
