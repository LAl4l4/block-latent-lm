"""在固定验证块上诊断冻结的混合 prior，可使用 CPU 运行。"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from latent_ar.checkpoints import MIXTURE_ARCHITECTURE, checkpoint_architecture, load_model
from latent_ar.data import StoryDataset


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.blocks < 2:
        parser.error("--blocks must be at least 2 for shuffled comparison")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_architecture(checkpoint) != MIXTURE_ARCHITECTURE:
        raise ValueError("This diagnostic requires a mixture prior checkpoint")
    checkpoint.pop("optimizer", None)
    step = checkpoint.get("step")
    model = load_model(checkpoint, device).eval()
    del checkpoint
    cfg = model.cfg
    dataset = StoryDataset(args.data, "dev", 16, 128, 64, cfg.pad_id)

    # 按顺序读取验证故事，每篇使用固定随机种子，保证窗口可复现。
    contexts, labels, targets = [], [], []
    found = 0
    for batch in dataset.evaluation_batches(16, None, 10042):
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx, target, _ = model.teacher_contexts(
            batch["prompt"], batch["prompt_lengths"], batch["labels"]
        )
        block_labels = F.pad(
            batch["labels"],
            (0, target.shape[1] * cfg.chunk_size - batch["labels"].shape[1]),
            value=-100,
        ).reshape(*target.shape[:2], cfg.chunk_size)
        full = (block_labels != -100).all(-1)
        contexts.append(ctx[full])
        targets.append(target[full])
        labels.append(block_labels[full])
        found += int(full.sum())
        if found >= args.blocks:
            break
    if found < args.blocks:
        raise ValueError(f"Only {found} complete dev blocks available")
    context = torch.cat(contexts)[: args.blocks]
    target = torch.cat(targets)[: args.blocks]
    label = torch.cat(labels)[: args.blocks]

    # 分批解码，避免词表 logits 占用过多 CPU 内存。
    def assess(condition, project_radius=None):
        result = {key: [] for key in ("joint", "weights", "responsibilities", "position", "dist", "norm", "normalized_norm")}
        for begin in range(0, len(condition), 8):
            ctx = condition[begin : begin + 8]
            y = label[begin : begin + 8]
            normalized_candidates, logits_w = model.latent_head(ctx)
            candidates = model.denormalize(normalized_candidates)
            if project_radius is not None:
                # 统一投影到验证集目标平均范数，只用于离线诊断。
                candidates = F.normalize(candidates.float(), dim=-1) * project_radius
            logits = model.codec.decoder(candidates).float()
            log_probs = logits.log_softmax(-1)
            selected = log_probs.gather(
                -1, y[:, None, :, None].expand(-1, candidates.shape[1], -1, 1)
            ).squeeze(-1)
            log_weights = logits_w.float().log_softmax(-1)
            joint_scores = log_weights + selected.sum(-1)
            result["joint"].append(-torch.logsumexp(joint_scores, -1))
            result["weights"].append(log_weights.exp())
            result["responsibilities"].append(joint_scores.softmax(-1))
            result["position"].append(-torch.logsumexp(log_weights[:, :, None] + selected, 1))
            # 只取上三角，排除自身配对与重复方向。
            matrix = torch.cdist(candidates.float(), candidates.float())
            i, j = torch.triu_indices(candidates.shape[1], candidates.shape[1], 1, device=device)
            result["dist"].append(matrix[:, i, j])
            result["norm"].append(candidates.float().norm(dim=-1))
            result["normalized_norm"].append(normalized_candidates.float().norm(dim=-1))
        return {key: torch.cat(value) for key, value in result.items()}

    main_result = assess(context)
    shuffled = assess(context.roll(1, 0))
    target_radius = target.float().norm(dim=-1).mean()
    projected = assess(context, project_radius=target_radius)
    oracle_logits = model.codec.decode(target[:, None]).float()
    oracle_ce = F.cross_entropy(oracle_logits.flatten(0, 1), label.flatten()).item()
    oracle_accuracy = (oracle_logits.argmax(-1) == label).float().mean().item()
    weights = main_result["weights"]
    responsibilities = main_result["responsibilities"]
    winners = torch.bincount(responsibilities.argmax(-1), minlength=weights.shape[1])
    report = {
        "checkpoint": str(args.checkpoint), "step": step, "device": str(device),
        "blocks": args.blocks, "chunk_size": cfg.chunk_size, "dev_seed": 10042,
        "oracle_reconstruction_ce_per_token": oracle_ce,
        "oracle_reconstruction_accuracy": oracle_accuracy,
        "mixture_joint_nll_per_token": (main_result["joint"].mean() / cfg.chunk_size).item(),
        "shuffled_context_joint_nll_per_token": (shuffled["joint"].mean() / cfg.chunk_size).item(),
        "shuffled_minus_matched_nll_per_token": ((shuffled["joint"] - main_result["joint"]).mean() / cfg.chunk_size).item(),
        "projected_candidate_joint_nll_per_token": (projected["joint"].mean() / cfg.chunk_size).item(),
        "projected_minus_original_nll_per_token": ((projected["joint"] - main_result["joint"]).mean() / cfg.chunk_size).item(),
        "component_weight_mean": weights.mean(0).tolist(),
        "component_weight_entropy_nats": entropy(weights).mean().item(),
        "responsibility_entropy_nats": entropy(responsibilities).mean().item(),
        "responsibility_wins": winners.tolist(),
        "candidate_pair_distance_mean": main_result["dist"].mean().item(),
        "candidate_latent_norm_mean": main_result["norm"].mean().item(),
        "candidate_latent_norm_range": [main_result["norm"].min().item(), main_result["norm"].max().item()],
        "candidate_normalized_norm_mean": main_result["normalized_norm"].mean().item(),
        "target_latent_norm_mean": target.float().norm(dim=-1).mean().item(),
        "target_latent_norm_range": [target.float().norm(dim=-1).min().item(), target.float().norm(dim=-1).max().item()],
        "target_normalized_norm_mean": model.normalize(target).norm(dim=-1).mean().item(),
        "token_marginal_ce_by_position": main_result["position"].mean(0).tolist(),
        "notes": [
            "Teacher-forced fixed dev windows; selected complete target blocks only.",
            "Position CE is a one-token mixture marginal, not joint NLL divided by length.",
            "Context roll is an off-pair diagnostic; it may include other blocks of the same story.",
            "Raw latent Euclidean distances and norms depend on codec scale.",
            "Projection uses the evaluation targets' aggregate mean norm: diagnostic intervention only, not a deployable test metric.",
        ],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
