"""统一评估 CALM 头对照；Brier 和文字辅助评分都不是语义真值。"""

import argparse
import hashlib
from contextlib import nullcontext
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from experiments.evaluate_text_distribution import brier_ngrams
from experiments.experiment_prior_losses import reference_score, repeated_trigrams
from latent_ar.checkpoints import load_model
from latent_ar.data import StoryDataset


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rater", type=Path, default=Path("runs/train-b128/last.pt"))
    p.add_argument("--data", type=Path, default=Path("data/prepared"))
    p.add_argument("--output", type=Path, required=True, help="新评估目录")
    p.add_argument("--device", default="cuda")
    p.add_argument("--windows", type=int, default=128)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--token-temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()
    if (
        min(args.windows, args.flow_steps, args.max_new_tokens) < 1
        or args.draws < 2
        or args.token_temperature < 0
    ):
        p.error("窗口、采样次数与温度参数无效")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("请选择新的空输出目录")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    rater_source = torch.load(args.rater, map_location="cpu", weights_only=True)
    expected_hash = sha256(args.data / "metadata.json")
    fingerprints = {
        f"{s}.{ext}": sha256(args.data / f"{s}.{ext}")
        for s in ("train", "dev")
        for ext in ("bin", "offsets.npy")
    }
    fingerprints["metadata"] = expected_hash
    content_hash = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode()
    ).hexdigest()
    if source.get("dataset_hash") not in {
        expected_hash,
        content_hash,
    } or rater_source.get("dataset_hash") not in {expected_hash, content_hash}:
        raise ValueError("模型、辅助评分模型与数据集不一致")
    import_type = source["architecture"]
    supported = {"latent_calm_v1", "latent_conditional_flow_v1", "latent_calm_token_v1"}
    if import_type not in supported:
        raise ValueError(f"不支持的架构: {import_type}")
    step = source["step"]
    source.pop("optimizer", None)
    rater_source.pop("optimizer", None)
    train = source["config"]["training"]
    model = load_model(source, device).eval()
    rater = load_model(rater_source, device).eval()
    del source, rater_source
    if import_type == "latent_conditional_flow_v1":
        model.cfg.flow_steps = args.flow_steps
    if import_type == "latent_calm_token_v1":
        model.cfg.token_temperature = args.token_temperature
    data = StoryDataset(
        args.data,
        "dev",
        train["min_prompt_tokens"],
        train["max_prompt_tokens"],
        train["max_target_tokens"],
        model.cfg.pad_id,
    )
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    amp = lambda: (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    generator = torch.Generator(device=device).manual_seed(10043)
    # 每个有效块的随机数由两个条件分支共用；不改变目标或历史块位置。
    collected, shuffled_collected, all_labels, prompt_batches = [], [], [], []
    seen = 0
    for batch in data.evaluation_batches(16, None, 10042):
        remaining = args.windows - seen
        if remaining <= 0:
            break
        batch = {k: v[:remaining].to(device) for k, v in batch.items()}
        seen += len(batch["prompt"])
        if not prompt_batches:
            prompt_batches.append({k: v.clone() for k, v in batch.items()})
        with amp():
            h, mean, _, valid = model.teacher_contexts(**batch)
        labels = F.pad(
            batch["labels"],
            (0, mean.shape[1] * model.cfg.chunk_size - batch["labels"].shape[1]),
            value=-100,
        )
        labels = labels.reshape(*valid.shape, model.cfg.chunk_size)
        # 仅同目标位置的有效行循环移位，避免把 padding 当历史。
        for position in range(valid.shape[1]):
            rows = valid[:, position].nonzero(as_tuple=True)[0]
            if len(rows) < 2:
                continue
            contexts = h[rows, position]
            for start in range(0, len(rows), 4):
                subset = contexts[start : start + 4]
                wrong = contexts.roll(1, 0)[start : start + 4]
                expanded = subset[:, None].expand(-1, args.draws, -1)
                wrong_expanded = wrong[:, None].expand_as(expanded)
                if import_type == "latent_conditional_flow_v1":
                    noise = torch.randn(
                        expanded.shape, device=device, generator=generator
                    )
                else:
                    noise = model.draw_noise(expanded.shape[:-1], device, generator)
                with amp():
                    z = model.sample_latent(expanded, noise=noise)
                    z_wrong = model.sample_latent(wrong_expanded, noise=noise)
                    b, n, d = z.shape
                    logits = (
                        model.codec.decode(z.reshape(b * n, 1, d))
                        .float()
                        .reshape(b, n, model.cfg.chunk_size, -1)
                    )
                    other = (
                        model.codec.decode(z_wrong.reshape(b * n, 1, d))
                        .float()
                        .reshape_as(logits)
                    )
                for value in (logits, other):
                    value[..., model.cfg.pad_id] = -torch.inf
                    value[..., model.cfg.bos_id] = -torch.inf
                collected.append(logits.argmax(-1).cpu())
                shuffled_collected.append(other.argmax(-1).cpu())
                all_labels.append(labels[rows[start : start + 4], position].cpu())
    if seen < args.windows or not collected:
        raise ValueError("验证窗口或有效配对不足")
    y = torch.cat(all_labels)
    same = brier_ngrams(torch.cat(collected), y)
    wrong = brier_ngrams(torch.cat(shuffled_collected), y)

    # 自主生成保留真实输出及 EOS；小 AR 评分只作为局部流畅度代理。
    batch = prompt_batches[0]
    free, ce_sum, token_sum = [], 0.0, 0
    for index in range(12):
        row = index % len(batch["prompt"])
        prompt = batch["prompt"][row : row + 1]
        length = batch["prompt_lengths"][row : row + 1]
        with amp():
            output = model.generate(
                prompt,
                length,
                max_new_tokens=args.max_new_tokens,
                generator=torch.Generator(device=device).manual_seed(2043 + index),
            )
        ids = output[0].tolist()
        has_eos = model.cfg.eos_id in ids
        if has_eos:
            ids = ids[: ids.index(model.cfg.eos_id) + 1]
        with amp():
            score, count = reference_score(rater, prompt, length, [ids])
        ce_sum += score
        token_sum += count
        content = ids[:-1] if has_eos else ids
        free.append(
            {
                "prompt": tokenizer.decode(prompt[0, : int(length[0])].tolist()),
                "continuation": tokenizer.decode(content),
                "token_ids": content,
                "eos": has_eos,
                "repeated_trigram_fraction": repeated_trigrams(content),
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "architecture": import_type,
        "windows": seen,
        "scored_blocks": len(y),
        "draws": args.draws,
        "flow_steps": (
            args.flow_steps if import_type == "latent_conditional_flow_v1" else None
        ),
        "token_temperature": (
            args.token_temperature if import_type == "latent_calm_token_v1" else None
        ),
        "conditioned_brier": same,
        "shuffled_context_brier": wrong,
        "reference_lm_ce_per_token": ce_sum / token_sum if token_sum else None,
        "samples": free,
        "notes": [
            "Brier reward 越高越好；所有非对角独立采样配对估计碰撞。",
            "打乱同一块位置的有效行；固定配对噪声，个别相同内容仍可能出现。",
            "小型 token AR 的 CE 是文本流畅度辅助代理，不是语义真值。",
            "block joint 概率未跨架构比较；预测采用真实历史，生成采用自身历史。",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "samples"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
