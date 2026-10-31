"""采集教师自由轨迹、训练反馈学生并评估闭环。"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from download_data import sha256
from latent_ar.checkpoints import (
    CALM_TOKEN_ARCHITECTURE,
    checkpoint_architecture,
    load_model,
)
from latent_ar.checkpoint_io import publish_checkpoint
from latent_ar.data import StoryDataset
from latent_ar.feedback_student import (
    AttentionFeedbackStudent,
    AttentionStudentConfig,
    FeedbackStudent,
    StudentConfig,
    materialize_history_windows,
    prompt_history,
    generate_student_feedback,
    load_feedback_student,
)
from train import precision_context, to_device


def fingerprint(directory):
    names = [
        f"{split}.{suffix}"
        for split in ("train", "dev")
        for suffix in ("bin", "offsets.npy")
    ]
    fingerprints = {name: sha256(directory / name) for name in names}
    fingerprints["metadata"] = sha256(directory / "metadata.json")
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()


def empty_directory(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"必须使用空目录：{path}")
    path.mkdir(parents=True, exist_ok=True)


def load_teacher(data, device):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if sha256(data / "teacher.pt") != manifest["teacher_snapshot_sha256"]:
        raise ValueError("教师快照哈希不匹配")
    saved = torch.load(data / "teacher.pt", map_location="cpu", weights_only=True)
    teacher = load_model(saved, device).eval()
    teacher.cfg.token_temperature = manifest.get("token_temperature", 0.0)
    return teacher, manifest


def collect_split(model, dataset, count, batch_size, device, autocast, seed):
    if count < 1 or count > dataset.size:
        raise ValueError("请求的故事数超出数据集")
    chunks = {key: [] for key in ("h", "r", "story_id", "block_index")}
    prompts, story_ids = [], []
    iterator = dataset.evaluation_batches(
        batch_size, (count + batch_size - 1) // batch_size, seed
    )
    processed = 0
    for batch in iterator:
        take = min(count - processed, len(batch["prompt"]))
        batch = {key: value[:take] for key, value in batch.items()}
        prompts.extend(
            batch["prompt"][i, : int(batch["prompt_lengths"][i])].clone()
            for i in range(take)
        )
        ids = torch.arange(processed, processed + take, device=device)
        story_ids.extend(range(processed, processed + take))
        batch = to_device(batch, device)
        with autocast():
            trace = model.generate(
                batch["prompt"],
                batch["prompt_lengths"],
                max_new_tokens=64,
                generator=torch.Generator(device=device).manual_seed(seed + processed),
                return_trace=True,
            )
        records = trace["records"]
        for index, record in enumerate(records):
            if (index + 1) * model.cfg.chunk_size >= 64:
                continue
            valid = record["active"] & record["lengths"].eq(model.cfg.chunk_size)
            valid &= ~record["token_ids"].eq(model.cfg.eos_id).any(-1)
            chunks["h"].append(record["context"][valid].float().cpu())
            chunks["r"].append(record["feedback"][valid].float().cpu())
            chunks["story_id"].append(ids[valid].cpu())
            chunks["block_index"].append(
                torch.full((int(valid.sum()),), index, dtype=torch.long)
            )
        processed += take
        if processed == count:
            break
    if not chunks["h"] or not sum(len(x) for x in chunks["h"]):
        raise ValueError("自由轨迹没有可用于下一步的完整块")
    return {
        **{key: torch.cat(value) for key, value in chunks.items()},
        "prompts": prompts,
        "prompt_story_ids": torch.tensor(story_ids),
    }


def collect(args):
    torch.set_num_threads(4)
    if min(args.train_prompts, args.dev_prompts, args.batch_size) < 1:
        raise ValueError("采集故事数和批量必须为正")
    device = torch.device(args.device)
    source_hash = sha256(args.teacher)
    saved = torch.load(args.teacher, map_location="cpu", weights_only=True)
    saved.pop("optimizer", None)
    saved.pop("scaler", None)
    if sha256(args.teacher) != source_hash:
        raise ValueError("读取期间教师检查点发生变化")
    if checkpoint_architecture(saved) != CALM_TOKEN_ARCHITECTURE:
        raise ValueError("教师必须是 CALM token 检查点")
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    digest = fingerprint(args.data)
    if (
        saved["dataset_hash"] != digest
        or sha256(args.data / "tokenizer.json") != metadata["tokenizer_sha256"]
    ):
        raise ValueError("教师与数据或词表不匹配")
    source_tokenizer = args.teacher.parent / "tokenizer.json"
    if sha256(source_tokenizer) != metadata["tokenizer_sha256"]:
        raise ValueError("教师检查点旁词表不匹配")
    empty_directory(args.output)
    snapshot = {
        key: saved[key]
        for key in (
            "architecture",
            "model_config",
            "model",
            "config",
            "dataset_hash",
            "step",
        )
    }
    torch.save(snapshot, args.output / "teacher.pt")
    if snapshot["step"] != saved["step"]:
        raise ValueError("教师快照步数不一致")
    shutil.copyfile(source_tokenizer, args.output / "tokenizer.json")
    teacher = load_model(snapshot, device).eval()
    teacher.cfg.token_temperature = 0.0
    cfg, tr = teacher.cfg, saved["config"]["training"]
    autocast = precision_context(device, tr["precision"])
    counts = {}
    for split, count in (("train", args.train_prompts), ("dev", args.dev_prompts)):
        dataset = StoryDataset(
            args.data,
            split,
            tr["min_prompt_tokens"],
            tr["max_prompt_tokens"],
            tr["max_target_tokens"],
            cfg.pad_id,
        )
        values = collect_split(
            teacher, dataset, count, args.batch_size, device, autocast, args.seed
        )
        torch.save(values, args.output / f"{split}.pt")
        counts[split] = {
            "prompts": count,
            "blocks": len(values["h"]),
            "story_ids": values["prompt_story_ids"].tolist(),
        }
    manifest = {
        "teacher_source": str(args.teacher.resolve()),
        "teacher_source_sha256": source_hash,
        "teacher_snapshot_sha256": sha256(args.output / "teacher.pt"),
        "teacher_step": saved["step"],
        "dataset_hash": digest,
        "tokenizer_sha256": sha256(args.output / "tokenizer.json"),
        "split_sha256": {s: sha256(args.output / f"{s}.pt") for s in counts},
        "counts": counts,
        "seed": args.seed,
        "rollout_tokens": 64,
        "token_temperature": 0,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, ensure_ascii=False))


def verified_data(data):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "dev"):
        if sha256(data / f"{split}.pt") != manifest["split_sha256"][split]:
            raise ValueError(f"{split} 轨迹文件哈希不匹配")
    if sha256(data / "teacher.pt") != manifest["teacher_snapshot_sha256"]:
        raise ValueError("教师快照哈希不匹配")
    if sha256(data / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("词表哈希不匹配")
    return manifest


@torch.inference_mode()
def extract_prompt_history(
    teacher, prompts, history_blocks, batch_size, device, autocast
):
    """批量提取提示词主干状态，不运行生成，也不更改已有轨迹。"""
    from torch.nn.utils.rnn import pad_sequence

    values, masks = [], []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        lengths = torch.tensor([len(prompt) for prompt in batch], device=device)
        padded = pad_sequence(
            batch, batch_first=True, padding_value=teacher.cfg.pad_id
        ).to(device)
        with autocast():
            encoded, valid = teacher.encode_history(padded, lengths)
            positions = torch.arange(encoded.shape[1], device=device).expand(
                len(batch), -1
            )
            hidden, _ = teacher.backbone(encoded, positions, valid)
            prefix, prefix_valid = prompt_history(hidden, valid.sum(-1), history_blocks)
        values.append(prefix.float().cpu())
        masks.append(prefix_valid.cpu())
    return torch.cat(values), torch.cat(masks)


def add_prompt_history(args):
    if args.history_blocks < 1 or args.batch_size < 1:
        raise ValueError("提示词历史块数和批量必须为正")
    manifest = verified_data(args.data)
    if args.output.resolve() == args.data.resolve():
        raise ValueError("输出目录必须不同于源数据目录")
    device = torch.device(args.device)
    teacher, _ = load_teacher(args.data, device)
    snapshot = torch.load(
        args.data / "teacher.pt", map_location="cpu", weights_only=True
    )
    autocast = precision_context(device, snapshot["config"]["training"]["precision"])
    empty_directory(args.output)
    for name in ("teacher.pt", "tokenizer.json"):
        shutil.copyfile(args.data / name, args.output / name)
    split_hashes = {}
    for split in ("train", "dev"):
        data = torch.load(
            args.data / f"{split}.pt", map_location="cpu", weights_only=True
        )
        if "prefix_h" in data or "prefix_valid" in data:
            raise ValueError("源数据已有提示词历史")
        if len(data["prompts"]) != len(data["prompt_story_ids"]):
            raise ValueError("提示词和故事编号长度不匹配")
        data["prefix_h"], data["prefix_valid"] = extract_prompt_history(
            teacher,
            data["prompts"],
            args.history_blocks,
            args.batch_size,
            device,
            autocast,
        )
        # 检查每条轨迹都能匹配提示词，且生成块编号连续。
        materialize_history_windows(data, args.history_blocks, use_prompt_history=True)
        torch.save(data, args.output / f"{split}.pt")
        split_hashes[split] = sha256(args.output / f"{split}.pt")
    manifest["prompt_history"] = {
        "history_blocks": args.history_blocks,
        "source_data": str(args.data.resolve()),
        "source_manifest_sha256": sha256(args.data / "manifest.json"),
        "source_split_sha256": manifest["split_sha256"],
        "precision": snapshot["config"]["training"]["precision"],
    }
    manifest["split_sha256"] = split_hashes
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["prompt_history"], ensure_ascii=False))


@torch.no_grad()
def student_metrics(student, data, mean, device, batch_size):
    sums = torch.zeros(3, dtype=torch.float64)
    windows = (
        materialize_history_windows(
            data, student.cfg.history_blocks, student.cfg.use_prompt_history
        )
        if isinstance(student, AttentionFeedbackStudent)
        else None
    )
    for start in range(0, len(data["h"]), batch_size):
        h = data["h"][start : start + batch_size].to(device)
        target = data["r"][start : start + batch_size].to(device)
        if windows is None:
            pred = student(h)
        else:
            window, valid = windows
            pred = student(
                window[start : start + batch_size].to(device),
                valid=valid[start : start + batch_size].to(device),
            )
        sums[0] += (pred - target).square().mean(-1).sum().double().cpu()
        sums[1] += F.cosine_similarity(pred, target, dim=-1).sum().double().cpu()
        sums[2] += (mean.to(device) - target).square().mean(-1).sum().double().cpu()
    return {
        "mse": float(sums[0] / len(data["h"])),
        "cosine": float(sums[1] / len(data["h"])),
        "train_mean_baseline_mse": float(sums[2] / len(data["h"])),
    }


def train(args):
    torch.set_num_threads(4)
    if min(args.steps, args.batch_size) < 1:
        raise ValueError("步数和批量必须为正")
    manifest = verified_data(args.data)
    if args.resume:
        if not args.run_dir.is_dir():
            raise ValueError("续训目录不存在")
    else:
        empty_directory(args.run_dir)
    data = torch.load(args.data / "train.pt", weights_only=True)
    dev = torch.load(args.data / "dev.pt", weights_only=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cfg = StudentConfig(
        context_dim=data["h"].shape[-1],
        feedback_dim=data["r"].shape[-1],
        hidden_dim=512,
        layers=2,
        noise_dim=0,
    )
    student_type = getattr(args, "student_type", "mlp")
    if student_type == "attention":
        cfg = AttentionStudentConfig(
            context_dim=data["h"].shape[-1],
            feedback_dim=data["r"].shape[-1],
            history_blocks=getattr(args, "history_blocks", 2),
            use_prompt_history="prompt_history" in manifest,
        )
    elif student_type != "mlp":
        raise ValueError("未知学生类型")
    windows = (
        materialize_history_windows(data, cfg.history_blocks, cfg.use_prompt_history)
        if student_type == "attention"
        else None
    )
    saved = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume
        else None
    )
    if saved:
        if (
            saved["teacher_snapshot_sha256"] != manifest["teacher_snapshot_sha256"]
            or saved["split_sha256"] != manifest["split_sha256"]
            or saved["dataset_hash"] != manifest["dataset_hash"]
        ):
            raise ValueError("续训教师或轨迹数据不匹配")
        saved_config = dict(saved["config"])
        if saved.get("architecture") == AttentionFeedbackStudent.architecture:
            saved_config.setdefault("use_prompt_history", False)
        if saved["device_type"] != device.type or saved_config != asdict(cfg):
            raise ValueError("续训设备类型或学生结构不匹配")
        if saved["batch_size"] != args.batch_size:
            raise ValueError("续训批量必须与原训练相同")
        student = load_feedback_student(saved).to(device)
    else:
        student = (
            AttentionFeedbackStudent(cfg)
            if student_type == "attention"
            else FeedbackStudent(cfg)
        ).to(device)
    optim = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    mean = data["r"].mean(0)
    best = float("inf")
    first = 1
    if saved:
        optim.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first, best = saved["step"] + 1, saved["best_mse"]
    if first > args.steps:
        raise ValueError("最终总步数必须大于检查点步数")
    if not args.resume:
        run_config = {
            "data": str(args.data.resolve()),
            "batch_size": args.batch_size,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "student": asdict(cfg),
            "teacher_snapshot_sha256": manifest["teacher_snapshot_sha256"],
            "split_sha256": manifest["split_sha256"],
            "dataset_hash": manifest["dataset_hash"],
        }
        (args.run_dir / "config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for step in range(first, args.steps + 1):
        indices = torch.randint(len(data["h"]), (args.batch_size,), generator=generator)
        h, r = data["h"][indices].to(device), data["r"][indices].to(device)
        optim.zero_grad(set_to_none=True)
        if windows is None:
            prediction = student(h)
        else:
            window, window_valid = windows
            prediction = student(
                window[indices].to(device),
                valid=window_valid[indices].to(device),
            )
        training_loss = F.mse_loss(prediction, r)
        training_loss.backward()
        optim.step()
        if step % 100 == 0 or step == args.steps:
            values = student_metrics(student, dev, mean, device, args.batch_size)
            improved = values["mse"] < best
            best = min(best, values["mse"])
            checkpoint = student.checkpoint() | {
                "step": step,
                "metrics": values,
                "teacher_snapshot_sha256": manifest["teacher_snapshot_sha256"],
                "split_sha256": manifest["split_sha256"],
                "dataset_hash": manifest["dataset_hash"],
                "optimizer": optim.state_dict(),
                "sampler_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "device_type": device.type,
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
                "batch_size": args.batch_size,
                "best_mse": best,
            }
            for name in ["last", "best"] if improved else ["last"]:
                temporary = args.run_dir / f"{name}.tmp"
                torch.save(checkpoint, temporary)
                publish_checkpoint(temporary, args.run_dir / f"{name}.pt", step)
            record = {
                "step": step,
                "train_mse": float(training_loss.detach()),
                **values,
            }
            with (args.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record), flush=True)


def truncate(tokens, eos):
    tokens = tokens.tolist()
    return tokens[: tokens.index(eos) + 1] if eos in tokens else tokens


def evaluate(args):
    torch.set_num_threads(4)
    manifest = verified_data(args.data)
    device = torch.device(args.device)
    teacher, _ = load_teacher(args.data, device)
    saved = torch.load(args.student, map_location="cpu", weights_only=True)
    if (
        saved["teacher_snapshot_sha256"] != manifest["teacher_snapshot_sha256"]
        or saved["split_sha256"] != manifest["split_sha256"]
    ):
        raise ValueError("学生不属于该教师及轨迹数据")
    student = load_feedback_student(saved).to(device).eval()
    dev = torch.load(args.data / "dev.pt", weights_only=True)
    if not 1 <= args.prompts <= len(dev["prompts"]) or args.max_new_tokens < 1:
        raise ValueError("无效的评估故事数或生成长度")
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    autocast = precision_context(device, "bf16")
    summary = {
        "common_prefix_tokens": 0,
        "matched_tokens": 0,
        "compared_tokens": 0,
        "max_length_tokens": 0,
        "teacher_tokens": 0,
        "student_tokens": 0,
        "teacher_eos": 0,
        "student_eos": 0,
        "first_block_matches": 0,
        "first_block_comparable": 0,
        "feedback_mse_sum": 0.0,
        "feedback_cos_sum": 0.0,
        "feedback_blocks": 0,
    }
    samples = []
    for i, prompt in enumerate(dev["prompts"][: args.prompts]):
        ids = prompt.to(device)[None]
        lengths = torch.tensor([len(prompt)], device=device)
        with autocast():
            teacher_out = teacher.generate(
                ids,
                lengths,
                max_new_tokens=args.max_new_tokens,
                generator=torch.Generator(device=device).manual_seed(args.seed + i),
            )
            result = generate_student_feedback(
                teacher,
                student,
                ids,
                lengths,
                max_new_tokens=args.max_new_tokens,
                generator=torch.Generator(device=device).manual_seed(args.seed + i),
                return_trace=True,
            )
        a = truncate(teacher_out[0].cpu(), teacher.cfg.eos_id)
        b = truncate(result["output_ids"][0].cpu(), teacher.cfg.eos_id)
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        summary["common_prefix_tokens"] += common
        summary["matched_tokens"] += sum(x == y for x, y in zip(a, b))
        summary["compared_tokens"] += min(len(a), len(b))
        summary["max_length_tokens"] += max(len(a), len(b))
        summary["teacher_tokens"] += len(a)
        summary["student_tokens"] += len(b)
        summary["teacher_eos"] += int(teacher.cfg.eos_id in a)
        summary["student_eos"] += int(teacher.cfg.eos_id in b)
        width = min(teacher.cfg.chunk_size, args.max_new_tokens)
        summary["first_block_comparable"] += 1
        summary["first_block_matches"] += int(a[:width] == b[:width])
        for index, rec in enumerate(result["records"]):
            if (index + 1) * teacher.cfg.chunk_size >= args.max_new_tokens:
                continue
            if (
                not bool(rec["active"][0])
                or int(rec["lengths"][0]) != teacher.cfg.chunk_size
            ):
                continue
            if bool(rec["token_ids"][0].eq(teacher.cfg.eos_id).any()):
                continue
            true, pred = (
                rec["true_feedback"][0].float(),
                rec["predicted_feedback"][0].float(),
            )
            summary["feedback_mse_sum"] += float((true - pred).square().mean())
            summary["feedback_cos_sum"] += float(
                F.cosine_similarity(true[None], pred[None])
            )
            summary["feedback_blocks"] += 1
        if i < 16:
            samples.append(
                {
                    "story_id": int(dev["prompt_story_ids"][i]),
                    "prompt": tokenizer.decode(prompt.tolist()),
                    "teacher": tokenizer.decode(a),
                    "student": tokenizer.decode(b),
                }
            )
    blocks = summary["feedback_blocks"]
    report = {
        "student_architecture": saved["architecture"],
        "student_config": saved["config"],
        "student_step": saved["step"],
        "prompts": args.prompts,
        "max_new_tokens": args.max_new_tokens,
        "teacher_snapshot_sha256": manifest["teacher_snapshot_sha256"],
        "student_sha256": sha256(args.student),
        **summary,
        "mean_common_prefix_tokens": summary["common_prefix_tokens"] / args.prompts,
        "token_match_rate": summary["matched_tokens"]
        / max(1, summary["compared_tokens"]),
        "token_match_rate_max_length": summary["matched_tokens"]
        / max(1, summary["max_length_tokens"]),
        "first_block_match_rate": summary["first_block_matches"] / args.prompts,
        "feedback_mse": summary["feedback_mse_sum"] / blocks if blocks else None,
        "feedback_cosine": summary["feedback_cos_sum"] / blocks if blocks else None,
        "samples": samples,
        "note": "学生历史下的真实与预测反馈比较；闭环质量诊断，不测异步速度。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("评估输出已存在")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "samples"},
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--teacher", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--data", type=Path, default=Path("data/prepared"))
    c.add_argument("--train-prompts", type=int, default=4096)
    c.add_argument("--dev-prompts", type=int, default=512)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--device", default="cuda")
    c.add_argument("--seed", type=int, default=2031)
    a = sub.add_parser("add-prompt-history")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--history-blocks", type=int, default=2)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--run-dir", type=Path, required=True)
    t.add_argument("--resume", type=Path)
    t.add_argument("--student-type", choices=("mlp", "attention"), default="mlp")
    t.add_argument("--history-blocks", type=int, default=2)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=2032)
    e = sub.add_parser("evaluate")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--student", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--prompts", type=int, default=128)
    e.add_argument("--max-new-tokens", type=int, default=64)
    e.add_argument("--device", default="cuda")
    e.add_argument("--seed", type=int, default=2033)
    args = parser.parse_args()
    {
        "collect": collect,
        "add-prompt-history": add_prompt_history,
        "train": train,
        "evaluate": evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
