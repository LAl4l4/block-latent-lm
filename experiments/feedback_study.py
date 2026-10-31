"""Independent feedback-distillation study; source artifacts are never overwritten."""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from download_data import sha256
from experiments.distill_gru_feedback import (
    empty_directory,
    fingerprint,
    load_teacher,
    verified_data,
)
from latent_ar.checkpoint_io import publish_checkpoint
from latent_ar.data import StoryDataset
from latent_ar.feedback_student import FeedbackStudent, StudentConfig
from train import precision_context


def collect_split(
    teacher,
    dataset,
    start,
    count,
    batch_size,
    device,
    precision,
    seed,
    student=None,
    rollout_seed=None,
):
    if count < 1 or start + count > dataset.size:
        raise ValueError("Invalid story range")
    rollout_seed = seed if rollout_seed is None else rollout_seed
    result = {
        k: []
        for k in (
            "h",
            "r",
            "u",
            "story_id",
            "block_index",
            "prompt_row",
            "past_feedback",
            "past_lengths",
            "next_tokens",
        )
    }
    prompts = []
    for offset in range(0, count, batch_size):
        ids = list(
            range(start + offset, min(start + offset + batch_size, start + count))
        )
        p = [
            torch.as_tensor(dataset.example(i, random.Random(seed + i))[0]) for i in ids
        ]
        prompts.extend(p)
        lengths = torch.tensor([len(x) for x in p], device=device)
        padded = pad_sequence(p, batch_first=True, padding_value=teacher.cfg.pad_id).to(
            device
        )
        with precision():
            if student is None:
                trace = teacher.generate(
                    padded,
                    lengths,
                    max_new_tokens=64,
                    generator=torch.Generator(device=device).manual_seed(
                        rollout_seed + offset
                    ),
                    return_trace=True,
                    record_sampling_noise=True,
                )
            else:
                from latent_ar.feedback_student import generate_student_feedback

                trace = generate_student_feedback(
                    teacher,
                    student,
                    padded,
                    lengths,
                    max_new_tokens=64,
                    generator=torch.Generator(device=device).manual_seed(
                        rollout_seed + offset
                    ),
                    return_trace=True,
                )
        records = trace["records"]
        for b, rec in enumerate(records[:-1]):
            valid = (
                rec["active"]
                & rec["lengths"].eq(teacher.cfg.chunk_size)
                & ~rec["token_ids"].eq(teacher.cfg.eos_id).any(-1)
            )
            rows = valid.nonzero(as_tuple=True)[0]
            if not len(rows):
                continue
            nxt = records[b + 1]
            tokens = torch.full(
                (len(rows), teacher.cfg.chunk_size),
                -100,
                dtype=torch.long,
                device=device,
            )
            width = nxt["token_ids"].shape[1]
            tokens[:, :width] = nxt["token_ids"][rows]
            mask = (
                torch.arange(teacher.cfg.chunk_size, device=device)[None]
                >= nxt["lengths"][rows, None]
            )
            tokens.masked_fill_(mask, -100)
            feedback_key = "feedback" if student is None else "predicted_feedback"
            target_key = "feedback" if student is None else "true_feedback"
            if student is not None:
                tokens.fill_(
                    -100
                )  # Student-trajectory next tokens are not teacher counterfactuals.
            past = rec[target_key].new_zeros((len(rows), 3, teacher.cfg.dim))
            for previous in range(b):
                past[:, previous] = records[previous][feedback_key][rows]
            values = {
                "h": rec["context"][rows],
                "r": rec[target_key][rows],
                "u": rec["noise"][rows],
                "story_id": torch.tensor(ids, device=device)[rows],
                "block_index": torch.full((len(rows),), b, device=device),
                "prompt_row": rows + offset,
                "past_feedback": past,
                "past_lengths": torch.full((len(rows),), b, device=device),
                "next_tokens": tokens,
            }
            for key, value in values.items():
                result[key].append(
                    value.float().cpu()
                    if key in ("h", "r", "u", "past_feedback")
                    else value.cpu()
                )
    if not result["h"]:
        raise ValueError("No continuing blocks")
    return {
        **{k: torch.cat(v) for k, v in result.items()},
        "prompts": prompts,
        "prompt_story_ids": torch.arange(start, start + count),
    }


def collect(args):
    if (
        min(args.train_prompts, args.dev_prompts, args.test_prompts, args.batch_size)
        < 1
        or args.temperature < 0
    ):
        raise ValueError("Invalid collection parameters")
    source = verified_data(args.source_data)
    saved = torch.load(
        args.source_data / "teacher.pt", map_location="cpu", weights_only=True
    )
    if fingerprint(args.prepared_data) != saved["dataset_hash"]:
        raise ValueError("Prepared dataset fingerprint differs from teacher")
    if sha256(args.prepared_data / "tokenizer.json") != source["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer differs from teacher")
    metadata = json.loads(
        (args.prepared_data / "metadata.json").read_text(encoding="utf-8")
    )
    for name, key in (
        ("test.bin", "tokens_sha256"),
        ("test.offsets.npy", "offsets_sha256"),
    ):
        if sha256(args.prepared_data / name) != metadata["splits"]["test"][key]:
            raise ValueError("Prepared test split hash mismatch")
    teacher, _ = load_teacher(args.source_data, args.device)
    teacher.cfg.token_temperature = args.temperature
    tr = saved["config"]["training"]
    precision = precision_context(torch.device(args.device), tr["precision"])
    empty_directory(args.output)
    for name in ("teacher.pt", "tokenizer.json"):
        shutil.copyfile(args.source_data / name, args.output / name)
    hashes, counts = {}, {}
    rollout_seeds = {
        "train": args.seed,
        "dev": args.seed + 1000000,
        "test": args.seed + 2000000,
    }
    for split, dataset_split, start, count in [
        ("train", "train", 0, args.train_prompts),
        ("dev", "dev", 0, args.dev_prompts),
        ("test", "test", 0, args.test_prompts),
    ]:
        ds = StoryDataset(
            args.prepared_data,
            dataset_split,
            tr["min_prompt_tokens"],
            tr["max_prompt_tokens"],
            tr["max_target_tokens"],
            teacher.cfg.pad_id,
        )
        values = collect_split(
            teacher,
            ds,
            start,
            count,
            args.batch_size,
            torch.device(args.device),
            precision,
            args.seed,
            rollout_seed=rollout_seeds[split],
        )
        torch.save(values, args.output / (split + ".pt"))
        hashes[split] = sha256(args.output / (split + ".pt"))
        counts[split] = {
            "prompts": count,
            "blocks": len(values["h"]),
            "first_story_id": start,
            "source_split": dataset_split,
        }
        if split == "train" and args.temperature == 0:
            old = torch.load(args.source_data / "train.pt", weights_only=True)
            lookup = {
                (int(story), int(block)): i
                for i, (story, block) in enumerate(
                    zip(old["story_id"], old["block_index"])
                )
            }
            pairs = [
                (i, lookup[(int(story), int(block))])
                for i, (story, block) in enumerate(
                    zip(values["story_id"], values["block_index"])
                )
                if (int(story), int(block)) in lookup
            ]
            alignment = {
                "old_blocks": len(old["h"]),
                "matched_keys": len(pairs),
                "prompt_matches": sum(
                    torch.equal(a, b) for a, b in zip(values["prompts"], old["prompts"])
                ),
                "compared_prompts": min(len(values["prompts"]), len(old["prompts"])),
            }
            for field in ("h", "r"):
                alignment[field + "_max_abs_difference"] = max(
                    (
                        float((values[field][i] - old[field][j]).abs().max())
                        for i, j in pairs
                    ),
                    default=None,
                )
    manifest = {
        "teacher_snapshot_sha256": source["teacher_snapshot_sha256"],
        "tokenizer_sha256": source["tokenizer_sha256"],
        "dataset_hash": saved["dataset_hash"],
        "source_data": str(args.source_data.resolve()),
        "source_manifest_sha256": sha256(args.source_data / "manifest.json"),
        "split_sha256": hashes,
        "counts": counts,
        "seed": args.seed,
        "rollout_seeds": rollout_seeds,
        "batch_size": args.batch_size,
        "rollout_tokens": 64,
        "token_temperature": args.temperature,
        "token_top_k": teacher.cfg.token_top_k,
        "test_note": "Independent prepared test split held out from teacher and student selection.",
        "prepared_test_sha256": {
            name: sha256(args.prepared_data / name)
            for name in ("test.bin", "test.offsets.npy")
        },
    }
    if args.temperature == 0:
        manifest["old_greedy_alignment"] = alignment
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


class PromptDataset:
    def __init__(self, prompts):
        self.prompts, self.size = prompts, len(prompts)

    def example(self, index, rng):
        return self.prompts[index], None


def collect_student(args):
    source = verify(args.data)
    if source["token_temperature"] != 0:
        raise ValueError("On-policy collection currently supports greedy teachers only")
    data = torch.load(args.data / "train.pt", weights_only=True)
    if (
        args.start_story < 0
        or args.train_prompts < 1
        or args.batch_size < 1
        or args.start_story + args.train_prompts > len(data["prompts"])
    ):
        raise ValueError("Invalid student story range")
    expected_ids = torch.arange(args.start_story, args.start_story + args.train_prompts)
    if not torch.equal(
        data["prompt_story_ids"][
            args.start_story : args.start_story + args.train_prompts
        ],
        expected_ids,
    ):
        raise ValueError("Source prompt rows do not match requested story IDs")
    teacher, _ = load_teacher(args.data, args.device)
    saved = torch.load(args.student, map_location="cpu", weights_only=True)
    if saved["study_settings"]["teacher_hash"] != source["teacher_snapshot_sha256"]:
        raise ValueError("Student teacher differs")
    student = FeedbackStudent.from_checkpoint(saved).to(args.device).eval()
    if student.cfg.noise_dim:
        raise ValueError("On-policy collection requires a greedy student")
    snapshot = torch.load(
        args.data / "teacher.pt", map_location="cpu", weights_only=True
    )
    precision = precision_context(
        torch.device(args.device), snapshot["config"]["training"]["precision"]
    )
    values = collect_split(
        teacher,
        PromptDataset(data["prompts"]),
        args.start_story,
        args.train_prompts,
        args.batch_size,
        torch.device(args.device),
        precision,
        source["seed"],
        student,
    )
    empty_directory(args.output)
    for name in ("teacher.pt", "tokenizer.json", "dev.pt", "test.pt"):
        shutil.copyfile(args.data / name, args.output / name)
    torch.save(values, args.output / "train.pt")
    manifest = dict(source)
    manifest["split_sha256"] = dict(source["split_sha256"])
    manifest["split_sha256"]["train"] = sha256(args.output / "train.pt")
    manifest["on_policy"] = {
        "source_data": str(args.data.resolve()),
        "source_train_sha256": source["split_sha256"]["train"],
        "student_sha256": sha256(args.student),
        "prompts": args.train_prompts,
        "start_story": args.start_story,
        "next_tokens": "unavailable; MSE only",
    }
    manifest["counts"] = dict(source["counts"])
    manifest["counts"]["train"] = {
        "prompts": args.train_prompts,
        "blocks": len(values["h"]),
        "first_story_id": args.start_story,
        "source_split": "train",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


ROW_FIELDS = (
    "h",
    "r",
    "u",
    "story_id",
    "block_index",
    "prompt_row",
    "past_feedback",
    "past_lengths",
    "next_tokens",
)


def matched_augment(args):
    base_manifest = verify(args.base_data)
    student_manifest = verify(args.student_data)
    provenance = student_manifest.get("on_policy")
    if (
        not provenance
        or provenance.get("source_train_sha256")
        != base_manifest["split_sha256"]["train"]
    ):
        raise ValueError("Student data must come from the supplied base training data")
    for key in (
        "teacher_snapshot_sha256",
        "tokenizer_sha256",
        "dataset_hash",
        "token_temperature",
    ):
        if base_manifest[key] != student_manifest[key]:
            raise ValueError("Source teacher or dataset mismatch")
    if base_manifest["token_temperature"] != 0:
        raise ValueError("Matched augmentation requires greedy data")
    for split in ("dev", "test"):
        if (
            base_manifest["split_sha256"][split]
            != student_manifest["split_sha256"][split]
        ):
            raise ValueError("Validation splits differ")
    base = torch.load(args.base_data / "train.pt", weights_only=True)
    student = torch.load(args.student_data / "train.pt", weights_only=True)
    n = args.base_prompts
    ids = student["prompt_story_ids"]
    if n < 1 or n > len(base["prompts"]) or len(ids) == 0:
        raise ValueError("Invalid baseline or student story count")
    if not torch.equal(base["prompt_story_ids"][:n], torch.arange(n)):
        raise ValueError("Baseline story IDs must start at zero")
    if not torch.equal(ids, torch.arange(n, n + len(ids))):
        raise ValueError(
            "Student stories must follow the baseline without overlap or gaps"
        )
    if n + len(ids) > len(base["prompts"]):
        raise ValueError("Student stories exceed base source")
    if any(
        not torch.equal(a, b)
        for a, b in zip(base["prompts"][n : n + len(ids)], student["prompts"])
    ):
        raise ValueError("Student prompts differ from base source")
    for data in (base, student):
        rows = data["prompt_row"]
        if rows.min() < 0 or rows.max() >= len(data["prompts"]):
            raise ValueError("Invalid prompt mapping")
        if not torch.equal(data["prompt_story_ids"][rows], data["story_id"]):
            raise ValueError("Story IDs do not match prompt rows")
    baseline_idx = (base["prompt_row"] < n).nonzero(as_tuple=True)[0]
    off_idx = ((base["prompt_row"] >= n) & (base["prompt_row"] < n + len(ids))).nonzero(
        as_tuple=True
    )[0]
    on_idx = torch.arange(len(student["h"]))
    keep = min(len(off_idx), len(on_idx))
    if not len(baseline_idx) or not keep:
        raise ValueError("No baseline or augmentation blocks")
    rng = torch.Generator().manual_seed(args.seed)
    off_idx = off_idx[torch.randperm(len(off_idx), generator=rng)[:keep]].sort().values
    on_idx = on_idx[torch.randperm(len(on_idx), generator=rng)[:keep]].sort().values
    prompts = base["prompts"][:n] + student["prompts"]
    prompt_ids = torch.cat((base["prompt_story_ids"][:n], ids))
    results = []
    for extra, selected, is_on in ((base, off_idx, False), (student, on_idx, True)):
        result = {
            field: torch.cat((base[field][baseline_idx], extra[field][selected]))
            for field in ROW_FIELDS
        }
        if is_on:
            result["prompt_row"][len(baseline_idx) :] += n
            result["next_tokens"].fill_(-100)
        result["prompts"], result["prompt_story_ids"] = prompts, prompt_ids
        if not torch.equal(
            result["prompt_story_ids"][result["prompt_row"]], result["story_id"]
        ):
            raise ValueError("Merged prompt mapping is invalid")
        results.append(result)
    destinations = (args.output_offpolicy, args.output_onpolicy)
    resolved = [p.resolve() for p in destinations]
    if (
        len(set(resolved + [args.base_data.resolve(), args.student_data.resolve()]))
        != 4
    ):
        raise ValueError("Output directories must differ from each other and sources")
    for path in destinations:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Output directory is not empty: {path}")
    original_counts = {
        "offpolicy": int(
            ((base["prompt_row"] >= n) & (base["prompt_row"] < n + len(ids))).sum()
        ),
        "onpolicy": len(student["h"]),
    }
    original_story_counts = {
        "offpolicy": torch.bincount(
            base["story_id"][
                (base["prompt_row"] >= n) & (base["prompt_row"] < n + len(ids))
            ]
            - n,
            minlength=len(ids),
        ).tolist(),
        "onpolicy": torch.bincount(
            student["story_id"] - n, minlength=len(ids)
        ).tolist(),
    }
    for path, result, arm in zip(destinations, results, ("offpolicy", "onpolicy")):
        retained_story_counts = torch.bincount(
            result["story_id"][len(baseline_idx) :] - n, minlength=len(ids)
        ).tolist()
        empty_directory(path)
        for name in ("teacher.pt", "tokenizer.json", "dev.pt", "test.pt"):
            shutil.copyfile(args.base_data / name, path / name)
        torch.save(result, path / "train.pt")
        manifest = dict(base_manifest)
        manifest["split_sha256"] = dict(base_manifest["split_sha256"])
        manifest["split_sha256"]["train"] = sha256(path / "train.pt")
        manifest["matched_augmentation"] = {
            "arm": arm,
            "seed": args.seed,
            "base_data": str(args.base_data.resolve()),
            "base_train_sha256": base_manifest["split_sha256"]["train"],
            "student_data": str(args.student_data.resolve()),
            "student_train_sha256": student_manifest["split_sha256"]["train"],
            "student_sha256": provenance["student_sha256"],
            "baseline_prompts": n,
            "baseline_blocks": len(baseline_idx),
            "additional_story_ids": ids.tolist(),
            "original_additional_blocks": original_counts,
            "retained_additional_blocks": keep,
            "original_blocks_by_story": original_story_counts[arm],
            "retained_blocks_by_story": retained_story_counts,
            "next_tokens_available": arm == "offpolicy",
            "mse_only": arm == "onpolicy",
        }
        if arm == "onpolicy":
            manifest["on_policy"] = provenance
        manifest["counts"] = dict(base_manifest["counts"])
        manifest["counts"]["train"] = {
            "prompts": len(prompts),
            "blocks": len(result["h"]),
            "first_story_id": 0,
            "source_split": "train",
        }
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def noise_for(mode, u):
    if mode == "none":
        return None
    return u if mode == "shared" else torch.full_like(u, 0.5)


def verify(data):
    manifest = verified_data(data)
    if sha256(data / "test.pt") != manifest["split_sha256"]["test"]:
        raise ValueError("Test hash mismatch")
    return manifest


def teacher_precision(data, device):
    snapshot = torch.load(data / "teacher.pt", map_location="cpu", weights_only=True)
    configured = snapshot["config"]["training"]["precision"]
    actual = configured if device.type == "cuda" else "fp32"
    return configured, actual, precision_context(device, configured)


def train(args):
    if min(args.steps, args.batch_size, args.hidden_dim, args.layers) < 1:
        raise ValueError("Invalid training parameters")
    manifest = verify(args.data)
    data = torch.load(args.data / "train.pt", weights_only=True)
    dev = torch.load(args.data / "dev.pt", weights_only=True)
    if args.train_prompts < 1 or args.train_prompts > len(data["prompts"]):
        raise ValueError("Invalid training prompt count")
    mask = data["prompt_row"] < args.train_prompts
    if not mask.any():
        raise ValueError("No training blocks")
    row_keys = (
        "h",
        "r",
        "u",
        "story_id",
        "block_index",
        "prompt_row",
        "past_feedback",
        "past_lengths",
        "next_tokens",
    )
    data = {k: v[mask] if k in row_keys else v for k, v in data.items()}
    if (manifest["token_temperature"] == 0) != (args.noise_mode == "none"):
        raise ValueError("Greedy requires none; stochastic requires shared or blind")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cfg = StudentConfig(
        context_dim=data["h"].shape[-1],
        feedback_dim=data["r"].shape[-1],
        noise_dim=0 if args.noise_mode == "none" else data["u"].shape[-1],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    settings = {
        "student": vars(cfg),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_prompts": args.train_prompts,
        "noise_mode": args.noise_mode,
        "objective": args.objective,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mse_weight": args.mse_weight,
        "device_type": device.type,
        "data_hashes": manifest["split_sha256"],
        "teacher_hash": manifest["teacher_snapshot_sha256"],
        "init_sha256": sha256(args.init_student) if args.init_student else None,
    }
    if args.objective == "behavior":
        configured_precision, actual_precision, teacher_autocast = teacher_precision(
            args.data, device
        )
        settings["teacher_precision"] = actual_precision
        settings["teacher_configured_precision"] = configured_precision
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["study_settings"] != settings:
            raise ValueError("Resume settings or source mismatch")
        student = FeedbackStudent.from_checkpoint(saved).to(device)
    else:
        empty_directory(args.run_dir)
        student = FeedbackStudent(cfg).to(device)
        if args.init_student:
            initial = torch.load(
                args.init_student, map_location="cpu", weights_only=True
            )
            initial_teacher = initial.get("study_settings", {}).get(
                "teacher_hash", initial.get("teacher_snapshot_sha256")
            )
            if initial_teacher != manifest["teacher_snapshot_sha256"]:
                raise ValueError("Initial student teacher mismatch")
            if initial["config"] != vars(cfg):
                raise ValueError("Initial student architecture mismatch")
            student.load_state_dict(initial["model"])
        saved = None
        (args.run_dir / "config.json").write_text(
            json.dumps(settings | {"checkpoint_selection": "dev_mse"}, indent=2),
            encoding="utf-8",
        )
    teacher = None
    if args.objective == "behavior":
        if "on_policy" in manifest:
            raise ValueError("On-policy behavior labels are unavailable; use MSE")
        if not args.init_student:
            raise ValueError("Behavior training requires --init-student")
        from experiments.feedback_dynamics import (
            prepare_feedback_history,
            behavior_distillation_loss,
        )

        teacher, _ = load_teacher(args.data, device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optim = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = torch.Generator().manual_seed(args.seed)
    first, best = 1, float("inf")
    if saved:
        optim.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first, best = saved["step"] + 1, saved["best_mse"]
    if first > args.steps:
        raise ValueError("Total steps must exceed checkpoint step")
    for step in range(first, args.steps + 1):
        idx = torch.randint(len(data["h"]), (args.batch_size,), generator=rng)
        h, r, u = (data[k][idx].to(device) for k in ("h", "r", "u"))
        optim.zero_grad(set_to_none=True)
        prediction = student(h, noise_for(args.noise_mode, u))
        feedback_mse = F.mse_loss(prediction, r)
        behavior_kl = None
        if teacher is None:
            loss = feedback_mse
        else:
            prompts = [data["prompts"][int(i)] for i in data["prompt_row"][idx]]
            lengths = torch.tensor([len(p) for p in prompts], device=device)
            padded = pad_sequence(
                prompts, batch_first=True, padding_value=teacher.cfg.pad_id
            ).to(device)
            with teacher_autocast():
                history = prepare_feedback_history(
                    teacher,
                    padded,
                    lengths,
                    data["past_feedback"][idx].to(device),
                    data["past_lengths"][idx].to(device),
                )
                loss, behavior_stats = behavior_distillation_loss(
                    teacher,
                    history,
                    prediction,
                    r,
                    data["next_tokens"][idx].to(device),
                    mse_weight=args.mse_weight,
                )
            behavior_kl = float(behavior_stats["kl"])
        loss.backward()
        optim.step()
        if step % 100 == 0 or step == args.steps:
            student.eval()
            total = 0.0
            with torch.no_grad():
                for i in range(0, len(dev["h"]), args.batch_size):
                    h, r, u = (
                        dev[k][i : i + args.batch_size].to(device)
                        for k in ("h", "r", "u")
                    )
                    total += float(
                        (student(h, noise_for(args.noise_mode, u)) - r)
                        .square()
                        .mean(-1)
                        .sum()
                    )
            mse = total / len(dev["h"])
            improved = mse < best
            best = min(mse, best)
            checkpoint = student.checkpoint() | {
                "step": step,
                "best_mse": best,
                "checkpoint_selection": "dev_mse",
                "metrics": {"mse": mse},
                "study_settings": settings,
                "optimizer": optim.state_dict(),
                "sampler_rng": rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
            }
            for name in ["last", "best"] if improved else ["last"]:
                temporary = args.run_dir / (name + ".tmp")
                torch.save(checkpoint, temporary)
                publish_checkpoint(temporary, args.run_dir / (name + ".pt"), step)
            student.train()
            log = {
                "step": step,
                "train_loss": float(loss.detach()),
                "feedback_mse": float(feedback_mse.detach()),
                "dev_mse": mse,
                "selection_metric": "dev_mse",
                "train_mse": float(loss.detach()),
            }  # Historical alias for readers.
            if behavior_kl is not None:
                log["behavior_kl"] = behavior_kl
            with (args.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log) + "\n")
            print(json.dumps(log), flush=True)


class EvaluationStudent(torch.nn.Module):
    def __init__(self, student, mode):
        super().__init__()
        self.student, self.cfg, self.mode = student, student.cfg, mode

    def forward(self, context, noise=None):
        return self.student(
            context, noise_for(self.mode, noise) if noise is not None else None
        )


def evaluate(args):
    from tokenizers import Tokenizer
    from latent_ar.feedback_student import generate_student_feedback
    from experiments.distill_gru_feedback import truncate

    manifest = verify(args.data)
    if args.output.exists():
        raise FileExistsError(args.output)
    data = torch.load(args.data / (args.split + ".pt"), weights_only=True)
    if not 0 < args.prompts <= len(data["prompts"]):
        raise ValueError("Invalid prompt count")
    teacher, _ = load_teacher(args.data, args.device)
    configured_precision, actual_precision, teacher_autocast = teacher_precision(
        args.data, torch.device(args.device)
    )
    saved = torch.load(args.student, map_location="cpu", weights_only=True)
    if saved["study_settings"]["teacher_hash"] != manifest["teacher_snapshot_sha256"]:
        raise ValueError("Student teacher differs")
    if (
        saved["study_settings"]["data_hashes"][args.split]
        != manifest["split_sha256"][args.split]
    ):
        raise ValueError("Evaluation split differs from training provenance")
    mode = saved["study_settings"]["noise_mode"]
    student = EvaluationStudent(
        FeedbackStudent.from_checkpoint(saved).to(args.device).eval(), mode
    )
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    reports = []
    for seed in args.seeds:
        totals = dict(
            matched=0,
            compared=0,
            max_length=0,
            tail_matched=0,
            tail_compared=0,
            tail_max_length=0,
            common_prefix=0,
            teacher_tokens=0,
            student_tokens=0,
            teacher_eos=0,
            student_eos=0,
            first_block_matches=0,
            feedback_mse_sum=0.0,
            feedback_cos_sum=0.0,
            feedback_blocks=0,
        )
        samples, per_prompt = [], []
        for i, prompt in enumerate(data["prompts"][: args.prompts]):
            ids = prompt.to(args.device)[None]
            lengths = torch.tensor([len(prompt)], device=args.device)

            def rng():
                return torch.Generator(device=args.device).manual_seed(seed + i)

            with teacher_autocast():
                a = truncate(
                    teacher.generate(
                        ids,
                        lengths,
                        max_new_tokens=args.max_new_tokens,
                        generator=rng(),
                    )[0].cpu(),
                    teacher.cfg.eos_id,
                )
                result = generate_student_feedback(
                    teacher,
                    student,
                    ids,
                    lengths,
                    max_new_tokens=args.max_new_tokens,
                    generator=rng(),
                    return_trace=True,
                )
            b = truncate(result["output_ids"][0].cpu(), teacher.cfg.eos_id)
            common = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                common += 1
            width = teacher.cfg.chunk_size
            prompt_metrics = {
                "story_id": int(data["prompt_story_ids"][i]),
                "prompt_row": i,
                "common_prefix": common,
                "teacher_tokens": len(a),
                "student_tokens": len(b),
                "teacher_eos": int(teacher.cfg.eos_id in a),
                "student_eos": int(teacher.cfg.eos_id in b),
                "first_block_matches": int(a[:width] == b[:width]),
                "feedback_mse_sum": 0.0,
                "feedback_cos_sum": 0.0,
                "feedback_blocks": 0,
            }
            for prefix, left, right in [("", a, b), ("tail_", a[width:], b[width:])]:
                prompt_metrics[prefix + "matched"] = sum(
                    x == y for x, y in zip(left, right)
                )
                prompt_metrics[prefix + "compared"] = min(len(left), len(right))
                prompt_metrics[prefix + "max_length"] = max(len(left), len(right))
                for suffix in ("matched", "compared", "max_length"):
                    totals[prefix + suffix] += prompt_metrics[prefix + suffix]
            totals["common_prefix"] += common
            totals["teacher_tokens"] += len(a)
            totals["student_tokens"] += len(b)
            totals["teacher_eos"] += int(teacher.cfg.eos_id in a)
            totals["student_eos"] += int(teacher.cfg.eos_id in b)
            totals["first_block_matches"] += int(a[:width] == b[:width])
            for j, rec in enumerate(result["records"]):
                if (
                    (j + 1) * width >= args.max_new_tokens
                    or not bool(rec["active"][0])
                    or int(rec["lengths"][0]) != width
                    or bool(rec["token_ids"][0].eq(teacher.cfg.eos_id).any())
                ):
                    continue
                true, pred = (
                    rec["true_feedback"][0].float(),
                    rec["predicted_feedback"][0].float(),
                )
                mse_value = float((true - pred).square().mean())
                cos_value = float(F.cosine_similarity(true[None], pred[None]))
                for metrics in (totals, prompt_metrics):
                    metrics["feedback_mse_sum"] += mse_value
                    metrics["feedback_cos_sum"] += cos_value
                    metrics["feedback_blocks"] += 1
            for prefix in ("", "tail_"):
                compared, maximum = (
                    prompt_metrics[prefix + "compared"],
                    prompt_metrics[prefix + "max_length"],
                )
                prompt_metrics[prefix + "match_rate"] = (
                    prompt_metrics[prefix + "matched"] / compared if compared else None
                )
                prompt_metrics[prefix + "match_rate_max_length"] = (
                    prompt_metrics[prefix + "matched"] / maximum if maximum else None
                )
            blocks = prompt_metrics["feedback_blocks"]
            prompt_metrics["feedback_mse"] = (
                prompt_metrics["feedback_mse_sum"] / blocks if blocks else None
            )
            prompt_metrics["feedback_cosine"] = (
                prompt_metrics["feedback_cos_sum"] / blocks if blocks else None
            )
            per_prompt.append(prompt_metrics)
            if i < 16:
                samples.append(
                    {
                        "story_id": int(data["prompt_story_ids"][i]),
                        "prompt": tokenizer.decode(prompt.tolist()),
                        "teacher": tokenizer.decode(a),
                        "student": tokenizer.decode(b),
                    }
                )
        totals["token_match_rate"] = totals["matched"] / max(1, totals["compared"])
        totals["token_match_rate_max_length"] = totals["matched"] / max(
            1, totals["max_length"]
        )
        totals["tail_match_rate"] = totals["tail_matched"] / max(
            1, totals["tail_compared"]
        )
        totals["tail_match_rate_max_length"] = totals["tail_matched"] / max(
            1, totals["tail_max_length"]
        )
        totals["mean_common_prefix"] = totals["common_prefix"] / args.prompts
        blocks = totals["feedback_blocks"]
        totals["feedback_mse"] = totals["feedback_mse_sum"] / blocks if blocks else None
        totals["feedback_cosine"] = (
            totals["feedback_cos_sum"] / blocks if blocks else None
        )
        reports.append(
            {
                "seed": seed,
                "metrics": totals,
                "per_prompt": per_prompt,
                "samples": samples,
            }
        )
    report = {
        "split": args.split,
        "prompts": args.prompts,
        "max_new_tokens": args.max_new_tokens,
        "teacher_sha256": manifest["teacher_snapshot_sha256"],
        "student_sha256": sha256(args.student),
        "noise_mode": mode,
        "temperature": manifest["token_temperature"],
        "teacher_configured_precision": configured_precision,
        "teacher_precision": actual_precision,
        "test_note": manifest["test_note"],
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--source-data", type=Path, required=True)
    c.add_argument("--prepared-data", type=Path, default=Path("data/prepared"))
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--train-prompts", type=int, default=16384)
    c.add_argument("--dev-prompts", type=int, default=512)
    c.add_argument("--test-prompts", type=int, default=128)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--temperature", type=float, default=0)
    c.add_argument("--device", default="cuda")
    c.add_argument("--seed", type=int, default=2031)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--run-dir", type=Path, required=True)
    t.add_argument("--resume", type=Path)
    t.add_argument("--init-student", type=Path)
    t.add_argument("--train-prompts", type=int, default=4096)
    t.add_argument("--hidden-dim", type=int, default=512)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--seed", type=int, default=2032)
    t.add_argument("--device", default="cuda")
    t.add_argument("--objective", choices=("mse", "behavior"), default="mse")
    t.add_argument("--noise-mode", choices=("none", "shared", "blind"), default="none")
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--mse-weight", type=float, default=0.1)
    o = sub.add_parser("collect-student")
    o.add_argument("--data", type=Path, required=True)
    o.add_argument("--student", type=Path, required=True)
    o.add_argument("--output", type=Path, required=True)
    o.add_argument("--train-prompts", type=int, default=4096)
    o.add_argument("--start-story", type=int, default=0)
    o.add_argument("--batch-size", type=int, default=32)
    o.add_argument("--device", default="cuda")
    m = sub.add_parser("matched-augment")
    m.add_argument("--base-data", type=Path, required=True)
    m.add_argument("--student-data", type=Path, required=True)
    m.add_argument("--output-offpolicy", type=Path, required=True)
    m.add_argument("--output-onpolicy", type=Path, required=True)
    m.add_argument("--base-prompts", type=int, default=4096)
    m.add_argument("--seed", type=int, default=2034)
    e = sub.add_parser("evaluate")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--student", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--split", choices=("test", "dev"), default="test")
    e.add_argument("--prompts", type=int, default=128)
    e.add_argument("--max-new-tokens", type=int, default=64)
    e.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    e.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {
        "collect": collect,
        "collect-student": collect_student,
        "matched-augment": matched_augment,
        "train": train,
        "evaluate": evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
