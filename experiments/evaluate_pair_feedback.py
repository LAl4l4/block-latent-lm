"""Measure one-step pair-feedback components and fixed-history effects."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from download_data import sha256
from experiments.distill_gru_feedback import load_teacher
from experiments.feedback_dynamics import prepare_feedback_history, next_block_logits
from latent_ar.feedback_student import load_feedback_student
from train import precision_context


def setup(data, split, device):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in [
        ("teacher.pt", manifest["teacher_snapshot_sha256"]),
        (split + ".pt", manifest["split_sha256"][split]),
    ]:
        if sha256(data / name) != expected:
            raise ValueError(f"{name} hash mismatch")
    teacher, _ = load_teacher(data, device)
    teacher.requires_grad_(False)
    snap = torch.load(
        data / "teacher.pt", map_location="cpu", weights_only=True, mmap=True
    )
    precision = snap["config"]["training"]["precision"]
    del snap
    values = torch.load(
        data / (split + ".pt"), map_location="cpu", weights_only=True, mmap=True
    )
    return teacher, manifest, values, precision


def student_for(path, manifest, device):
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    identity = saved.get("study_settings", {}).get(
        "teacher_hash", saved.get("teacher_snapshot_sha256")
    )
    if identity != manifest["teacher_snapshot_sha256"]:
        raise ValueError(f"{path}: teacher hash mismatch")
    student = load_feedback_student(saved).to(device).eval()
    if student.cfg.noise_dim:
        raise ValueError("This greedy experiment requires a deterministic student")
    return student


def history_for(teacher, data, idx, device):
    prompts = [data["prompts"][int(i)] for i in data["prompt_row"][idx]]
    lengths = torch.tensor([len(x) for x in prompts], device=device)
    padded = pad_sequence(
        prompts, batch_first=True, padding_value=teacher.cfg.pad_id
    ).to(device)
    return prepare_feedback_history(
        teacher,
        padded,
        lengths,
        data["past_feedback"][idx].to(device),
        data["past_lengths"][idx].to(device),
    )


def metrics(ref, logits, tokens):
    mask = tokens.ne(-100)
    if not mask.any(-1).all():
        raise ValueError("Every row must contain at least one next token")
    logp = F.log_softmax(ref.float(), -1)
    logq = F.log_softmax(logits.float(), -1)
    kl = (logp.exp() * (logp - logq)).sum(-1)
    ref_ce = F.cross_entropy(
        ref.float().transpose(1, 2), tokens, ignore_index=-100, reduction="none"
    )
    ce = F.cross_entropy(
        logits.float().transpose(1, 2), tokens, ignore_index=-100, reduction="none"
    )
    return ce - ref_ce, kl, mask, logits[:, 0].argmax(-1).ne(ref[:, 0].argmax(-1))


def single(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    teacher, manifest, data, precision = setup(args.data, args.split, device)
    autocast = precision_context(device, precision)
    students = [(str(p), student_for(p, manifest, device)) for p in args.students]
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(data["h"]), generator=generator)[: args.blocks]
    if not len(indices):
        raise ValueError("No rows selected")
    records = []
    with torch.no_grad(), autocast():
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start : start + args.batch_size]
            hist = history_for(teacher, data, idx, device)
            h, r, tokens = (data[k][idx].to(device) for k in ("h", "r", "next_tokens"))
            ref = next_block_logits(teacher, hist, r, tokens).float()
            for name, student in students:
                prediction = student(h)
                logits = next_block_logits(teacher, hist, prediction, tokens).float()
                delta, kl, mask, changed = metrics(ref, logits, tokens)
                error = prediction.float() - r.float()
                relative = error.norm(dim=-1) / r.float().norm(dim=-1).clamp_min(1e-12)
                for j, i in enumerate(idx.tolist()):
                    valid = mask[j]
                    records.append(
                        {
                            "sample_index": i,
                            "story_id": int(data["story_id"][i]),
                            "block_index": int(data["block_index"][i]),
                            "student": name,
                            "mse": float(error[j].square().mean()),
                            "relative_error": float(relative[j]),
                            "per_position_delta_ce": [
                                float(delta[j, k]) if bool(valid[k]) else None
                                for k in range(tokens.shape[1])
                            ],
                            "overall_delta_ce": float(delta[j][valid].mean()),
                            "kl": float(kl[j][valid].mean()),
                            "first_argmax_changed": bool(changed[j]),
                            "valid_tokens": int(valid.sum()),
                        }
                    )
    summary = {}
    for name, _ in students:
        rows = [x for x in records if x["student"] == name]
        thresholds = {}
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2):
            selected = [x for x in rows if x["relative_error"] <= threshold]
            avg = lambda k: (
                statistics.mean(x[k] for x in selected) if selected else None
            )
            thresholds[str(threshold)] = {
                "count": len(selected),
                "coverage": len(selected) / len(rows),
                "delta_ce": avg("overall_delta_ce"),
                "kl": avg("kl"),
                "first_argmax_change_rate": avg("first_argmax_changed"),
            }
        summary[name] = {"count": len(rows), "thresholds": thresholds}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "samples.jsonl").open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    report = {
        "split": args.split,
        "seed": args.seed,
        "blocks": len(indices),
        "teacher_hash": manifest["teacher_snapshot_sha256"],
        "data_hash": manifest["split_sha256"][args.split],
        "precision": precision,
        "effective_precision": precision if device.type == "cuda" else "fp32",
        "note": "Fixed teacher history; CE is against teacher-generated next tokens, not corpus truth. Teacher-forced scores do not establish free-generation quality. Select thresholds on dev only.",
        "students": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def components(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    teacher, manifest, data, precision = setup(args.data, "test", device)
    student = student_for(args.student, manifest, device)
    candidates = [
        i
        for i in range(len(data["h"]))
        if data["next_tokens"][i].ne(-100).all()
        and not data["next_tokens"][i].eq(teacher.cfg.eos_id).any()
        and len(data["prompts"][int(data["prompt_row"][i])])
        <= 8 * teacher.cfg.chunk_size
    ]
    if not candidates:
        raise ValueError("No full non-EOS block with prompt of at most eight latents")
    index = max(
        candidates, key=lambda i: len(data["prompts"][int(data["prompt_row"][i])])
    )
    idx = torch.tensor([index])
    autocast = precision_context(device, precision)
    with torch.no_grad(), autocast():
        hist = history_for(teacher, data, idx, device)
        h = data["h"][idx].to(device)
        f = data["r"][idx].to(device)
        words = (
            data["next_tokens"][idx]
            .to(device)
            .masked_fill(
                data["next_tokens"][idx].to(device).eq(-100), teacher.cfg.pad_id
            )
        )
        lengths = torch.full((1,), teacher.cfg.chunk_size, device=device)

        def backbone(q):
            x = f[:, None].expand(-1, q, -1)
            valid = torch.cat(
                (hist.valid, torch.ones((1, q), dtype=torch.bool, device=device)), 1
            )
            positions = (
                hist.next_position[:, None] + torch.arange(q, device=device)[None]
            )
            return teacher.backbone(x, positions, valid, hist.caches)

        def decode(n):
            context = h.expand(n, -1)
            noise = torch.full((n, teacher.cfg.chunk_size), 0.5, device=device)
            z = teacher.sample_latent(context, noise=noise)
            scores = teacher.codec.decode(z[:, None]).float().clone()
            scores[..., teacher.cfg.pad_id] = -torch.inf
            scores[..., teacher.cfg.bos_id] = -torch.inf
            return scores.argmax(-1)

        def encode(n):
            return teacher.encode_history(words.expand(n, -1), lengths.expand(n))

        functions = {
            "backbone_q1": lambda: backbone(1),
            "backbone_q2": lambda: backbone(2),
            "decode_batch1": lambda: decode(1),
            "decode_batch2": lambda: decode(2),
            "encode_batch1": lambda: encode(1),
            "encode_batch2": lambda: encode(2),
            "predict_batch1": lambda: student(h),
        }
        result = {}
        for name, fn in functions.items():
            for _ in range(3):
                fn()
            samples = []
            for _ in range(args.repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t = time.perf_counter()
                fn()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - t) * 1000)
            quantiles = torch.quantile(
                torch.tensor(samples), torch.tensor([0.25, 0.5, 0.75])
            ).tolist()
            result[name] = {
                "p25_ms": quantiles[0],
                "median_ms": quantiles[1],
                "p75_ms": quantiles[2],
                "samples_ms": samples,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "teacher_hash": manifest["teacher_snapshot_sha256"],
                "precision": precision,
                "effective_precision": precision if device.type == "cuda" else "fp32",
                "warmup": 3,
                "repeats": args.repeats,
                "history_valid_latents": int(hist.next_position[0]),
                "sample_index": int(idx[0]),
                "timings": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("components")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--student", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--repeats", type=int, default=30)
    c.add_argument("--device", default="cuda")
    s = sub.add_parser("single")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--students", type=Path, nargs="+", required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--split", choices=("dev", "test"), default="test")
    s.add_argument("--blocks", type=int, default=256)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--seed", type=int, default=2043)
    s.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "components":
        if args.repeats < 1:
            parser.error("repeats must be positive")
        components(args)
    else:
        if min(args.blocks, args.batch_size) < 1:
            parser.error("blocks and batch-size must be positive")
        single(args)


if __name__ == "__main__":
    main()
