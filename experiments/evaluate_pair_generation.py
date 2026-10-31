"""Evaluate two-block lookahead quality and synchronized fixed-length timing."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer

from download_data import sha256
from experiments.distill_gru_feedback import load_teacher
from experiments.feedback_study import teacher_precision, verify
from latent_ar.feedback_student import FeedbackStudent, generate_student_feedback
from latent_ar.pair_feedback import generate_pair_feedback

MODES = ("teacher", "continuous", "pair_serial", "pair_batch")


def truncate(ids, eos):
    ids = ids.tolist()
    return ids[: ids.index(eos) + 1] if eos in ids else ids


def comparison(reference, other, chunk, max_tokens):
    def section(a, b):
        matched = sum(x == y for x, y in zip(a, b))
        compared, maximum = min(len(a), len(b)), max(len(a), len(b))
        return {
            "matched": matched,
            "compared": compared,
            "max_length": maximum,
            "match_rate": matched / compared if compared else None,
            "match_rate_max_length": matched / maximum if maximum else None,
        }

    prefix = 0
    for x, y in zip(reference, other):
        if x != y:
            break
        prefix += 1
    blocks = []
    for start in range(0, max_tokens, chunk):
        a, b = reference[start : start + chunk], other[start : start + chunk]
        blocks.append(
            {
                "index": start // chunk,
                **section(a, b),
                "both_present": bool(a and b),
                "whole_match": a == b if a and b else None,
            }
        )
    return {
        "full": section(reference, other),
        "tail": section(reference[chunk:], other[chunk:]),
        "first_block_match": reference[:chunk] == other[:chunk],
        "common_prefix": prefix,
        "second_block": blocks[1] if len(blocks) > 1 else None,
        "blocks": blocks,
    }


def run_mode(
    mode,
    teacher,
    student,
    prompt,
    lengths,
    max_tokens,
    seed,
    trace=False,
    verify_repair=False,
    fixed=False,
):
    generator = torch.Generator(device=prompt.device).manual_seed(seed)
    kwargs = dict(
        max_new_tokens=max_tokens,
        generator=generator,
        min_new_tokens=max_tokens if fixed else 0,
    )
    if mode == "teacher":
        return teacher.generate(prompt, lengths, **kwargs)
    if mode == "continuous":
        return generate_student_feedback(teacher, student, prompt, lengths, **kwargs)
    return generate_pair_feedback(
        teacher,
        student,
        prompt,
        lengths,
        **kwargs,
        parallel="batch" if mode == "pair_batch" else False,
        return_trace=trace,
        verify_repair=verify_repair,
    )


def quality(
    teacher, student, data, count, max_tokens, device, precision, seed, tokenizer
):
    records, samples, repair = [], [], []
    for i, prompt in enumerate(data["prompts"][:count]):
        ids = prompt.to(device)[None]
        lengths = torch.tensor([len(prompt)], device=device)
        outputs = {}
        with precision():
            for mode in MODES:
                outputs[mode] = truncate(
                    run_mode(
                        mode, teacher, student, ids, lengths, max_tokens, seed + i
                    )[0].cpu(),
                    teacher.cfg.eos_id,
                )
            if i < min(8, count):
                for mode in ("pair_serial", "pair_batch"):
                    traced = run_mode(
                        mode,
                        teacher,
                        student,
                        ids,
                        lengths,
                        max_tokens,
                        seed + i,
                        trace=True,
                        verify_repair=True,
                    )
                    if (
                        truncate(traced["output_ids"][0].cpu(), teacher.cfg.eos_id)
                        != outputs[mode]
                    ):
                        raise ValueError("Repair verification changed generation")
                    errors = [
                        r["max_cache_error"]
                        for r in traced["records"]
                        if "max_cache_error" in r
                    ]
                    repair.append(
                        {
                            "story_id": int(data["prompt_story_ids"][i]),
                            "mode": mode,
                            "max_active_h_kv_error": max(errors) if errors else None,
                            "repairs": len(errors),
                        }
                    )
        reference = outputs["teacher"]
        metrics = {
            mode: comparison(
                reference, outputs[mode], teacher.cfg.chunk_size, max_tokens
            )
            for mode in MODES
            if mode != "teacher"
        }
        metrics["pair_batch_vs_serial"] = comparison(
            outputs["pair_serial"],
            outputs["pair_batch"],
            teacher.cfg.chunk_size,
            max_tokens,
        )
        records.append(
            {
                "story_id": int(data["prompt_story_ids"][i]),
                "prompt_row": i,
                "seed": seed + i,
                "token_ids": outputs,
                "lengths": {k: len(v) for k, v in outputs.items()},
                "eos": {k: teacher.cfg.eos_id in v for k, v in outputs.items()},
                "metrics": metrics,
            }
        )
        if i < 16:
            samples.append(
                {
                    "story_id": int(data["prompt_story_ids"][i]),
                    "prompt": tokenizer.decode(prompt.tolist()),
                    **{
                        mode: tokenizer.decode(tokens)
                        for mode, tokens in outputs.items()
                    },
                }
            )
        if (i + 1) % 16 == 0:
            print(f"Quality: {i + 1}/{count}", flush=True)
    return records, samples, repair


def summarize(records, max_tokens, chunk):
    output = {}
    for mode in ("continuous", "pair_serial", "pair_batch", "pair_batch_vs_serial"):
        items = [r["metrics"][mode] for r in records]
        summary = {
            "first_block_match_rate": sum(x["first_block_match"] for x in items)
            / len(items),
            "mean_common_prefix": sum(x["common_prefix"] for x in items) / len(items),
        }
        for section in ("full", "tail"):
            matched = sum(x[section]["matched"] for x in items)
            compared = sum(x[section]["compared"] for x in items)
            maximum = sum(x[section]["max_length"] for x in items)
            summary[section] = {
                "matched": matched,
                "compared": compared,
                "max_length": maximum,
                "match_rate": matched / compared if compared else None,
                "match_rate_max_length": matched / maximum if maximum else None,
            }
        summary["blocks"] = []
        for idx in range((max_tokens + chunk - 1) // chunk):
            blocks = [x["blocks"][idx] for x in items]
            comparable = [x for x in blocks if x["both_present"]]
            matched, compared, maximum = (
                sum(x[k] for x in blocks) for k in ("matched", "compared", "max_length")
            )
            summary["blocks"].append(
                {
                    "index": idx,
                    "both_present": len(comparable),
                    "matched": matched,
                    "compared": compared,
                    "max_length": maximum,
                    "match_rate": matched / compared if compared else None,
                    "match_rate_max_length": matched / maximum if maximum else None,
                    "whole_match_rate": (
                        sum(x["whole_match"] for x in comparable) / len(comparable)
                        if comparable
                        else None
                    ),
                }
            )
        output[mode] = summary
    output["length_eos"] = {
        mode: {
            "mean_length": sum(r["lengths"][mode] for r in records) / len(records),
            "eos_count": sum(r["eos"][mode] for r in records),
        }
        for mode in MODES
    }
    return output


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def benchmark(
    teacher, student, prompts, batch_size, repeats, warmup, device, precision, seed
):
    batch = prompts[:batch_size]
    ids = pad_sequence(batch, batch_first=True, padding_value=teacher.cfg.pad_id).to(
        device
    )
    lengths = torch.tensor([len(p) for p in batch], device=device)
    samples = {mode: [] for mode in MODES}
    order_rng = random.Random(seed + batch_size)

    def once(mode, n):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with precision():
            output = run_mode(
                mode, teacher, student, ids, lengths, 64, seed + n, fixed=True
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000
        if output.shape != (batch_size, 64) or bool(
            output.eq(teacher.cfg.eos_id).any()
        ):
            raise ValueError(
                "Fixed timing must generate exactly 64 non-EOS tokens per prompt"
            )
        return elapsed

    for n in range(warmup):
        for mode in MODES:
            once(mode, n)
    for n in range(repeats):
        order = list(MODES)
        order_rng.shuffle(order)
        for mode in order:
            samples[mode].append(once(mode, n + warmup))
    return {
        mode: {
            "raw_ms": values,
            "median_ms": statistics.median(values),
            "p25_ms": percentile(values, 0.25),
            "p75_ms": percentile(values, 0.75),
            "actual_tokens_per_repeat": batch_size * 64,
        }
        for mode, values in samples.items()
    }


def main():
    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompts", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--benchmark-repeats", type=int, default=20)
    p.add_argument("--benchmark-warmup", type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    if (
        args.prompts < 4
        or args.max_new_tokens < 1
        or args.benchmark_repeats < 1
        or args.benchmark_warmup < 0
    ):
        raise ValueError("Invalid experiment size")
    manifest = verify(args.data)
    if manifest["token_temperature"] != 0:
        raise ValueError("Only greedy teacher is supported")
    data = torch.load(args.data / "test.pt", map_location="cpu", weights_only=True)
    if args.prompts > len(data["prompts"]):
        raise ValueError("Too many prompts")
    device = torch.device(args.device)
    teacher, _ = load_teacher(args.data, device)
    saved = torch.load(args.student, map_location="cpu", weights_only=True)
    if (
        saved.get("study_settings", {}).get(
            "teacher_hash", saved.get("teacher_snapshot_sha256")
        )
        != manifest["teacher_snapshot_sha256"]
    ):
        raise ValueError("Student teacher mismatch")
    if (
        saved.get("study_settings", {}).get("data_hashes", manifest["split_sha256"])[
            "test"
        ]
        != manifest["split_sha256"]["test"]
    ):
        raise ValueError("Student test provenance mismatch")
    student = FeedbackStudent.from_checkpoint(saved).to(device).eval()
    if student.cfg.noise_dim != 0:
        raise ValueError("Only greedy students are supported")
    configured, actual, precision = teacher_precision(args.data, device)
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    args.output.mkdir(parents=True)
    records, samples, repair = quality(
        teacher,
        student,
        data,
        args.prompts,
        args.max_new_tokens,
        device,
        precision,
        args.seed,
        tokenizer,
    )
    timing = {}
    for size in (1, 4):
        print(f"Timing batch size {size}", flush=True)
        timing[str(size)] = benchmark(
            teacher,
            student,
            data["prompts"],
            size,
            args.benchmark_repeats,
            args.benchmark_warmup,
            device,
            precision,
            args.seed,
        )
    report = {
        "teacher_sha256": manifest["teacher_snapshot_sha256"],
        "student_sha256": sha256(args.student),
        "test_sha256": manifest["split_sha256"]["test"],
        "split": "test",
        "prompts": args.prompts,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "configured_precision": configured,
        "actual_precision": actual,
        "device": str(device),
        "benchmark_repeats": args.benchmark_repeats,
        "benchmark_warmup": args.benchmark_warmup,
        "timing_note": "Includes prompt prefill and complete feedback encoding and cache repair; 64 non-EOS tokens each. Pair batch is batching, not CUDA streams.",
        "quality_note": "Always emit predicted second block; no acceptance policy or threshold.",
        "summary": summarize(records, args.max_new_tokens, teacher.cfg.chunk_size),
        "per_prompt": records,
        "samples": samples,
        "repair_verification": repair,
        "timing": timing,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": report["summary"], "timing": timing}, ensure_ascii=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
