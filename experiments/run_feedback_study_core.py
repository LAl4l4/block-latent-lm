"""可恢复的顺序实验清单：损失、共享随机数、访问状态和闭环。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path("runs/feedback-study")
ARTIFACTS = Path("artifacts/feedback-supervision-study")
BASELINE = ROOT / "scan/n4096-h512-s2032/best.pt"
SEEDS = (2032, 2033, 2034)


def execute(label, arguments, complete=None, training_steps=None):
    """已有完成结果直接跳过，未结束训练通过last恢复，禁止覆盖旧实验。"""
    if complete is not None and complete.exists():
        if training_steps is None:
            print(json.dumps({"case": label, "status": "already_finished"}), flush=True)
            return
        saved = torch.load(complete, weights_only=True, map_location="cpu")
        step = saved["step"]
        del saved
        if step >= training_steps:
            print(json.dumps({"case": label, "status": "already_finished"}), flush=True)
            return
        arguments = [*arguments, "--resume", str(complete)]

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"case": label, "status": "running"}), flush=True)
    with (logs / f"{label}.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-m", "experiments.feedback_study", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(json.dumps({"case": label, "status": "finished"}), flush=True)


def train_case(name, data, seed, *, prompts=4096, objective="mse", noise="none", init=None, steps=3000, lr=3e-4):
    directory = ROOT / "core" / name
    args = [
        "train", "--data", str(data), "--run-dir", str(directory),
        "--train-prompts", str(prompts), "--hidden-dim", "512", "--layers", "2",
        "--steps", str(steps), "--batch-size", "256", "--seed", str(seed),
        "--objective", objective, "--noise-mode", noise, "--learning-rate", str(lr),
        "--device", "cuda",
    ]
    if init is not None:
        args.extend(("--init-student", str(init)))
    execute(name, args, directory / "last.pt", steps)


def core_training():
    # 三个scratch训练seed用于基线/随机噪声复验。
    for seed in SEEDS[1:]:
        train_case(f"baseline-s{seed}", ROOT / "greedy-data", seed)

    # 两种目标从完全相同的初始化开始；主比较固定500步last，而非按test挑步数。
    for seed in SEEDS:
        for objective in ("mse", "behavior"):
            train_case(
                f"objective-{objective}-s{seed}", ROOT / "greedy-data", seed,
                objective=objective, init=BASELINE, steps=500, lr=1e-4,
            )

    # shared与blind都保留16维噪声投影，参数量和初始化严格相同。
    for seed in SEEDS:
        for noise in ("blind", "shared"):
            train_case(f"noise-{noise}-s{seed}", ROOT / "stochastic-data", seed, noise=noise)


def state_training():
    on_data = ROOT / "student-new-data"
    execute(
        "collect-student-new",
        ["collect-student", "--data", str(ROOT / "greedy-data"),
         "--student", str(BASELINE), "--output", str(on_data),
         "--start-story", "4096", "--train-prompts", "4096", "--batch-size", "32", "--device", "cuda"],
        on_data / "manifest.json",
    )
    off, on = ROOT / "matched-offpolicy-data", ROOT / "matched-onpolicy-data"
    if not (off / "manifest.json").exists() or not (on / "manifest.json").exists():
        # 两个目录必须一并新建；已有半成品不会被覆盖。
        execute(
            "match-state-data",
            ["matched-augment", "--base-data", str(ROOT / "greedy-data"),
             "--student-data", str(on_data), "--output-offpolicy", str(off),
             "--output-onpolicy", str(on), "--base-prompts", "4096", "--seed", "2034"],
        )
    for seed in SEEDS:
        for mode, data in (("offpolicy", off), ("onpolicy", on)):
            train_case(
                f"state-{mode}-s{seed}", data, seed, prompts=8192,
                init=BASELINE, steps=1000, lr=1e-4,
            )


def evaluation_cases():
    for prompts in (1024, 4096, 16384):
        for width in (512, 1024):
            name = f"scan-n{prompts}-h{width}"
            yield name, ROOT / "greedy-data", ROOT / f"scan/n{prompts}-h{width}-s2032/best.pt"
    for seed in SEEDS[1:]:
        yield f"baseline-s{seed}", ROOT / "greedy-data", ROOT / f"core/baseline-s{seed}/best.pt"
    for seed in SEEDS:
        for objective in ("mse", "behavior"):
            yield f"objective-{objective}-s{seed}", ROOT / "greedy-data", ROOT / f"core/objective-{objective}-s{seed}/last.pt"
        for mode in ("offpolicy", "onpolicy"):
            yield f"state-{mode}-s{seed}", ROOT / f"matched-{mode}-data", ROOT / f"core/state-{mode}-s{seed}/last.pt"
        for noise in ("blind", "shared"):
            yield f"noise-{noise}-s{seed}", ROOT / "stochastic-data", ROOT / f"core/noise-{noise}-s{seed}/best.pt"


def evaluate_all():
    for name, data, checkpoint in evaluation_cases():
        output = ARTIFACTS / "evaluations" / f"{name}-64.json"
        execute(
            f"evaluate-{name}",
            ["evaluate", "--data", str(data), "--student", str(checkpoint),
             "--output", str(output), "--split", "test", "--prompts", "128",
             "--max-new-tokens", "64", "--seeds", "41", "--device", "cuda"],
            output,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "states", "evaluate"))
    args = parser.parse_args()
    {"train": core_training, "states": state_training, "evaluate": evaluate_all}[args.phase]()


if __name__ == "__main__":
    main()
