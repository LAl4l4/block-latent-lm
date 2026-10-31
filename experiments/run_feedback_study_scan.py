"""顺序运行数据量/容量对照；每次只占用一个GPU训练任务。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/feedback-study/greedy-data"))
    parser.add_argument("--root", type=Path, default=Path("runs/feedback-study/scan"))
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    # 固定更新次数与batch；数据量变化同时意味着平均样本重复次数变化。
    # 这些是单seed探索，主要目标是观察容量与泛化，不能据此拟合普适scaling law。
    for prompts in (1024, 4096, 16384):
        for width in (512, 1024):
            name = f"n{prompts}-h{width}-s2032"
            directory = args.root / name
            last = directory / "last.pt"
            command = [
                sys.executable,
                "-m",
                "experiments.feedback_study",
                "train",
                "--data", str(args.data),
                "--run-dir", str(directory),
                "--train-prompts", str(prompts),
                "--hidden-dim", str(width),
                "--layers", "2",
                "--steps", str(args.steps),
                "--batch-size", "256",
                "--seed", "2032",
                "--objective", "mse",
                "--noise-mode", "none",
                "--device", "cuda",
            ]
            if last.exists():
                saved = torch.load(last, weights_only=True, map_location="cpu")
                settings = saved["study_settings"]
                if settings["train_prompts"] != prompts or settings["student"]["hidden_dim"] != width:
                    raise ValueError(f"已有检查点与任务配置不一致：{directory}")
                done = saved["step"]
                del saved
                if done >= args.steps:
                    print(json.dumps({"case": name, "status": "already_finished"}), flush=True)
                    continue
                command.extend(("--resume", str(last)))

            print(json.dumps({"case": name, "status": "running"}), flush=True)
            # 训练日志追加写入，保留故障前输出；子进程退出后才开始下一组。
            with (args.root / f"{name}.log").open("a", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({"case": name, "status": "finished"}), flush=True)


if __name__ == "__main__":
    main()
