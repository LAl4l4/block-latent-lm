"""汇总 schedule-mask 各调度的最终 dev 指标，按调度合并多个 seed，以 causal 为基准。

稳态 = 第 2 个及以后的生成块；首词 = 块内第 1 个位置（衔接处）。
代价 = 本调度 − causal（同 seed 配对后取均值）；代价比例以 delayed 为 1。
最小丢失距离与每步词数由可见关系直接计算。
"""


from collections import defaultdict
import json
import re
import statistics
import sys
from pathlib import Path

from seam.schedules import rounds, visibility


def fmt(values, digits=3):
    if not values:
        return "—"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{statistics.stdev(values):.{digits}f}"


def structure(config):
    v = visibility(
        config["schedule"], config["prompt"], config["blocks"], config["block_size"],
        config.get("stagger"), config.get("segment", 1),
    )
    n = v.shape[0]
    i = list(range(n))
    lost = [a - b for a in i for b in i if a > b and not v[a, b] and not v[b, a]]
    steps = int(rounds(v, config["prompt"]).max()) + 1
    return (min(lost) if lost else None), config["blocks"] * config["block_size"] / steps


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/schedule-mask")
    arms = defaultdict(dict)
    configs = {}
    for run in sorted(root.iterdir()):
        if not (run / "final.json").exists():
            continue
        match = re.fullmatch(r"(.+)-s(\d+)", run.name)
        if not match:
            continue
        arm, seed = match.group(1), int(match.group(2))
        arms[arm][seed] = json.loads((run / "final.json").read_text(encoding="utf-8"))
        configs.setdefault(arm, json.loads((run / "config.json").read_text(encoding="utf-8")))
    base = arms.get("causal", {})
    delayed = arms.get("delayed", {})

    def paired(arm, key, other):
        return [arms[arm][s][key] - other[s][key] for s in arms[arm] if s in other]

    lines = [
        "| 调度 | seeds | 每步词数 | 最小丢失距离 | 稳态CE | 稳态代价 | 代价比例 | 首词CE | 末词CE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = sorted(arms, key=lambda a: statistics.mean(r["steady_ce"] for r in arms[a].values()))
    for arm in order:
        runs = arms[arm]
        distance, speed = structure(configs[arm])
        cost = paired(arm, "steady_ce", base)
        span = statistics.mean(paired("delayed", "steady_ce", base)) if delayed and base else None
        ratio = [c / span for c in cost] if span else []
        lines.append(
            f"| {arm} | {len(runs)} | {speed:.2f} | {distance if distance else '无'} | "
            f"{fmt([r['steady_ce'] for r in runs.values()])} | {fmt(cost)} | {fmt(ratio, 2)} | "
            f"{fmt([r['steady_first_ce'] for r in runs.values()], 2)} | "
            f"{fmt([r['steady_ce_by_slot'][-1] for r in runs.values()], 2)} |"
        )
    print("\n".join(lines))
    print("\n稳态各块内位置 CE（seed 均值，第 1 列为首词）：\n")
    for arm in order:
        slots = zip(*(r["steady_ce_by_slot"] for r in arms[arm].values()))
        print(f"- {arm}: " + " ".join(f"{statistics.mean(x):.2f}" for x in slots))
    print(
        "\n代价 = 本调度 − causal（nats/token，同 seed 配对，均值±标准差）；代价比例以 delayed 为 1。"
        "每步词数 = 112 个生成词 / 串行步数。CE 为 dev 上 4096 个固定窗口的自然语料 CE。"
    )


if __name__ == "__main__":
    main()
