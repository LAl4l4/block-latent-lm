"""汇总 runs/pair-pilot 下各模式的 dev 指标，输出 Markdown 表。

第一张表是各运行自身的指标；第二张表以同预算 sync 运行为基准做跨运行比较：

  延迟代价 = 本运行 pair 调度下组内第二块 CE − sync 运行同步调度下第二块 CE
  历史优势 = sync 运行第二块在错配历史下的 CE − 正确历史下的 CE
  代价比例 = 延迟代价 / 历史优势

sync 运行自己的 pair 列使用冻结 MSE 学生，即"事后蒸馏 P"的负面基线。
判定阈值见 docs/paper-plan.md 第 1 步，在看结果前确定。
"""

import json
import sys
from pathlib import Path

COLUMNS = (
    ("sync_ce", "同步CE"),
    ("pair_ce", "pair CE"),
    ("sync_odd_ce", "第二块CE(同步)"),
    ("pair_odd_ce", "第二块CE(pair)"),
    ("shuffled_odd_ce", "第二块CE(错配)"),
    ("pair_minus_sync_odd_ce", "第二块代价(本模型)"),
    ("sync_odd_first_ce", "第二块首词CE(同步)"),
    ("pair_odd_first_ce", "第二块首词CE(pair)"),
    ("pair_q_rel_error", "Q相对误差"),
    ("context_advantage", "上下文优势"),
)

# 代价比例的预先登记阈值：≤SMALL 主线成立；>LARGE 视为代价大。
SMALL, LARGE = 0.15, 0.40


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def dev_records(run):
    return [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if '"kind": "dev"' in line
    ]


def verdict(ratio):
    if ratio is None:
        return "—"
    if ratio <= SMALL:
        return "小：主线成立"
    if ratio <= LARGE:
        return "中：需要第2步(u)"
    return "大"


def difference(a, b):
    return None if a is None or b is None else a - b


def comparison(runs):
    baseline = next((r for r in runs if r.name.startswith("sync")), None)
    if baseline is None:
        return ["（没有 sync 运行，跳过跨运行比较）"]
    by_step = {r["step"]: r for r in dev_records(baseline)}
    lines = [
        "| 运行 | step | 延迟代价 | 历史优势 | 代价比例 | 首词代价 | 首词历史优势 "
        "| 组首块CE变化 | 判定 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        records = dev_records(run)
        if not records:
            continue
        record = records[-1]
        base = by_step.get(record["step"])
        if base is None:
            lines.append(f"| {run.name} | {record['step']} | sync 运行没有同一步的评估 |")
            continue
        cost = difference(record.get("pair_odd_ce"), base.get("sync_odd_ce"))
        advantage = difference(base.get("shuffled_odd_ce"), base.get("sync_odd_ce"))
        first_cost = difference(
            record.get("pair_odd_first_ce"), base.get("sync_odd_first_ce")
        )
        first_advantage = difference(
            base.get("shuffled_odd_first_ce"), base.get("sync_odd_first_ce")
        )
        ratio = None if cost is None or not advantage else cost / advantage
        even = difference(record.get("pair_even_ce"), base.get("sync_even_ce"))
        lines.append(
            f"| {run.name} | {record['step']} | {fmt(cost)} | {fmt(advantage)} | "
            f"{fmt(ratio)} | {fmt(first_cost)} | {fmt(first_advantage)} | {fmt(even)} "
            f"| {verdict(ratio)} |"
        )
    return lines


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pair-pilot")
    runs = sorted(p for p in root.iterdir() if (p / "metrics.jsonl").exists())
    lines = [
        "| 运行 | step | " + " | ".join(name for _, name in COLUMNS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in COLUMNS) + " |",
    ]
    for run in runs:
        dev = dev_records(run)
        for record in (dev[0], dev[-1]) if len(dev) > 1 else dev:
            lines.append(
                f"| {run.name} | {record['step']} | "
                + " | ".join(fmt(record.get(key)) for key, _ in COLUMNS)
                + " |"
            )
    print("\n".join(lines))
    print(
        "\n第二块代价(本模型) = 同一模型 pair 条件下组内第二块 CE − 同步条件下第二块 CE。"
        "sync 运行的 pair 列使用冻结的 MSE 学生。CE 标签为语料真实文字。\n"
    )
    print("\n".join(comparison(runs)))
    print(
        f"\n跨运行比较以同预算 sync 运行为基准（见文件头）。代价比例 ≤{SMALL} 判为小，"
        f">{LARGE} 判为大。sync 行即事后蒸馏 P 的代价。"
    )


if __name__ == "__main__":
    main()
