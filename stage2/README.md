# 阶段 2：倒序错开解码

2026-10-31 至 11-04。几个块错开起步时，正序方案丢掉的恰好是最强的依赖：下一块的首词看不到上一块的结尾（衔接处）。把块内顺序改成倒序或按段倒序，首词就变成块内最后写的词，丢掉的依赖数和并行度都不变，但每个被丢掉的依赖都至少隔开一个块。

只改 attention mask 的对照实验表明，同样数量的依赖挪离衔接处后，信息代价降到原来的 1/20 以下，样本里的衔接处断裂基本消失。剩下的是排序代价（约 0.25 nat/词），集中在每块一个跳跃词上，扩大模型和加块级计划都没有降低它。

想法的提出记录、推导和全部数字见 [docs/idea-reverse-staggered.md](docs/idea-reverse-staggered.md)，论文见 [paper/](paper/README.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `schedules.py` | 可见关系：每个调度下"谁能看到谁"，以及计划码时刻、采样轮次 |
| `models.py` | 双流 Transformer（TwoStreamLM）与带块级计划的 PlanLM |
| `mask/` | 主实验，各调度只差 attention mask：`train`、`sample_eval`（采样 + causal 评委）、`examples`（论文附录样例）、`summarize`（多 seed 汇总） |
| `plan/` | 每块加离散计划码：`train`、`samples` |
| `scripts/` | 批量运行脚本（`.bat`） |
| `tests/` | 距离下界、采样轮次、计划码与重解码的测试 |
| `docs/` | 想法记录与实验结果 |
| `paper/` | 论文（LaTeX）、作图脚本和 arXiv 上传包 |

## 运行

在仓库根目录：

```powershell
& D:\Code\venv313\Scripts\python.exe -m stage2.mask.train --schedule reverse --stagger 8 --segment 8
& D:\Code\venv313\Scripts\python.exe -m stage2.mask.summarize runs/schedule-mask
& D:\Code\venv313\Scripts\python.exe -m stage2.mask.sample_eval --runs causal-s42 reverse8g8-s42 --judge causal-s43 --out runs/schedule-mask/sample-eval.json
```

- 调度可选 `causal`、`delayed`、`forward`、`reverse`、`landing`、`meet`、`alternate`。`--stagger` 是错开步数 s，`--segment` 是段长 g。
- 批量脚本里已写好各组实验的参数，结果写入 `runs/schedule-mask*/`、`runs/schedule-plan/`。
