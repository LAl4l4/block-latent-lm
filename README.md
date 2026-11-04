# 块级并行语言模型：衔接处的依赖该怎么丢

本仓库研究块级（block）并行文本生成：几个块同时生成时，必须丢掉一部分词之间的依赖，丢哪些最划算。研究分三个阶段，现在的代码只服务阶段 2 和阶段 3。

| 阶段 | 时间 | 内容 | 状态 | 文档 |
| --- | --- | --- | --- | --- |
| 1 | 10-28 至 10-31 | 块级潜变量系统：主干 B、局部解码器 D、历史编码器 E、反馈学生 P | 已停止，代码已删除 | [docs/stage1](docs/stage1/README.md) |
| 2 | 10-31 至 11-04 | 倒序错开解码：把每块的首词放到最后写 | 论文已完成，待投 arXiv | [docs/stage2/idea.md](docs/stage2/idea.md)、[论文](paper/reverse-staggered/README.md) |
| 3 | 11-01 起 | 三级解码：大 latent → 联合采样的块 latent → 倒序错开解码 | 进行中 | [docs/stage3/idea.md](docs/stage3/idea.md) |

## 阶段 2 的结论

几个块错开起步时，正序方案丢掉的恰好是最强的依赖：下一块的首词看不到上一块的结尾（衔接处）。把块内顺序改成倒序或按段倒序后，首词变成块内最后写的词，丢掉的依赖数和并行度都不变，但每个被丢掉的依赖都至少隔开一个块。

只改 attention mask 的对照实验表明：

- 同样数量的依赖，挪离衔接处后信息代价降到原来的 1/20 以下，样本里的衔接处断裂基本消失；
- 剩下的是排序代价（约 0.25 nat/词），集中在每块一个跳跃词上，扩大模型和加块级计划都没有降低它。

完整数字见论文和 [docs/stage2/idea.md](docs/stage2/idea.md)。

## 阶段 3 的进展

- 第 0 步（连续块 latent，`seam/latent/`）的结果写在 [docs/stage3/idea.md](docs/stage3/idea.md)。
- 强 z + 小窗口解码器（`seam/window/`）的扫描已在 2026-11-03 跑完，结果在本地 `runs/schedule-window/`，还没有写入文档。

## 目录

| 路径 | 内容 |
| --- | --- |
| `seam/` | 研究代码。`common.py`、`schedules.py`、`models.py`、`data.py` 为公共部分；`mask/`、`plan/` 属于阶段 2；`latent/`、`window/` 属于阶段 3 |
| `data_prep/` | TinyStories 下载与预处理，以及数据卡 |
| `scripts/stage2/`、`scripts/stage3/` | 各阶段的批量运行脚本（`.bat`） |
| `tests/` | 调度与模型的测试 |
| `paper/reverse-staggered/` | 阶段 2 的论文（LaTeX）、作图脚本和 arXiv 上传包 |
| `docs/stage1/`、`docs/stage2/`、`docs/stage3/` | 各阶段的想法记录、实验记录与结果 |
| `timestamps/` | 想法提出时间的 OpenTimestamps 证明，不要改动 |
| `runs/`、`data/` | 训练结果与数据，只存在本地，不进 Git |

## 运行

```powershell
# 数据（输出到 data/prepared）
& D:\Code\venv313\Scripts\python.exe -m data_prep.download
& D:\Code\venv313\Scripts\python.exe -m data_prep.prepare

# 阶段 2：mask 实验、汇总、样本评估
& D:\Code\venv313\Scripts\python.exe -m seam.mask.train --schedule reverse --stagger 8 --segment 8
& D:\Code\venv313\Scripts\python.exe -m seam.mask.summarize runs/schedule-mask
& D:\Code\venv313\Scripts\python.exe -m seam.mask.sample_eval --runs causal-s42 reverse8g8-s42 --judge causal-s43 --out runs/schedule-mask/sample-eval.json

# 阶段 3
& D:\Code\venv313\Scripts\python.exe -m seam.latent.train --help
& D:\Code\venv313\Scripts\python.exe -m seam.window.train --help

# 测试
& D:\Code\venv313\Scripts\python.exe -m pytest
```

- 调度可选 `causal`、`delayed`、`forward`、`reverse`、`landing`。`--stagger` 是错开步数 s，`--segment` 是段长 g。
- 批量脚本里已写好各组实验的参数，运行结果写入 `runs/` 下同名目录。

## 仓库说明

- 远程仓库：`github.com/LAl4l4/block-latent-lm`，目前为私有。
- 阶段 1 的代码（`latent_ar/`、`experiments/`、根目录的 `train*.py`、`configs/`、`artifacts/`）在 2026-11-04 整理时删除，完整保留在 tag `stage1-latent-blocks`：`git checkout stage1-latent-blocks` 即可取回。
- 2026-10-31 推送前，所有大文件（`*.pt`、`*.bin`、`*.npy`、`*.parquet`、准备好的数据）已从 Git 历史中删除，只保留在本地。提交的时间和内容不变，但哈希值已改变：旧文档中提到的 `1763f0f` 现在对应 `45a20d2`。
