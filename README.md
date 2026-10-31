# 块级并行语言模型：倒序错开解码

本仓库研究块级（block）并行文本生成。**当前主线**是块内倒序 + 错开起点的流水线解码（reverse-staggered decoding）。

这个想法由作者在 2026-10-31 05:03:50（UTC+11）提出，提出记录与推导见 [想法记录](docs/idea-reverse-staggered-decoding.md)。

## 核心想法

几个块并行生成时，下一块往往要在上一块写完之前开始，所以一部分词对之间无法互相看到。现有的错开方案在块内都是从左往右写，丢掉的恰好是最强的依赖：下一块的首词看不到上一块的结尾，我们称之为"衔接处"问题。

改成块内倒着写，或者按段倒序（段与段倒序，段内仍从左往右），再让各块错开 s 步起步，首词就会是块内最后写的词，那时上一块已经写完。

- **距离下界：** 被丢掉的依赖，距离一律至少为 `L − g + 1 + g·⌈(s−g+1)/g⌉⁺`。逐词倒序（g=1）时是 L+s；正序错开时最短距离只有 1。
- **因果性：** 方案保持因果，KV cache 可以照常使用，训练和推理一致。

## 当前结论（2026-10-31）

实验用只改 attention mask 的双流 Transformer，数据是 TinyStories，做了 3 个 seed、4 倍长训练、3 倍容量和交叉评委的样本评估。下表中的代价是比普通 causal LM 多出的 nat/词：

| 调度 | 每步词数 | 代价 |
| --- | ---: | ---: |
| **按段倒序 reverse8g8** | 1.75 | **0.276** |
| 正序错开 forward8 | 1.75 | 0.350 |
| 延迟块 delayed | 1.75 | 0.508 |
| **完全并行倒序 reverse0** | 7 | **0.466** |
| PSF 式着陆目标（非因果） | 7 | 0.498 |

- **成立的部分：**
  - 在相同并行度下，倒序是所有测过的方案中最好的因果调度。
  - 信息代价只有 0.01 到 0.02。
  - 样本中衔接处的评委 CE 从约 11.5 降到约 2.7。
  - 相对正序的优势随训练和容量扩大。
- **速度与质量的权衡：**
  - 倒序在 1.75 到 7 词/步的每个速度上都优于正序，差距从 0.08 增大到 0.11。
  - 2 倍速时按段倒序 reverse8g8 最好；更高并行时逐词倒序最好（0.356 到 0.459）。
  - 段更短的按段倒序跳跃词太多，反而变差。
- **未解决的部分：**
  - 倒着写更难学，小模型上约 0.25 的排序代价不随训练缩小，混合 causal 训练也降不下来。
  - 宽模型（3150 万参数）训练 20000 步后排序代价降到 0.146，但基线过拟合（训练集只有 2200 万词），结论待 100 万故事数据的重跑确认。
  - 块级离散计划只是把代价从词搬到计划码上。
  - 事后重解码一个词只会把断裂往后挪一格，替代不了顺序设计。
- **下一步：** 在不过拟合的数据上确认规模效应（`runs/schedule-mask-wide-1m`），再把逐词倒序用到更大的预训练模型上。

完整数字与判断见 [想法记录](docs/idea-reverse-staggered-decoding.md)，论文草稿见 [paper-v1-draft.md](docs/paper-v1-draft.md)。

## 复现 mask 实验

```powershell
& D:\Code\venv313\Scripts\python.exe -m seam.train_mask --schedule reverse --stagger 8 --segment 8
& D:\Code\venv313\Scripts\python.exe -m seam.summarize runs/schedule-mask
& D:\Code\venv313\Scripts\python.exe -m seam.sample_eval --runs causal-s42 reverse8g8-s42 --judge causal-s43 --out runs/schedule-mask/sample-eval.json
```

- 调度可选 `causal`、`delayed`、`forward`、`reverse`、`landing`。
- 参数含义：`--stagger` 是错开步数 s，`--segment` 是段长 g，`--mix-causal` 是混合 causal 训练的比例，`--refine` 给跳跃词（reverse）或块首（forward）加重解码查询，`--data data/prepared-1m` 用 100 万故事的数据。
- 批量脚本：`scripts/run_schedule_mask.bat`（批次 A）、`scripts/run_schedule_overnight.bat`（多 seed、长训练）、`scripts/run_schedule_capacity.bat`（宽模型）、`scripts/run_schedule_samples2.bat`（交叉评委）、`scripts/run_schedule_mix.bat`（混合训练）、`scripts/run_schedule_wide_long.bat`（宽模型 20000 步）、`scripts/run_schedule_tradeoff.bat`（权衡曲线）、`scripts/run_schedule_refine_wide1m.bat`（重解码与 1M 数据重跑）。
- 结果写入 `runs/schedule-mask*/`，这些目录不进 Git。
- 测试：`python -m pytest tests/test_seam_schedules.py`。

## 块级潜变量系统（背景与早期主线）

倒序想法源自这个系统中的首词问题：

- 主干 B 按 16 词一块推进；
- 局部解码器 D（GRU + VAE）负责写出每块的词；
- 历史编码器 E 把写好的块回灌给主干；
- 反馈学生 P 在上一块还没写完时，给主干提供替代输入。

| 文档 | 内容 |
| --- | --- |
| [PROGRESS.md](PROGRESS.md) | 研究进度、已完成与待做事项 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | B/D/E/P 与 h/z/f 记号、两块前瞻和 cache 修复 |
| [paper-plan.md](docs/paper-plan.md) | 延迟反馈方向的论文计划与第 1 步结果 |
| [pair-feedback-protocol.md](docs/pair-feedback-protocol.md) | 两块一组调度与验收口径 |
| [calm-repair-results.md](docs/calm-repair-results.md) | CALM-16 排查与修复 |
| [vae-calm-16.md](docs/vae-calm-16.md) | 16 词 VAE → CALM 式主干的训练路线 |

继续训练旧主模型（第 3100 步）：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py --resume runs/main/last.pt
```

## 目录

| 路径 | 用途 |
| --- | --- |
| `seam/` | **倒序错开主线**：`schedules.py`（可见关系）、`models.py`（双流模型）、`train_mask.py`、`train_plan.py`、`sample_eval.py`、`plan_samples.py`、`summarize.py` |
| `scripts/` | 批量运行脚本（`.bat`） |
| `paper/reverse-staggered/` | arXiv v1 稿件（LaTeX）与上传包 |
| `latent_ar/`、`experiments/` | 早期块级潜变量系统与其实验，保持原样 |
| `configs/` | codec、VAE、CALM、反馈实验配置 |
| `docs/` | 想法记录、论文草稿、方法与复现说明 |
| `paper/` | 早期论文材料与历史记录 |
| `artifacts/` | 小型实验报告与 JSON 结果 |
| `tests/` | 模型、调度与复现检查 |
| `runs/`、`data/` | 训练结果与数据，只存在本地，不进 Git |

## 仓库说明

- 远程仓库：`github.com/LAl4l4/block-latent-lm`，目前为私有。
- 2026-10-31 推送前，所有大文件（`*.pt`、`*.bin`、`*.npy`、`*.parquet`、准备好的数据）已从 Git 历史中删除，只保留在本地。提交的时间和内容不变，但哈希值已改变：旧文档中提到的 `1763f0f` 现在对应 `45a20d2`。
- 大文件也已不再使用 Git LFS。第 3100 步的备份 `artifacts/checkpoints/main-step3100.pt` 只在本地保存。
