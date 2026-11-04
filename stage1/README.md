# 阶段 1：块级潜变量系统（已停止）

2026-10-28 至 10-31。主干 B 按 16 词一块推进，局部解码器 D（GRU + VAE）写出每块的词，历史编码器 E 把写好的块回灌给主干，反馈学生 P 在上一块还没写完时提供替代输入。研究的问题是"不等采样回灌"要付出多少代价。

这一阶段在 10-31 停止：延迟反馈让下一块首词的代价很高，由此提出了阶段 2 的倒序错开解码（见 [stage2/docs/idea-reverse-staggered.md](../stage2/docs/idea-reverse-staggered.md)）。

## 代码在哪里

这一阶段的代码（`latent_ar/`、`experiments/`、根目录的 `train*.py` 与 `generate.py`、`configs/`、`artifacts/`、`paper/archive/`、`paper/experiments/` 及相应测试）已在 2026-11-04 从工作区删除，完整保留在 tag `stage1-latent-blocks`：

```powershell
git checkout stage1-latent-blocks      # 查看或运行旧代码
git checkout master                    # 回到当前代码
```

下面的文档（在 `docs/` 下）原样保留，其中指向 `latent_ar/`、`experiments/`、`artifacts/` 的链接需在该 tag 下查看。训练结果仍在本地 `runs/` 下（`main/`、`calm-16/`、`feedback-study/` 等），第 3100 步主模型的备份在 `runs/main/backup-step3100.pt`。

## 文档

| 文档 | 内容 |
| --- | --- |
| [PROGRESS.md](docs/PROGRESS.md) | 研究进度与结论（截至 10-31） |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | B/D/E/P 与 h/z/f 记号、两块前瞻和 cache 修复 |
| [CHANGELOG.md](docs/CHANGELOG.md) | 变更记录 |
| [observations.md](docs/observations.md) | 作者的两条观察笔记 |
| [method.md](docs/method.md) | 块级 latent 主干与块内离散输出头 |
| [paper-plan.md](docs/paper-plan.md) | 延迟反馈方向的论文计划与第 1 步结果 |
| [pair-feedback-protocol.md](docs/pair-feedback-protocol.md) | 两块一组调度与验收口径 |
| [vae-calm-16.md](docs/vae-calm-16.md) | 16 词 VAE → CALM 式主干的训练路线 |
| [calm-repair-results.md](docs/calm-repair-results.md)、[calm-official-comparison.md](docs/calm-official-comparison.md)、[calm-training-audit-20261030.md](docs/calm-training-audit-20261030.md) | CALM-16 排查、修复与官方实现核对 |
| [feedback-distillation-16.md](docs/feedback-distillation-16.md)、[feedback-supervision-study.md](docs/feedback-supervision-study.md)、[gru-feedback-distillation-plan.md](docs/gru-feedback-distillation-plan.md)、[gru-baseline-distillation-run.md](docs/gru-baseline-distillation-run.md)、[window-feedback-student.md](docs/window-feedback-student.md) | 反馈蒸馏的实验设计与运行 |
| [block-transformer-feedback-audit.md](docs/block-transformer-feedback-audit.md) | Block Transformer 相关工作核对 |
| [reproduce.md](docs/reproduce.md) | 当时的使用与复现说明 |
| [results/](docs/results/) | 实验报告：CALM-16 最终评估、反馈 attention、GRU 反馈蒸馏、研究选题 |
