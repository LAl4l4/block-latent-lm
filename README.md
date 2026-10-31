# Latent AR：块级主干与小输出头

[当前研究进度与接续记录](PROGRESS.md)：汇总已完成训练与实测、暂停和待执行项目，以及共享随机数的1份/8份对照建议。方案与成功结果分开记录。

[当前架构约定与实现快照](ARCHITECTURE.md)：统一B/D/E/P和h/z/f记号，固定“学生与GRU共享同一份选词噪声”的配对目标，记录两块一步前瞻、缓存修复、准确参数量和已测/待测状态。后续讨论优先以此文档对齐。

[两块一步前瞻实测](artifacts/pair-feedback/results.md)：新增组末真实反馈修复cache和双块批量解码，当前固定64词耗时有下降，但单步反馈预测尚不足以保持第二块质量。[调度与验收口径](docs/pair-feedback-protocol.md)。

[滑动窗口反馈学生](docs/window-feedback-student.md)：新增小型attention学生，读取当前及前两块主干状态；与只读当前状态的同结构学生对照。实际结果见 [实验报告](artifacts/feedback-attention-results.md)。当前尚无异步加速结论。

[CALM-16排查与实测修复](docs/calm-repair-results.md)：保留16-token主干和冻结VAE的局部GRU联合CE基线已完成10000步及验证集上下文消融，原MLP反馈蒸馏结果见 [首轮报告](artifacts/gru-feedback-distillation-results.md)。该基线需要块内顺序选词；原单次连续生成头仍未达到可用质量。

Energy训练路线：[16-token VAE → CALM 式主干](docs/vae-calm-16.md)。先运行 `train_vae.py`，采样重建达标后运行 `train_calm.py`；包含后验 Energy 训练及实际文本回灌。反馈蒸馏使用上述独立GRU实验入口。

仓库旧训练入口的默认四词路线：主干按四 token 的 latent 块推进，小输出头内部选择四个词，再编码成回灌 latent。它区别于上述当前16词反馈蒸馏研究架构；此路线包含离散选择，其异步加速尚未实现和验证。

独立的 [16-token 真实反馈实验](docs/feedback-distillation-16.md)：使用 `configs/prior-16-feedback.json`，冻结已训练的 codec，训练有限混合 latent 输出头，并逐块解码实际文本后重新编码回灌；该实验是早期方案，区别于上述GRU反馈蒸馏。

## 继续训练

已完整恢复第 **3100** 步，没有启动新的主模型训练。执行：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py --resume runs/main/last.pt
```

将从第 3101 步继续，恢复优化器与随机状态。活动训练目录不再进入 Git 扫描。保存遇到临时读锁会重试；持续被占用时另存版本文件，终端显示实际保存路径。

## 目录

| 路径 | 用途 |
|---|---|
| `latent_ar/` | 当前模型、共用训练逻辑及论文对照模块 |
| `configs/` | 当前小输出头和 codec 两份配置 |
| `docs/` | 当前方法与复现说明 |
| `paper/` | 论文材料索引、数据来源、环境、原始结果和历史研究记录 |
| `experiments/` | 离线评估、对照与归档恢复工具 |
| `artifacts/checkpoints/main-step3100.pt` | Git LFS 保存的第 3100 步固定备份 |
| `runs/main/` | 活动训练模型、日志与恢复记录，不由 Git 跟踪 |
| `runs/` 其他目录 | 关键初始化、codec 和论文基线权重 |
| `data/` | 原始与准备数据，不由 Git 实时跟踪 |
| `tests/` | 模型、保存与复现检查 |

[论文材料索引](paper/README.md) · [当前方法](docs/method.md) · [继续训练、评估与恢复归档](docs/reproduce.md)

根目录 `train_prior.py` 是当前训练入口，`train_codec.py` 用于重建 codec，`generate.py` 用于生成；`train.py` 保留共用训练工具和历史基线入口。旧工具已移到 `experiments/`，使用 `python -m experiments.<name>` 运行。

整理前完整研究快照为 Git 提交 `1763f0f`，研究数据和旧权重由本地 Git LFS 保留，可按 `paper/artifacts.json` 恢复。旧环境和缓存可重建；历史依赖记录在 `paper/archive/`。本项目尚未推送到远端。
