# Latent AR：块级主干与小输出头

当前默认方案：主干按四 token 的 latent 块推进，小输出头内部选择四个词，再编码成回灌 latent。decoder 只负责还原。这包含离散选择，不是纯连续向量回归；异步加速尚未实现和验证。

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
