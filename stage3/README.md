# 阶段 3：三级解码

2026-11-01 起，进行中。结构是：大 latent 指明 64 词的方向，联合采样出 4 个前后连续的块 latent z，每块 16 词再按倒序错开并行解码。用意是让 z 承担块级内容，看倒序错开能否在此基础上解决衔接处，并降低阶段 2 剩下的排序代价。

提出记录、结构和第 0 步结果见 [docs/idea-three-level.md](docs/idea-three-level.md)。

## 进展

- 第 0 步，连续块 latent（`latent/`）：结果已写入文档。
- 强 z + 小窗口解码器（`window/`）：扫描在 2026-11-03 跑完，结果在本地 `runs/schedule-window/`，还没有写入文档。

## 目录

| 路径 | 内容 |
| --- | --- |
| `schedules.py` | 在阶段 2 的调度上加块窗口（`window_visibility`） |
| `models.py` | LatentLM（每块 K 个连续潜变量槽，VAE）与 WindowLM（强 z + 小窗口解码器），复用阶段 2 的双流层 |
| `latent/` | `train`、`samples`、`one_block`（在真实上下文里重采样一块） |
| `window/` | `train`、`samples` |
| `scripts/` | 批量运行脚本（`.bat`） |
| `tests/` | 两个模型的测试 |
| `docs/` | 方向记录与结果 |

阶段 3 的代码会直接调用阶段 2 的调度、模型和采样评估，所以依赖 `stage2/`。

## 运行

在仓库根目录：

```powershell
& D:\Code\venv313\Scripts\python.exe -m stage3.latent.train --help
& D:\Code\venv313\Scripts\python.exe -m stage3.window.train --help
```

结果写入 `runs/schedule-latent/`、`runs/schedule-window/`。
