# Latent AR：固定 latent 目标与块内小头对照

新训练默认不使用 Energy：`configs/prior.json` 与 `configs/prior-latent-mse.json` 都是确定性下一 latent MSE，文字 CE 权重为 0。原 Energy 配置保存在 `configs/prior-energy-v4.json`，原实验和检查点保留。历史用法见 [Energy V4 说明](README-energy-v4.md)。

完整实测、失败尝试与限制见 [固定 latent 实验报告](docs/fixed-latent-experiments-2026-10-29.md)。

## 当前结果

| 方法 | 结果 |
|---|---|
| 确定性 latent MSE，1000 步 | 固定 dev MSE 0.9260，利用了条件，文字仍杂乱 |
| 连续 latent 去噪，保留 step 1100 | 去噪 loss 明显下降，自由生成仍未解决 |
| 块级主干＋块内离散小头，1000 步 | 固定 dev NLL/token 3.1614，主干和正确历史的作用已验证，文字流畅性明显改善 |

块内小头先逐位置选择四个词，再经冻结 encoder 得到一个回灌 latent；decoder 只负责还原。主干仍每块推进一次，但这个方案包含离散词选择，不能称为纯连续向量回归。尚未实现或测量异步加速；生成仍有语义跳跃、重复和指代问题。

## 使用已验证模型

在项目目录、Python 3.13 环境中执行：

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/prior-block-head-verified/last.pt `
  --prompt "Once upon a time, a little girl found a strange box." `
  --max-new-tokens 64 --seed 42 --show-info
```

继续训练这个块级模型：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-block-head-verified/last.pt
```

每 100 步保存到该实验的 `last.pt`。`--resume` 恢复检查点原来的架构、目标、优化器与随机状态，不能通过替换外部配置更换目标。

## 继续纯 latent MSE 实验

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-latent-mse-check/last.pt
```

若要新建同类实验、只继承旧主干：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --config configs/prior-latent-mse.json `
  --init-prior runs/prior-energy-v4-check/last.pt `
  --run-dir runs/prior-latent-mse-new
```

旧检查点仅提供主干和 latent 归一化统计，新输出头与优化器重新开始，训练目标不含 Energy。直接运行 `train_prior.py` 则从头建立默认 MSE 实验。MSE 点预测会趋向条件均值，loss 下降不保证生成具体可解码文字。

## 文件与复查

- `configs/prior-latent-mse.json`：确定性真实 latent 目标。
- `configs/prior-denoising.json`：连续去噪 MSE，推理需要多次小头迭代。
- `configs/prior-token-head.json`：块内离散小头，主干仍按 latent 块推进。
- `evaluate_continuous_prior.py`：独立噪声的真实历史/自由回灌对照，保存全部文本。
- `diagnose_backbone_learning.py`：恢复初始主干、抹除/错配上下文的干预。
- [本轮完整实验](docs/fixed-latent-experiments-2026-10-29.md)、[此前损失诊断](docs/loss-investigation-2026-10-29.md)、[其他消融](docs/training-ablation-results-2026-10-29.md)。

所有新实验仅使用 train/dev。冻结 codec 在固定 2048 个 dev 块上的检查：token 99.43%、完整四 token 全对 97.83%。数据准备流程见 [V3 用法](docs/usage-v3.md)。

测试：

```powershell
$env:PYTHONPATH = Join-Path $PWD '.tmp/dev-tools'
& D:\Code\venv313\Scripts\python.exe -m pytest tests -q `
  --basetemp .tmp/pytest-fixed-latent-final
```

最终 87 项测试通过，包括 latent-only 无 CE 反传、主干梯度、冻结 codec、去噪双峰学习、块内因果性、实际 latent 回灌、decoder 不进入回灌环，以及中断恢复。
