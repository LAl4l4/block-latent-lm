# Latent AR：分阶段训练 V3

当前方案是：**先让 encoder / decoder 准确还原四个 token，再冻结它们，用普通因果 Transformer 直接预测下一块 latent。** 每个 latent 仍由一层块内双向 decoder 翻译成四个 token；没有 cross-attention 和文本回灌。

结构说明见 [DESIGN.md](DESIGN.md)，变更见 [CHANGELOG.md](CHANGELOG.md)，实际检查结果见 [VALIDATION.md](VALIDATION.md)。这仍是文本研究原型，异步调度与图像、音频尚未实现。

## 1. 第一阶段：先学会准确还原

在项目目录执行；所有命令统一使用已有的 Python 3.13 环境：

```powershell
& D:\Code\venv313\Scripts\python.exe train_codec.py
```

对应 `configs/codec.json`，写入 `runs/codec-v3/`。只训练 encoder / decoder，不训练语言主干。训练样本是故事中的真实四 token 小块，另外包含短尾块；输入小块本身就是还原目标。

看两项验证指标：

- `token`：单个 token 的还原准确率。
- `四 token 块`：四个位置全部还原正确的比例，只统计完整四 token 块，短块不会抬高这个指标。

进入第二阶段前默认要求 **token ≥99%，完整四 token 块 ≥95%**。主干训练入口会重新检查 2,048 个固定 dev 块；不满足时先继续训练 codec。阈值是当前实验的进入条件，不代表对全部输入完全可逆。

默认 `batch 128 × 累计 2`，每步看到 256 个小块，最多 1,024 个有效 token。配置为 10,000 步，每 100 步保存、每 250 步验证。阶段一与阶段二的一步，处理的文本量不同。

## 2. 第二阶段：主干直接预测下一块

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --codec-checkpoint runs/codec-v3/last.pt
```

对应 `configs/prior.json`，写入 `runs/prior-v3/`。先检查 codec 质量，再将 encoder / decoder 全部冻结。训练时：

```text
真实 prompt → 冻结 encoder → U
真实续写块  → 冻结 encoder → z1, z2, z3…

U              → 因果 Transformer → 线性层 → 预测 z1
U, 真实 z1      → 因果 Transformer → 线性层 → 预测 z2
U, 真实 z1,z2   → 因果 Transformer → 线性层 → 预测 z3
```

因果 mask 允许一次并行计算这些训练位置，当前块和未来块不会泄漏给自身预测。损失是：

```text
loss = 1.0 × 标准化 latent MSE + 1.0 × 冻结 decoder 还原文字的 CE
```

MSE 按有效块平均；CE 按有效 token 平均，padding 不参与。两个权重可在 `configs/prior.json` 的 `training` 中调整。冻结 decoder 的参数，但保留它对输入 latent 的梯度，让文字 CE 能监督主干。

默认仍是 **batch 128 × 累计 2 = 每步 256 个故事窗口**，每个窗口最多 128 个 prompt token、64 个目标 token。训练 10,000 步、每 100 步保存、每 250 步验证。数据有放回随机取窗，包含重复；目标 token 总量最多 1.6384 亿，实际累计数记录在日志 `tokens_seen`。

完整 codec 有 **4,266,240** 参数；主干及线性头有 **27,534,336** 可训练参数。阶段二合计 **31,800,576** 参数，其中 codec 冻结。

## 3. 保存、暂停与恢复

两个训练阶段都每 100 步自动保存 `last.pt`。结束或按一次 `Ctrl+C` 后，会完成当前参数更新、保存再退出。再次中断或强制关闭可能回退到上一个保存点。

```powershell
# 第一阶段恢复
& D:\Code\venv313\Scripts\python.exe train_codec.py `
  --resume runs/codec-v3/last.pt

# 第二阶段恢复；断点已经包含 codec，无需再次提供其路径
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-v3/last.pt
```

恢复读取断点中的配置、优化器、计数和随机状态。可以显式覆盖 `--batch-size`、`--grad-accum`、`--save-every`、`--max-steps`。改变 batch 或训练总步数会改变后续训练轨迹和学习率计划。

已有的 `runs/codec-v3-integration-check/last.pt` 是 1,000 步 codec 短训检查点，可以继续训练：

```powershell
& D:\Code\venv313\Scripts\python.exe train_codec.py `
  --resume runs/codec-v3-integration-check/last.pt
```

它在阶段入口检查中为 token 98.83%、完整四 token 块 95.42%，**目前未满足默认 token 要求**。继续这个目录后，第二阶段应显式传这个目录的检查点路径。

旧 V1/V2 的联合训练检查点不能直接当成新 codec / prior 续训。旧版训练入口与用法保留，见 [V2 用法](docs/usage-v2.md)。

## 4. 第三阶段：检查自主回灌

```powershell
& D:\Code\venv313\Scripts\python.exe evaluate_rollout.py `
  --checkpoint runs/prior-v3/last.pt
```

输出 `runs/rollout-check-v3/rollout.md` 与 `results.json`。比较相同 prompt 下两条分支：一条读取真实的前序 latent，仅用于诊断；另一条只回灌自身预测 latent，才是实际推理方式。同时检查真实 latent 的 codec 还原率、逐步数值、两分支漂移和短文本重复。

普通续写：

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/prior-v3/last.pt `
  --prompt "Once upon a time, a little girl found a strange box." `
  --max-new-tokens 64 `
  --show-info
```

主干直接预测确定的 latent，不使用 flow matching。默认贪心选词；`--temperature` 只改变 decoder 输出的词采样，不改变主干 latent。由于采样出的词不回灌，先用贪心检查内容能否连贯。64 token 以外的生成尚未验证。

这个训练方案减少了早期长链自由回灌的优化困难，**不能保证消除连续 MSE 的平均化或自主回灌漂移**。需要看真实的主干训练结果和第三阶段输出，不能仅凭 codec 重建成功判断续写成功。

原始 latent / 时间均值对照继续可用：

```powershell
& D:\Code\venv313\Scripts\python.exe compare_latent_generation.py `
  --checkpoint runs/prior-v3/last.pt `
  --output runs/prior-v3-latent-comparison
```

## 5. 日志和独立验证

每个训练目录包含 `config.json`、`tokenizer.json`、`metrics.jsonl`、`samples.jsonl`、`last.pt`、`summary.json`。codec 日志显示重建 CE / 准确率；prior 日志分别显示 latent MSE / token CE，验证还记录错配历史后的 CE。后者用于检查是否使用条件，不是生成质量分数。

独立验证默认抽取配置指定批数，不是完整 dev：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-v3/last.pt --eval-only --eval-batches 16
```

`train_codec.py` 同样支持该用法。`--eval-batches` 必须为正数。最终 test 需要显式 `--eval-only --split test`；训练过程不访问 test。prior 的 CE 是给定真实块历史时的并行块解码损失，不报告成普通逐 token 自回归模型的 PPL。

## 6. 数据与代码

复用已准备的 100,000 篇 TinyStories（含 EOS 共 21,997,870 token），dev/test 各 10,995 篇；8,192 词表仅使用选中训练故事拟合。官方完整 Parquet 文件在 `data/raw/`，选中语料、词表和哈希在 `data/prepared/`，来源及下载校验见 `data/raw/source.json` 和准备元数据。

| 文件 | 内容 |
|---|---|
| `train_codec.py` / `train_prior.py` | 两阶段训练入口 |
| `latent_ar/codec.py` / `codec_data.py` | 小块重建与采样 |
| `latent_ar/staged.py` | 真实历史训练、线性 latent 预测、自主回灌 |
| `latent_ar/staged_training.py` | 两种损失、冻结、质量检查、日志与保存恢复 |
| `latent_ar/model.py` | 共用 Transformer、块内双向 decoder；同时保留 V2 模型 |
| `evaluate_rollout.py` | 真实历史 / 自主回灌的短文本对照 |
| `latent_ar/checkpoints.py` | 按架构严格加载历史权重 |

本机测试使用同一 Python 3.13 环境；临时 pytest 工具已位于 `.tmp/dev-tools`：

```powershell
$env:PYTHONPATH = Join-Path $PWD '.tmp/dev-tools'
& D:\Code\venv313\Scripts\python.exe -m pytest tests -q `
  --basetemp .tmp/pytest-manual-check
```

`--basetemp` 请指向专用临时目录，pytest 会清空该目录。当前 47 项测试通过，包含冻结 decoder 梯度、因果边界、padding、自主回灌、双向块隔离、重建学习、自动保存和中断恢复。
