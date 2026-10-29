# Latent AR：块内双向解码器 V2

当前版本将每个生成 latent 直接展开为四个 token 位置，使用 **一层块内双向 self-attention** 并行预测。Decoder 只读取当前块的 latent，不读取文本历史、原始 prompt 编码或其他块。

变更记录见 [CHANGELOG.md](CHANGELOG.md)，完整设计见 [DESIGN.md](DESIGN.md)。旧版带 cross-attention 的实现保留在 `latent_ar/legacy_model.py`，可继续加载历史检查点做生成与对照。

## 1. 直接开始训练

所有命令都在项目目录执行，统一使用 `D:\Code\venv313` 的 Python 3.13。

```powershell
& D:\Code\venv313\Scripts\python.exe train.py --config configs/train.json
```

默认使用已准备好的 **10 万篇 TinyStories**，训练 **10,000 步**，`batch 128 × 累计 2 次`，每 **100 步自动保存**。

新版目录为 `runs/train-block-v2/`，与旧版 `runs/train-b128/` 分开。这次是新架构训练，不能直接 `--resume` 旧版 AR decoder 检查点。

正常暂停按一次 `Ctrl+C`。程序完成当前参数更新、保存后退出，等到 `[保存]` 与 `[结束]` 后再关闭窗口。恢复命令：

```powershell
& D:\Code\venv313\Scripts\python.exe train.py `
  --resume runs/train-block-v2/last.pt
```

再次按 Ctrl+C 或强制关闭进程可能回退到上一个已完成的保存点。保存采用临时文件写完后替换 `last.pt`。

## 2. 配置怎么读

| 配置 | 数据范围 | 更新步数 | 输出目录 |
|---|---|---:|---|
| `configs/smoke.json` | 训练集前 1,000 篇，缩小网络 | 4 | `runs/smoke-block-v2/` |
| `configs/overfit.json` | 训练集前 1,000 篇，完整网络 | 1,000 | `runs/overfit-block-v2/` |
| `configs/train.json` | 选定的 10 万篇，完整网络 | 10,000 | `runs/train-block-v2/` |

完整网络约 **3,390 万参数**。主干仍是 8 层、宽度 512；decoder 改成 1 层、宽度 256。

| 参数 | 含义 |
|---|---|
| `batch_size: 128` | 一批 128 个故事窗口 |
| `grad_accum: 2` | 累计两批，更新一次参数；每步共 256 个窗口 |
| `max_target_tokens: 64` | 每个窗口最多监督 64 个续写 token，对应 16 个 latent |
| `save_every: 100` | 每 100 个优化步骤自动保存，结束或正常中断时额外保存 |
| `eval_batches: 2` | 正式配置每次 dev 检查 256 篇；overfit 为 1 批、128 篇 |

训练按 step 停止，有放回随机选择故事并截取窗口。10,000 步最多监督约 1.64 亿目标 token，实际取决于片段长度；其中包含重复内容。

命令行可以覆盖配置，并写入后续检查点：

```powershell
& D:\Code\venv313\Scripts\python.exe train.py `
  --config configs/train.json `
  --run-dir runs/my-block-experiment `
  --batch-size 128 `
  --grad-accum 2 `
  --save-every 100
```

## 3. 生成文本

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/train-block-v2/last.pt `
  --prompt "Once upon a time, a little girl found a strange box." `
  --max-new-tokens 64 `
  --show-info
```

V2 默认贪心解码（`--temperature 0`），优先检查 latent 能否确定具体词句。显式设置 `--temperature 0.8` 可启用随机采样，但四个位置的采样没有文本反馈，词语间可能不协调。训练长度目前最多 64 token，更长生成属于尚未验证的外推。

`--min-new-tokens 16` 可以临时屏蔽过早的 EOS，默认为 0。它只是长度约束，不代表模型质量提升。

同一个入口也能加载旧版模型，自动使用历史结构；旧版默认温度仍为 0.8：

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/train-b128/last.pt `
  --prompt "Once upon a time, a little girl found a strange box." `
  --max-new-tokens 64
```

## 4. 代码阅读顺序

| 文件 | 主要内容 |
|---|---|
| `train.py` | `main()` 总流程、`train_one_step()` 联合更新、自动保存与恢复 |
| `latent_ar/model.py` | V2 模型；先读 `LatentARModel.rollout()`，再读 `TextDecoder` |
| `latent_ar/data.py` | 把故事切成 prompt 与 target；target 仅用于损失，没有 teacher-forcing 输入 |
| `latent_ar/control.py` | 在完整更新后响应 Ctrl+C |
| `latent_ar/checkpoints.py` | 识别并严格加载新旧权重 |
| `latent_ar/legacy_model.py` | 历史 V1 模型，仅用于旧权重生成和诊断 |

V2 的关键形状：

```text
prompt token → 每四个编码成一个 512 维向量
             → latent 主干自由生成 z1…z16
             → 每个 z 展开为 [4, 256]
             → 各块独立的双向 self-attention
             → 每块 [4, 8192] 个词表分数
             → 最多 64 个输出 token
```

四个块内位置互相可见；不同块在 decoder 中完全隔离。真实 token、原始 prompt latent、其他块的 latent 均不是 decoder 输入。

## 5. 运行记录与验证

每个实验目录包含 `config.json`、`tokenizer.json`、`metrics.jsonl`、`samples.jsonl`、`last.pt`、`summary.json`。终端显示简短进度，JSONL 保留详细数值。

独立评价 dev；`--eval-batches 0` 表示全部固定 dev 窗口：

```powershell
& D:\Code\venv313\Scripts\python.exe train.py `
  --resume runs/train-block-v2/last.pt `
  --eval-only `
  --eval-batches 0
```

最终 test 需要另外指定 `--split test`，训练过程不访问它。V2 不读取真实续写历史，它的 CE/PPL 与旧版 teacher-forced AR 模型使用的条件信息不同，应连同自由生成一起比较。

`--ablation zero-latents` 将所有生成 latent 置零，并跳过 encoder 和主干；这时 decoder 没有任何 prompt 信息，只是无条件位置分布诊断。旧参数名 `no-latents` 仍可用，但不再代表旧版“保留 prompt 的小 AR decoder”基线。

## 6. 数据与依赖

官方 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 的完整 Parquet 分片已经下载到 `data/raw/`。固定选出的 10 万篇训练故事、隔离的 dev/test，以及只用训练故事拟合的 8,192 词表，位于 `data/prepared/`，V2 可直接复用。

需要重新准备时：

```powershell
& D:\Code\venv313\Scripts\python.exe download_data.py
& D:\Code\venv313\Scripts\python.exe prepare_data.py --output data/prepared-new
```

准备脚本拒绝覆盖非空目录。数据版本、种子和哈希保存在 `metadata.json`，每篇原文和行号保存在对应 JSONL。数据许可为 CDLA-Sharing-1.0。

共享环境已包含 PyTorch、tokenizers、NumPy。若缺少数据准备和测试依赖，可安装 `pyarrow`、`pytest`。`requirements.txt` 与 `requirements-lock.windows-cuda128.txt` 是最初 Python 3.12 检查环境的依赖记录；当前实际验证环境为 Python 3.13 / PyTorch 2.11.0+cu128，沿用已有核心依赖。

本项目用于整理代码的临时工具位于 `.tmp/dev-tools`。在本机执行测试：

```powershell
$env:PYTHONPATH = Join-Path $PWD '.tmp/dev-tools'
& D:\Code\venv313\Scripts\python.exe -m pytest -q
```

测试覆盖块内双向性、块间隔离、无文本输入和 prompt 旁路、主干因果性、跨 rollout 梯度、固定样本过拟合、EOS、自动保存和恢复，以及历史模型兼容性。运行结果见 [VALIDATION.md](VALIDATION.md)。

## 7. latent 对照与显存短测

原始 latent 与时间均值的生成对比支持 V1/V2，结果写入独立目录：

```powershell
& D:\Code\venv313\Scripts\python.exe compare_latent_generation.py `
  --checkpoint runs/train-block-v2/last.pt `
  --output runs/block-v2-latent-comparison
```

V2 默认贪心；如需多种子随机对比可显式传 `--temperature 0.8`。均值来自完整生成段，属于离线扰动诊断，不是无 latent 基线。

短测默认检查 V2 的 `128×2`，结果写入 `runs/batch-benchmark.json`：

```powershell
& D:\Code\venv313\Scripts\python.exe benchmark_batch.py --batch-sizes 128 --grad-accum 2
```

当前仍是同步训练与同步参考生成；有界队列和异步推理尚待实现。
