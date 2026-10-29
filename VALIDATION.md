# 第一阶段实现检查记录

本文件按时间记录不同版本的检查。2026-10-29 的 V2 / V3 结果分列在后文；各版本结果不能混用。V3 当前是分阶段 codec + 直接下一块预测。

日期：2026-10-28。以下为流程验证，不是正式生成质量或性能实验。

2026-10-29 更新：后续命令统一使用 D:\Code\venv313\Scripts\python.exe（Python 3.13.9）。下列结果仍来自原 Python 3.12 环境，不代表已在 3.13 环境重复完成验证。

## 数据与环境

- 官方数据：`roneneldan/TinyStories`，commit `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`。
- 四个训练分片和一个验证分片全部下载；总计 1,000,775,442 字节，均核对官方 SHA-256。
- 固定种子 42；选取 100,000 篇训练故事、10,995 篇 dev、10,995 篇 test。
- 训练/dev/test 含 EOS 的 token 数分别为 21,997,870 / 2,334,755 / 2,357,072。
- tokenizer 仅在选中训练集上拟合；实际词表 8,192。
- 已检查三份数据的空白归一化文本哈希互不重叠；抽查 100 篇训练故事，tokenizer 编码/解码逐字还原成功。
- Python 3.12.10、PyTorch 2.7.1+cu128、Windows、NVIDIA GeForce RTX 4060 Laptop GPU（8 GB）。环境安装在项目 `.venv`；实际依赖版本见 `requirements-lock.windows-cuda128.txt`。

## 自动化测试

自动化测试：11 项全部通过（原 Python 3.12 环境）。当前重跑命令为 `& D:\Code\venv313\Scripts\python.exe -m pytest -q`。

包括 prompt padding 不影响 rollout、未来 token/latent 不可见、主干与 decoder 的 KV cache 等价性、末端 latent 梯度传回前序步骤、三模块联合 CE 梯度、无 latent 分支、EOS 与截断处理、边界拼接，以及固定小样本过拟合。

## 真实数据端到端检查

### 缩小模型与断点续训

使用 `configs/smoke.json` 在 CPU 上训练 4 步；另一个运行在第 2 步保存后恢复到第 4 步。两次运行的最终模型所有张量逐元素相同，有效目标 token 数同为 421。

运行记录：`runs/smoke-uninterrupted/`、`runs/smoke-resume/`。

### 设计尺寸模型的 GPU 检查

使用当前统一解释器重跑的命令（原检查使用项目内 Python 3.12 环境）：

```powershell
& D:\Code\venv313\Scripts\python.exe train.py --config configs/overfit.json --run-dir runs/full-config-check --stop-after 2 --eval-batches 2
```

- 参数量：35,010,816。
- 主干：8 层、宽度 512；decoder：2 层、宽度 256。
- 最大 16 步 latent rollout，BF16，批量 2、累计 8 次。
- 完成 2 个优化步骤，共 1,484 个有效目标 token；梯度有限，dev 评价、自由生成、checkpoint 保存均完成。
- PyTorch 峰值已分配 GPU 张量显存约 719.58 MiB。这不包括全部驱动/其他应用开销，也不代表长时间训练的显存上限。
- 两步训练 CE 约 9.0280 / 9.0344；小规模 dev CE 约 9.0231。训练尚处于初始化附近，不据此判断学习效果。
- 独立 `generate.py` 成功加载 checkpoint 并生成 12 个 token；输出出现重复，符合仅训练两步的未训练状态，不是质量达标证据。

运行记录：`runs/full-config-check/`。本次没有执行完整 1,000 步或 10,000 步训练。

### 无生成 latent 分支

使用缩小配置运行 1 步无 latent 训练，保存后执行独立 dev 评价成功。该配置最大 latent 步数为 0，有效参数量 576,160；对应完整缩小模型为 687,136 参数。

运行记录：`runs/smoke-no-latents/`。

## 尚未验证

- 1,000 篇和 10 万篇训练后的生成质量、长 rollout 稳定性。
- latent 分支相对同尺寸 decoder 或普通 token AR 模型的实际收益。
- 异步执行、同步/异步一致性以及相近质量下的端到端加速。
- 跨硬件、跨依赖版本的完全可复现性；本次逐元素断点一致性检查仅覆盖本机 CPU 缩小配置。

## 2026-10-29：Python 3.13 环境的批量实测

使用 `D:\Code\venv313\Scripts\python.exe benchmark_batch.py`，PyTorch 2.11.0+cu128。完整 35,010,816 参数模型，BF16，真实训练故事截成固定 128 token prompt、64 token target，16 步 latent 展开；每种配置预热一次，测量三次更新。

| 配置 | 每次更新窗口数 | 秒/更新 | 目标 token/秒 | 峰值张量显存 | 峰值保留显存 |
|---|---:|---:|---:|---:|---:|
| batch 2，累计 8 | 16 | 3.349 | 305.8 | 722.3 MiB | 746 MiB |
| batch 16，累计 2 | 32 | 0.862 | 2,375.3 | 1,105.7 MiB | 1,134 MiB |

两种配置均完成前向、完整反向传播、梯度裁剪和优化器更新，梯度有限，无显存溢出。测量时已有另一训练进程在运行；这里只作为本机批量选择的短测依据，不是隔离条件下的正式性能对照，也不能用于比较两种批量的收敛质量。

默认 overfit/train 配置已改为 `16×2`，对应有效批量从 16 增至 32；原定优化步骤数保留，因此总采样窗口预算翻倍。dev 批次数分别改为 4 和 16，保持原来的 64/256 篇验证预算。缩小模型的 smoke 配置保持其流程检查用途。

原训练进程不会因 JSON 文件修改而自动改变批量；从旧 checkpoint 恢复时需显式传入 `--batch-size 16 --grad-accum 2`。本次没有改动正在运行的训练进程。

完整短测结果保存在 `runs/batch-benchmark.json`。

## 2026-10-29：自动保存与代码可读性

根据实际训练日志，默认 overfit/train 批量进一步调整为 `64×2`（每次更新 128 个窗口）。用户日志显示该批量连续运行，典型吞吐约 6,000–8,000 目标 token/s；这与前述固定长度短测不是同一个对照条件。dev 默认分别取 1/4 个批次，保持 64/256 篇预算。

按用户要求，两份训练配置默认每 **100 步** 自动保存；训练结束时保存，第一次 Ctrl+C 则完成当前更新、保存后退出。终端打印保存位置与步骤。旧进程不自动获得新逻辑；强制结束进程仍只能恢复最后一次完成写入的检查点。

在 Python 3.13.9、PyTorch 2.11.0+cu128 环境中运行全部测试，**15 项通过**。其中新增的四项验证：

1. 临时小模型完整训练到第 101 步，实际保存顺序为第 100、101 步。
2. 在优化器更新时发送真实 SIGINT，保存第 1 步并退出；恢复至第 3 步后，所有模型张量与连续训练逐元素相同，累计 token 与数据随机状态一致。
3. 模拟保存中途写盘失败，原检查点内容保持完整。
4. 中断处理器在退出后恢复原信号处理方式，第二次中断可强制退出。

新增测试自建临时数据，不依赖下载好的 TinyStories。测试工具安装在项目 `.tmp/dev-tools`，没有向共享 Python 环境安装新包。本次测试命令：

```powershell
$env:PYTHONPATH = Join-Path $PWD '.tmp/dev-tools'
& D:\Code\venv313\Scripts\python.exe -m pytest -q --basetemp .tmp/pytest-readable-check
```

代码整理为统一的 88 字符目标行宽，主要模块和关键张量/边界处理增加中文说明。`train.py` 拆出配置、单步训练、验证、保存等函数；终端进度使用简短可读文字，详细指标继续写入 JSONL。

## 2026-10-29：V2 块内双向 decoder

当前架构为 `latent_block_v2`：每个生成 latent 展开为四个位置，经过一层块内双向 attention 与小前馈层，再并行输出。移除了文本历史、cross-attention、prompt 旁路及 decoder 跨块缓存。

### 自动化回归

Python 3.13.9 / PyTorch 2.11.0+cu128，全部 **33 项测试通过**。

- 用恒等 V 投影、均匀注意力证明：改动第四个槽位会影响第一个槽位，排除意外使用因果 mask。
- 只扰动一个生成块，其余块的 logits 不变；从一个块反传，其他块的输入 latent 梯度为零。
- 固定生成 latent 后更换 prompt，decoder 输出不变；decoder 没有文本 embedding 输入或 cross-attention。
- 测试 prompt padding、主干因果性、晚期损失回传到早期 latent 和三个模块的联合梯度。
- 测试无文本历史条件下的小样本过拟合、非整块输出长度、块内 EOS，以及并行 logits 与贪心输出对应。
- 保留历史 V1 模型回归测试，并检查新旧权重按架构严格恢复、训练入口拒绝把 V1 当成 V2 续训。
- 自动保存、安全中断和逐元素一致的续训检查继续通过；新 checkpoint 包含 V2 架构标记。

### GPU 端到端流程

在真实准备好的 10 万篇数据上，用完整 **33,897,728 参数**模型、BF16、`128×2` 执行：

```powershell
& D:\Code\venv313\Scripts\python.exe train.py `
  --config configs/train.json `
  --run-dir runs/block-v2-integration-check `
  --stop-after 2 `
  --eval-batches 1

& D:\Code\venv313\Scripts\python.exe train.py `
  --resume runs/block-v2-integration-check/last.pt `
  --stop-after 3 `
  --eval-batches 1
```

完成三次参数更新，共 32,769 个有效目标 token，梯度有限，保存、恢复、dev 评价和自由生成均成功。恢复进程的峰值 GPU 张量显存约 4,085.74 MiB；窗口随机取样，这不是固定最大长度的显存上限测试。仅训练三步，生成文本尚无质量意义。

独立 `generate.py` 已成功加载 V2 检查点生成 7 个 token；同一入口也成功加载旧版 `runs/train-b128/last.pt`，保留历史输出。latent 均值比较脚本在新旧两种检查点上均完成短流程检查，输出目录分别为 `runs/block-v2-integration-check/latent-comparison-smoke` 与 `runs/legacy-comparison-smoke-v2-loader`。

未启动 V2 的正式 10,000 步训练。当前仍没有异步队列或正式质量与速度对照结果。


## 2026-10-29：V3 分阶段与直接 latent 自回归

### 环境和测试

- 全部使用 `D:\Code\venv313\Scripts\python.exe`，Python 3.13.9 / PyTorch 2.11.0+cu128，NVIDIA RTX 4060 Laptop GPU（8 GB）。
- 命令：`python.exe -m pytest tests -q --basetemp .tmp/pytest-direct-ar-v3-final`，47 项通过，11.12 秒。此处 `python.exe` 为上面完整路径；pytest 来自项目 `.tmp/dev-tools`。
- 新增 14 项测试：真实四 token 重建学习、两阶段拟合后自主三块完整还原、完整块计数、冻结 codec 且 CE 能给主干梯度、不能读取当前/未来目标、并行真实历史和缓存推理等价、padding 不传损失梯度、第一块预测一致、反馈自身预测 latent、确定性 latent 不受词采样影响、共享权重及统计加载、两个阶段的 Ctrl+C 中断恢复逐张量一致、重建未达标时拒绝主干训练。
- 保留原 33 项 V1/V2 结构与训练回归检查。

### Codec 短训检查

- 目录 `runs/codec-v3-integration-check/`，4,266,240 参数，batch 128×累计 2，BF16。计划仍为 10,000 步，只运行至第 1,000 步，并检查了保存恢复。
- 累计 256,000 个小块、921,142 个有效 token。第 1,000 步日常 dev 指标 token 98.80%，含短块的全对率 96.04%。这不是完整四 token 指标。
- prior 默认入口重新评价固定 2,048 个 dev 小块，得到 token **98.83%**、完整四 token 块 **95.42%**；因 token 未达 99%，正确拒绝进入，未创建正式 prior 训练目录。
- PyTorch 峰值已分配 GPU 张量显存约 149.41 MiB；短训仅说明此 codec 能有效学习，尚未满足默认全部进入条件。

### 直接 AR prior 的 GPU 流程检查

- 目录 `runs/prior-ar-v3-plumbing-check/`。为了仅验证 GPU 接线，显式使用上面的 codec 并把这次流程检查 token 阈值临时设为 98%；完整四 token 要求保持 95%。默认配置中的 99% 没有降低。
- 当前架构 `latent_codec_ar_v3`，总参数 31,800,576，可训练 27,534,336；冻结 codec 权重未变。batch 128×累计 2，先完成 2 步，随后从断点恢复至第 3 步。
- 共 32,769 个有效目标 token、8,367 个目标块。第 3 步 train latent MSE 1.2111、token CE 9.5820；dev MSE 1.2088、CE 9.5281。三步不足以评价学习趋势与条件利用。
- 恢复检查记录的 PyTorch 峰值已分配显存约 2,756.02 MiB。用的是实际抽样窗口，且只是短测，不是所有窗口都填满 128+64 token 的显存压力测试。
- `generate.py` 跑通 8 token 输出；文本仍是未训练主干的杂乱输出，不能当作生成能力结果。
- `evaluate_rollout.py` 在 2 个 dev 窗口、每个最多 8 目标 token 上跑通；第一块真实历史与自主分支 latent 距离为 0，所有自主 latent 有限，报告在 `runs/rollout-ar-v3-plumbing-check/`。该极小样本 codec 还原率 93.75%，与 2,048 块检查不是同一批数据，不能混为总体准确率。
- 原始 / 均值 latent 对照跑通，输出在 `runs/prior-ar-v3-latent-plumbing-check/`。仅验证诊断脚本可工作，不推断 latent 的语言质量贡献。

### 尚待正式实验

Codec 继续训练至默认进入检查通过，然后用默认阈值训练主干，再检查 64 token 内的真实历史和自主续写。连续点预测的平均化、回灌漂移以及多模态/异步收益都尚未得到解决或性能结论。


## 2026-10-29：V4 Energy 分布生成头

### 自动检查

- 使用统一 Python 3.13 环境执行 `-m pytest tests -q --basetemp .tmp/pytest-energy-v4-full`，**59 项通过，9.45 秒**。
- 验证非平方欧氏距离、独立候选的非对角配对估计、padding 梯度隔离、相同候选的有限梯度、至少两个候选限制、冻结 codec、Energy 梯度到主干和噪声输入层、不同种子采样、词温度不改变 latent、真实/自主首块一致、直接回灌生成 latent、检查点加载、两个阶段及 Energy 的中断恢复一致性、严格主干初始化。
- 双峰任务：相同上下文的目标为等概率 ±1 向量，训练小生成头 600 步。4,096 个生成样本中正模式占 49.80%，到最近真实模式的平均 RMS 距离约 0.0174，接近零均值区域的比例约 0.29%。这确认目标能学习多模态分布，不代表真实文本任务质量。

### GPU 短训与恢复

- 本机 RTX 4060 Laptop GPU，BF16，batch 128×累计 2，每块 8 个候选，宽度 512、2 层生成头、noise_dim 128。
- 原 `runs/codec-v3/last.pt` 在固定 2,048 dev 块上 token 99.43%、完整四 token 97.83%，按默认阈值通过，没有降低要求。
- 使用 `--init-prior runs/prior-v3/last.pt` 继承旧 step 2202 的主干，输出头和优化器重新开始。短训目录为 `runs/prior-energy-v4-check/`，原实验保留。
- 总参数 33,966,336，可训练 29,700,096。完成 100 步，自动保存后恢复到第 101 步；累计 1,146,061 个有效目标 token、292,148 个目标块。
- 第 1 步 Energy 0.7966 / spread 0.6214 / 样本 CE 8.8856；第 100 步 Energy 0.7295 / spread 0.9327 / 样本 CE 8.3491；恢复后的第 101 步 Energy 0.7285。
- step 101 的固定 dev 抽样 Energy 0.7248，spread 0.9459，样本 CE 8.3266，错配条件 Energy 优势 +0.0236。CE 不参与优化，不转换成普通 LM 的 PPL。
- 常见完整更新耗时约 0.26～0.27 秒。恢复短测的 PyTorch 峰值已分配 GPU 张量显存约 3,081.16 MiB；窗口随机长度，不是全长填满的压力测试。
- 生成、原始/均值 latent 对照与 4 个 dev 窗口的 64 token 自主回灌报告已跑通，分别保存在 `runs/energy-v4-latent-check/` 与 `runs/rollout-energy-v4-check/`。全部自主 latent 有限；短训文字仍杂乱，不能据此宣称语言质量提升。

当前只确认随机分布训练、初步优化、生成接线和保存恢复。真实语料的续写质量、不同 seed 的连贯性及纯 latent 长链稳定性，需要正式训练评价。


## 2026-10-29：Energy 后续诊断，真实语料未跑通

用户在第 1404 步安全中断，断点和原实验保留；没有替用户恢复长训。累计 15,954,841 个有效目标 token。step 1250 固定 dev 的 Energy 约 0.6769，但样本 CE 约 9.7941，自主续写仍不通顺。因此不能把前面的短训优化和自动测试当成语言建模成功。

### 同条件分布诊断

在 CPU / FP32、固定 32 个 dev 窗口（1,439 个 token、367 个有效块）上，用同一份 8 候选噪声对比：按原 seed 42 和旧主干 step 2202 重建的 Energy 初始化模型，与当时可读的 step 1000 模型。

| 指标 | 初始化 | step 1000 |
|---|---:|---:|
| Energy | 0.7943 | 0.6808 |
| 单候选 CE | 8.8514 | 9.5936 |
| 8 候选混合 token CE | 8.5248 | 6.6745 |
| 8 候选混合块 NLL | 33.5710 | 29.7230 |

这说明单候选 CE 的恶化不能直接等同于整个预测分布恶化；但真实续写仍很弱，不能用替代指标掩盖。混合指标是有限候选的有偏估计，对应 decoder softmax 诱导的分布，而非默认 argmax 文本生成的精确似然。真实历史诊断也不代表自主回灌效果。记录在 `runs/energy-v4-initial-vs-step-1000.json`；高斯与无条件真实块的辅助比较在 `runs/energy-v4-diagnostic-step-700.json`。

### 固定真实文本过拟合

从 step 1404 模型创建独立副本，使用 train 中固定 4 个窗口，每个目标最多 16 token。仅对这些窗口运行 500 次 Energy 更新，AdamW lr=3e-4、无 weight decay、BF16。第 50 次更新时，在固定生成 seed 42 下，真实历史与自主回灌 token 准确率均达到 100%，后续检查保持全对。举例：

```text
参考：other food. The crane was happy and kept eating fish from the pond. The
自主：other food. The crane was happy and kept eating fish from the pond. The
```

这排除了“梯度、codec、纯 latent 回灌完全不能学习这些具体文本”的解释；它只证明记忆能力，不能证明大语料泛化或证明问题的唯一根因。副本未保存模型/优化器，也未改写用户检查点，报告为 `runs/energy-v4-real-text-overfit.json`。

验证代码现已额外报告候选混合块 NLL，不修改实际 Energy 目标。两项新增测试覆盖概率混合顺序、块内共享候选和诊断不改变采样/梯度目标。完整测试当前为 61 项通过。


## 2026-10-29：进一步损失、反馈和结构对照

详见 [完整测试报告](docs/training-ablation-results-2026-10-29.md) 及所链接的原始 JSON。三种随机头目标各 300 步，一致性对照两组各 200 步，新文字小头 600 步并通过正式入口恢复到 601。所有模型沿用同一训练 / dev 划分，不使用 test。结构基线的小头重新初始化，不能将其收益归因于单一因素。

当前全套测试在隐藏 CUDA 的 CPU 环境、统一 Python 3.13 下执行 `-m pytest tests -q --basetemp .tmp/pytest-prior-ablation-final`，85 项通过（16.70 秒）。覆盖新增文字损失梯度与激活重算等价、完整块一致性约束、文字小头的因果性、主干梯度、实际所选代码编码与检查点复现，以及当前其他架构回归检查。GPU 的正式入口恢复、词表采样与独立 dev 窗口检查另外完成。

另外使用独立配置从旧主干初始化并完成 1 步 GPU 训练（`runs/token-head-init-check/`），确认配置入口可用；已训练的 601 步文字小头通过正式生成入口输出 64 token，完成 GPU BF16 与 CPU FP32 的 16 token 回灌报告。BF16 首块向量 RMS 差异约 0.0081，FP32 约 0.0，属于此处浮点路径差异，不能要求随机离散选择在所有近边界数值上逐位相同。原始报告在 `runs/token-head-ablation-v1/rollout-check/` 与 `rollout-check-fp32/`。这不改变文本仍有语法和语义错误的结论。
