# 16-token 块反馈蒸馏实验

本实验的目标是用小预测器近似实际文本的回灌表示：`C(z) ≈ E(D(z))`。
这里 z 是已经选定的内容 latent，不是尚未确定输出内容的上下文状态。
当前已有阶段一独立 codec 配置和阶段二条件分布训练、真实文本回灌接口。正式主干尚待训练与质量验证；批量配对数据采集及反馈蒸馏尚未实现。

## 阶段一：一起训练 encoder 和 decoder

训练路径：真实文本块（最多 16 token）→ encoder → 512 维 latent → decoder → 原文本块。
用 token CE 联合训练 E 和 D。decoder 保留一层块内双向注意力，一次输出 16 个位置的 logits，不读取真实目标 token 作为输入。
此阶段不训练语言主干，也没有从上下文猜未来的任务。

配置在 `configs/codec-16.json`：batch 128 × 梯度累计 2；每次更新抽取 256 个文本块，最多 4096 个有效 token，短尾块会减少实际数量。数据沿用 `data/prepared` 的 TinyStories 子集和 8192 词表。计划 10000 步，每 100 步保存，每 250 步验证。

在仓库根目录运行，先到 500 步检查训练速度和曲线（500 步不代表应当收敛）：

```powershell
& D:\Code\venv313\Scripts\python.exe train_codec.py `
  --config configs/codec-16.json `
  --run-dir runs/codec-16 `
  --device cuda `
  --stop-after 500
```

继续原计划到 10000 步：

```powershell
& D:\Code\venv313\Scripts\python.exe train_codec.py `
  --resume runs/codec-16/last.pt `
  --device cuda
```

单独验证：

```powershell
& D:\Code\venv313\Scripts\python.exe train_codec.py `
  --resume runs/codec-16/last.pt `
  --device cuda `
  --eval-only --split dev --eval-batches 16
```

4-token 检查点的参数形状不同，不能直接用于这里的续训。128×2 的显存适配尚未在本次实验中验证。

阶段门槛：在固定 dev 抽样上检查 token accuracy 和完整 16-token 块 accuracy，目标先取 99% 和 95%；两者分别实测，短块不计入完整块准确率。达不到门槛时先检查 codec 容量及训练曲线，不降低门槛来直接启动主干。
这些是实验门槛，不是成功保证；还需检查轻微 latent 扰动后的重建，以及主干生成分布上的可解码性。

## 阶段二：训练并验证真实回灌主干

冻结 E、D。采用仓库已有的有限混合头：主干输出 4 个 512 维候选 latent 及其选择权重。训练将 4 个候选通过冻结的 D，计算整块文字的联合负对数似然，对共享分量做精确求和。梯度穿过 D 回到主干和输出头，但不更新 E、D。这里没有 latent MSE，也不是四高斯；每个分量内部的 token 概率仍然条件独立，模型能否充分表达 16-token 联合分布需要实际验证。

生成时抽一个 latent 分量，D 使用 argmax。必须区分：训练 NLL 对应的是分量内词表概率的联合分布，argmax 生成是不同的解码策略，NLL 不能直接当成该策略的似然或生成质量分数。

从仓库根目录启动：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --config configs/prior-16-feedback.json `
  --run-dir runs/prior-16-feedback `
  --device cuda
```

配置使用 batch 128 × 累计 2，10000 步，每 100 步保存，每 250 步验证和保存固定 prompt 的续写到 `samples.jsonl`。每个微批先读取 128 个故事窗口的真实历史，然后均匀抽最多 64 个目标块计算混合 NLL，并按抽样比例加权；因此不是对 256 个窗口的所有目标 token 都计算 decoder 损失。这样控制 4 个候选解码的显存，验证时则分批评价所有目标块。训练日志显式显示这个区别。完整配置的 GPU 峰值显存尚未实测。

恢复训练：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-16-feedback/last.pt --device cuda
```

生成和对照：

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/prior-16-feedback/last.pt `
  --prompt "Once upon a time, Lily found a little box." `
  --max-new-tokens 64 --seed 42 --feedback-mode text --show-info
```

将最后的 `--feedback-mode text` 换为 `--feedback-mode raw`，可用同一随机种子比较实际文本回灌与原 latent 回灌。两者都逐块解码，用于质量对照，不是异步速度基线。默认采用检查点中的 text 模式；此实验只接受 decoder temperature=0，随机性来自候选 latent 的选择。旧混合检查点未设置反馈模式时保留原始生成路径。

模型接口 `model.generate(..., feedback_mode="text", return_trace=True)` 返回 `output_ids` 与 `records`，逐块记录原始 `z`、反馈向量、token IDs、实际长度、有效样本标记和上下文状态。后续蒸馏必须按 `active`、`lengths` 过滤矩形输出中的结束后填充；长度上限造成的短尾、EOS 最短长度屏蔽会改变 D 的映射，需过滤或将相应策略信息输入 C，不能混成相同 z 的不同监督标签。

训练使用因果右移的真实块表示，当前目标只能出现在标签侧。自由生成时必须走完整路径：

```text
历史反馈 → 主干/分布头 → z → D 的逐位置 argmax → 实际文本 y
                                                    ↓
下一块主干 ← 追加到缓存 ← E(y，实际长度)
```

第一版固定 D 为确定性解码，随机性来自上游的 z。D 必须与真实生成共享特殊 token 屏蔽、首个 EOS 截断和短尾长度规则；遇到 EOS 后不再推进无效块。
如果以后允许 D 随机采样，C 需要接收相同采样噪声，否则同一个 z 没有唯一反馈标签。

比较同一模型、相同 prompt 和随机数计划下的原 latent 回灌与真实文本回灌，检查短续写和多块自由生成。完整计时 D 与 E；真实回灌必须先有可用质量，且反馈路径具有值得优化的耗时，才进入下一阶段。

## 阶段三：蒸馏反馈预测器

冻结教师，收集其自由生成轨迹中的 `(z, E(y, length))`，按来源故事或 prompt 划分训练与验证，避免同一轨迹随机拆分泄漏。先拟合小 MLP，另设 identity (`C(z)=z`) 和线性映射对照。

训练时可以从反馈向量 MSE 起步，因为在确定性 D 下这是已选定 z 的确定映射；它与历史上下文预测随机未来 latent 的 MSE 不是同一个问题。单步误差只作为诊断，必须再测预测反馈闭环中的漂移、文本质量与实际耗时。

```text
             ┌→ 原始 z → D → 输出文本
主干输出 z ──┤
             └→ C(z) → 下一块主干
```

D 始终解码原始 z；C(z) 只用于回灌。第一版不将迟到的真实反馈替换进已经使用过的缓存。两个分支在数学上可独立执行，并不保证在单 GPU 上加速；最终计时必须包括所有文本解码完成。

论文比较至少包含原 latent 回灌、真实文本回灌、预测反馈三组。16-token 块长本身不构成创新，预测器也不能仅靠低 MSE 证明有效。

## 变更记录（2026-10-30）

- 新增独立 16-token codec 配置，默认四词配置不变。
- 编码器、解码器和数据注释改为可配置块长；训练日志与质量检查显示实际块长。
- 使用指定 venv313 在 CPU 上完成真实数据的一步训练、保存、续训到第二步及独立 dev 评估，模型参数为 7,424,256。短跑仅验证执行流程，不代表重建达标。
- 第一阶段未启动正式 GPU 训练，由用户运行并报告重建达标。
- 后续新增 `configs/prior-16-feedback.json`、逐块真实文本回灌与 trace 接口、生成 CLI 对照开关，并保持旧配置兼容；反馈预测器 C 尚未实现。
- 用用户的真实 codec 检查点完成 CPU 小批量 prior 训练、保存与恢复检查。此验证不代表主干训练收敛或生成质量合格。
- 相关模型与反馈测试共 23 项通过；9 项文件系统/中断集成用例未在本轮运行。生成 CLI 的 text/raw 模式均完成 CPU 短续写检查。
