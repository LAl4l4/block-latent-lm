# Changelog

## 2026-10-29 — 同起点损失对照与文字小输出头基线

- 新增 `experiment_prior_losses.py`，在相同起点、故事窗口、候选随机输入下比较 Energy、每候选 CE 和整块候选概率混合 NLL；保存完整断点、原始样例及辅助语言评分。
- 为文字损失通过冻结 decoder 传梯度，使用激活重算控制八候选显存，并验证其梯度等同于直接计算。新增可选训练一致性约束，默认关闭；权重 1 的实测没有改善文字，不推荐。
- 额外诊断解码/重编码反馈，确认反馈表示的差异会影响实际续写；该诊断等待 decoder，不作为异步方案。
- 新增 `latent_ar/token_head.py` 与独立 `configs/prior-token-head.json`：主模型内的小 GRU 头先完成四 token 的联合选择，再由冻结 encoder 编成实际 latent，直接回灌；输出 decoder 保持双向还原。
- 新架构 `latent_codec_token_head_v6` 支持正式训练、保存恢复、生成和 latent 对照；默认 Energy 配置不迁移。该基线含离散内容采样，不宣称纯连续或多模态问题已解决。
- 600 步文字基线 dev CE 3.454，辅助生成 CE 3.106，重复 token 三元组约 1.04%；另验证新窗口的上下文依赖与小样本内容还原。完整数据与限制见测试报告。

## 2026-10-29 — V4 后续诊断：真实语料尚未跑通

- 验证新增候选概率先混合、再取负对数的 token / block 指标，整块位置共享同一 latent 候选；仅用于验证，不改变 Energy 训练目标，也不隐藏单候选 CE。
- 明确有限候选的混合 NLL 估计有偏，且对应 softmax decoder，不能当成默认贪心生成的精确似然或实际语言质量。
- 新增“概率先平均”和“整块共享候选”检查，确认诊断不改变 Energy 或采样。完整测试 61 项通过。
- 记录全语料 step 1404 仍未得到可用文本；另用副本在 4 个真实训练窗口上做隔离过拟合，50 次更新后固定种子的 16 token 续写全对。这只定位了链路可训练，不能证明泛化。

## 2026-10-29 — V4：单步随机生成头与 Energy loss

- 将默认 prior 从确定性线性回归改为 `h + 独立随机输入 → 两层残差 MLP → latent`。主干按真实历史一次并行计算，每块用 8 个候选训练；推理每块只采样一个向量。
- 新增 `latent_ar/energy.py`，按有效块计算标准 Energy loss：平均目标欧氏距离减去一半不同候选之间的平均距离。距离按 sqrt(dim) 缩放、FP32 计算、排除自配对和 padding。
- V4 主干只优化 Energy；原 latent MSE 移除，文字 CE 降为第一个候选的 no-grad 诊断。继续冻结 codec，decoder 保持一层块内双向注意力与块间隔离。
- 保留纯 latent 回灌；latent 与词采样随机数流分开，同种子可复查，温度 0 只控制贪心选词。
- 默认配置改为 `configs/prior.json` / `runs/prior-energy-v4`，仍为 batch 128×累计 2、10,000 步、每 100 步自动保存。总参数 33,966,336，可训练 29,700,096。
- 新增 `--init-prior`：检查数据、相同 codec 和结构后继承旧主干及标准化统计；新输出头和优化器重新训练。`--resume` 保持断点原架构和目标，不作静默迁移。
- 新架构标记 `latent_codec_energy_v4`；生成、均值 latent 对照和固定噪声的回灌报告支持 V4。历史 V3 配置及设计分别保留为 `configs/prior-ar-v3.json`、`docs/architecture-v3.md`。
- 59 项测试通过，新增分布目标公式、双峰学习、随机性、噪声路径梯度、相同噪声首块一致性、Energy 安全中断恢复及严格主干初始化检查。
- GPU 使用现有 codec、继承旧 step 2202 主干，完成 100 步短训并恢复到 101。短训 Energy 从 0.7966 降到 0.7285；只确认实现与初步优化，尚无正式文本质量结论。

## 2026-10-29 — V3：先重建，再直接预测下一块 latent

### 架构与训练

- 将默认训练改为 `train_codec.py` → `train_prior.py` → `evaluate_rollout.py`。V2 的从零联合自由展开训练入口保留为历史实验。
- 第一阶段训练真实四 token 小块的 encoder / decoder 重建；decoder 保持一层块内双向 attention，块间隔离，无文本输入或 cross-attention。
- Codec 共享输入 embedding / 输出词表权重，共 4,266,240 参数。增加短尾块采样与完整四 token 全对率，避免短块抬高重建指标。
- 第二阶段冻结 codec，用普通因果 Transformer 读取真实前序 latent，单次线性预测下一块 latent；训练位置通过因果 mask 并行计算。
- 使用标准化 latent MSE + 冻结 decoder 的文字 CE，两个权重默认各 1。CE 保留对预测 latent 的梯度，codec 参数不更新；归一化统计仅来自 train。
- 自主推理只回灌预测 latent，不回灌解码文字。默认贪心，不使用 flow matching。主干及线性头 27,534,336 可训练参数，总计 31,800,576 参数。
- 默认两个阶段各 10,000 步，batch 128×累计 2，每 100 步自动保存，支持 Ctrl+C 完整更新后保存与续训。

### 质量检查与兼容

- 主干初始化重新检查 2,048 个固定 dev 小块，默认要求 token ≥99%、完整四 token 块 ≥95%；未达标时在创建 prior 训练目录前退出。
- prior 断点内嵌冻结 codec、其来源哈希、质量检查和 latent 统计，不依赖源文件继续存在。
- 阶段三比较真实历史与自主回灌，记录逐步数值和短文本；真实历史结果明确标记为诊断，不作为生成质量结论。
- 新架构标记 `block_codec_v1` / `latent_codec_ar_v3`；V1/V2 检查点继续严格按原架构加载。早期 flow 实验保存在 `legacy_flow_prior.py`，仅兼容历史检查点，不参与默认训练。
- README / DESIGN 改为当前流程；旧 V2 设计及使用说明归档到 `docs/architecture-v2.md`、`docs/usage-v2.md`。

### 验证范围

- 47 项自动测试通过：新增重建学习、两阶段拟合后的自主三块完整还原、冻结与 CE 传梯度、真实历史因果性、并行/缓存等价、padding、自主回灌和两个阶段的中断恢复一致性检查。
- GPU 以 128×2 完成 codec 1,000 步短训检查和直接 AR prior 两步训练、恢复至第三步；生成、回灌报告、latent 均值对照均跑通。
- Codec 短训检查点在默认进入检查中 token 98.83%、完整四 token 块 95.42%，尚未达到 99% token 要求。prior 三步流程检查显式临时使用 98% token 阈值，不能作为正式质量结果或推荐训练设置。
- 确定性 MSE 平均化与真实历史/自主回灌的分布差异仍需后续验证；当前没有主干生成质量或异步加速结论。

## 2026-10-29 — V2：latent 唯一输入与块内双向解码

### 架构

- 删除当前 decoder 的 cross-attention、文本 token embedding 输入、文本自回归回路和跨块 KV cache。
- 删除 decoder 直接读取原始 prompt latent 的旁路。全部 prompt 与跨块信息必须通过主干生成的当前 `z_t` 传入。
- 每个 512 维 latent 线性展开为四个 256 维位置，加局部位置 embedding，经过一层双向 self-attention、小型 SwiGLU 和共享词表输出层，并行预测四个 token。
- **块内双向**：四个位置互相可见，无因果 mask。**块间隔离**：将故事与块合并到 batch 维，decoder 不访问其他块。
- 保留主干的因果 attention、RoPE、自由 latent 回灌和可微 KV cache。
- 完整模型从 35,010,816 参数改为 **33,897,728 参数**，decoder 从两层改为一层。

### 训练与生成

- V2 模型输入仅包含 prompt、有效长度与期望输出长度；移除 `decoder_input` 和 decoder teacher forcing。真实 target 只用于 CE。
- 新配置默认 batch 128、累计 2 次、每 100 步自动保存，结束与安全中断时额外保存。
- 新目录分别为 `runs/train-block-v2`、`runs/overfit-block-v2`、`runs/smoke-block-v2`。
- V2 默认贪心解码。显式温度大于零时，采样也不向 decoder 或主干反馈 token；块内首个 EOS 后停止输出。
- `zero-latents` 为全零 latent 诊断。兼容 `no-latents` 参数名，但新版没有保留 prompt 的 decoder 旁路，不能作为旧版小 AR decoder 基线解释。

### 检查点与历史兼容

- 新检查点保存 `format_version=2`、`architecture=latent_block_v2`、`decoder_type=block_parallel`。
- 旧模型实现在 `latent_ar/legacy_model.py`；`generate.py` 与 latent 均值比较脚本按检查点自动选择新旧模型。旧模型默认温度保持 0.8。
- 旧版 AR 检查点不能直接恢复为 V2。训练入口在写实验目录之前明确拒绝该操作，避免静默丢弃或错载权重。
- 历史检查点与数据不迁移、不覆盖；V1 设计归档到 `docs/architecture-v1.md`。

### 文档与验证

- 重写当前 README / DESIGN，说明双向局部结构、训练条件变化、运行命令及概率分解的限制。
- 增加双向性、块隔离、无文本输入和 prompt 旁路、跨步梯度、并行生成与 EOS 测试；保留旧模型回归测试。
- 验证新旧权重按架构严格加载，继续检查自动保存、安全中断和续训一致性。
- 完整模型在本机 GPU 以 `128×2` 完成两步训练，并恢复到第三步。这里只确认实现可运行，未开展正式质量或异步性能比较；详细记录见 `VALIDATION.md`。

## 2026-10-29 — V1 运行改进

- 统一使用 `D:\Code\venv313`（Python 3.13）。
- 将批量逐步实测到 `128×2`；增加每 100 步自动保存、Ctrl+C 安全退出及原子替换检查点。
- 增加中文注释、长行拆分和可读终端进度，保留 JSONL 指标。
- 增加 EOS 诊断、可选最短生成长度，以及原始 latent / 时间均值的成对生成脚本。

## 2026-10-28 — V1 文本原型

- 实现 prompt 分块 encoder、自由 latent 主干和带文本历史 / cross-attention 的小型 AR decoder。
- 实现文本 CE 联合训练、同步生成、断点续训及诊断对照。
- 下载官方 TinyStories，固定抽取 10 万篇，准备隔离的 dev/test 和训练集专用 tokenizer。
- 记录“文本原型先行，后续扩展多模态还原器”的研究路线。
