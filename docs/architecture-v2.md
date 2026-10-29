# Latent AR + Async Decoder：块内双向解码器 V2

更新日期：2026-10-29。V2 已实现同步联合训练与同步参考生成；尚无正式质量或异步加速结论。历史 V1 设计见 [docs/architecture-v1.md](docs/architecture-v1.md)，变更见 [CHANGELOG.md](CHANGELOG.md)。

## 1. 研究目标与当前阶段

长期目标是统一 latent 主干决定输出内容和细节，轻量 decoder 将其还原为文本、图像、音频。视觉等模态的生成训练是否能改善主干的世界理解，是后续独立研究问题。

当前仍以文本验证：主干不等待解码文本回灌，自主推进 latent；下游解码能否保持质量并节省计算。输入压缩、latent 自回灌和异步执行的贡献要分别测量。

V1 使用小型文本 AR decoder，通过 cross-attention 读取 prompt 与生成 latent。V2 改为唯一内容入口是当前生成 latent 的局部 decoder，进一步把跨块依赖交给主干。

## 2. 总体结构

```text
prompt token
    ↓
Chunk Encoder：每四个 token → 一个 512 维向量
    ↓
Latent Transformer：prefill prompt，建立主干 KV cache
    ↓
自主生成：z1 → z2 → z3 → …
           │    │    │
           ▼    ▼    ▼
       各自展开成四个 256 维位置
           ↓    ↓    ↓
       一层块内双向 self-attention + 小前馈层
           ↓    ↓    ↓
       各自并行预测四个 token
```

Decoder 不接收文本历史、不直接读取 prompt latent、不访问其他块，也不维护跨块 KV cache。主干保留因果 attention 和可微 KV cache。

## 3. 配置与参数

| 模块 | 当前完整配置 | 参数量 |
|---|---|---:|
| Prompt encoder | embedding 256；四 token 拼接并追加四个有效标志；MLP 到 512；RMSNorm | 2,887,168 |
| Latent Transformer | 8 层、宽度 512、8 heads、SwiGLU 1536、RoPE、RMSNorm | 27,271,680 |
| Latent 输出头 | 512→512、RMSNorm | 262,656 |
| Block decoder | 512→4×256；四个位置 embedding；1 层双向 attention、4 heads、SwiGLU 768；词表输出 | 3,476,224 |
| 合计 | 词表 8,192 | **33,897,728** |

这些是实验起步配置，不是最优尺寸结论。每个 latent 对应最多四个 token，仍需检验其信息承载能力，不能宣称已形成可逆编码。

## 4. latent 依赖

记 prompt 为 p，输入编码为 U，主干为 F，latent 输出头为 A。

```text
U = E(p)
z1 = A(F(U) 的最后一个有效 hidden state)
z_t = A(F(U, z1, …, z_(t-1)) 的最后一个 hidden state)
```

第一步来自 prompt prefill；之后逐步回灌自身 latent。生成 16 个 latent 需要一次 prefill 和 15 次新增位置计算。主干只接收 latent，不接收真实续写或采样出的文本。

Prompt 编码与生成 latent 的分布是否兼容，由联合训练检验；维度相同本身不保证语义相同。

## 5. decoder 输入与双向 attention

每个生成 latent 独立展开：

```text
H_t = reshape(W_expand z_t, [4, 256]) + P_局部位置
logits_t = W_vocab LocalBlock(H_t)
```

LocalBlock 的四个位置全部互相可见，`attn_mask=None`、`is_causal=False`。块内 attention 的输入没有真实 target，所以双向计算不会读取未来答案。

实现将 `[batch, latent数, 4, 256]` 合并为 `[batch×latent数, 4, 256]` 做局部计算；不同块是独立的 batch 项，不存在跨块 attention。无文本 embedding 输入、无 cross-attention、无 prompt 旁路。

词表输出是 `[batch, latent数, 4, 8192]`。还原为 token 位置后，裁掉最后一个块超出预算的槽位。

## 6. 训练方案

Encoder、主干和块内 decoder 从头联合训练，唯一目标为有效续写位置的文本 CE：

```text
文本 CE → 块内 decoder → 生成 latent 链 → 主干 → prompt encoder
```

1. 按故事取窗口并切成 prompt / target，encoder 只读取 prompt。
2. 主干按 target 所需长度自由展开 latent，不使用真实后续文本编码。
3. 所有生成块可并行局部解码。
4. 对每个有效 target token 计算 CE；标签只用于损失，不作为模型输入。
5. 保留所有 latent 和主干 KV cache 的计算图，反向更新三个模块。

V2 没有 decoder teacher forcing。主干展开仍是串行的，不能通过 detach 或随意截断缓存把梯度范围改掉而不报告。

条件于固定 latent，当前位置词表概率不依赖已选出的其他 token。四个隐藏位置虽联合计算，独立随机采样仍可能产生不协调的词句；因此第一轮默认贪心解码。必要的随机 latent 建模留作后续实验，当前不引入 VAE、latent MSE 或 codec 预训练。

## 7. 边界、EOS 和停止

Prompt 从自身起点每四个 token 分块，最后不足四个的位置置零并显式标记有效槽位，不用 target 补齐。输出从 prompt 边界重新分块。

完整故事自然结束处有 EOS。窗口截断不伪造 EOS；batch padding 的标签为 -100，不计算 CE。生成时一个块内任意位置出现 EOS，就在该位置停止，剩余位置丢弃。最多生成 token 数始终生效；`min_new_tokens` 仅作为显式诊断选项。

PAD/BOS 不参与普通生成。BOS 保留为数据词表特殊编号，V2 不再把它作为 decoder 输入。

## 8. 同步与未来异步执行

训练先同步执行并保持可微。当前生成参考实现先计算所需 latent，再并行计算块 logits，随后选择输出 token；这不代表已实现异步，也不能据此声称低首 token 延迟。

未来可用主干作 producer、局部 decoder 作 consumer，中间加有界队列。每个块只有自身 z_t 依赖，不再需要 decoder 文本历史或跨块缓存。EOS、背压和取消作为执行控制，不进行文本语义回灌。单卡能否真正重叠、是否加速，需要实际计时。

## 9. 数据与训练预算

沿用已下载的官方 TinyStories Parquet 版本。官方训练集约 212 万篇，固定 seed 42 抽取 10 万篇，约 2,200 万 token。官方验证故事去重后分为 dev/test，各 10,995 篇；只用抽取的训练故事拟合 8,192 词表。

- smoke：缩小模型，4 步流程检查。
- overfit：训练子集前 1,000 篇，1,000 步。
- train：10 万篇，10,000 步。
- 完整配置默认 batch 128、梯度累计 2，有效批量 256。
- prompt 最长 128 token，target 最长 64 token，最多 16 个 latent。
- AdamW、学习率 3e-4、BF16；每 100 步自动保存。

窗口有放回采样，训练按更新步数停止。1 万步最多监督约 1.64 亿目标 token，实际更少，包含重复内容。不能把累计 token 当成同等数量的独立数据。

## 10. 对照与评价

需区分：实现是否正确、主干是否被利用、主干是否带来质量收益、端到端是否更快。

- 观察验证 CE、自由生成、人物和事件一致性、EOS 与重复，以及长度增长时的退化。
- 原始 latent 与时间均值、扰动或置零的对比仅作诊断。分布变化与随机采样可能放大差异，不能直接量化质量收益。
- V2 全零 latent 后没有任何 prompt 信息，不是 V1 的“保留 prompt、移除生成分支”的条件语言模型基线。
- 与 V1、普通 token AR 模型比较时，记录参数量、训练量和计算预算。V2 的概率不以真实输出历史为条件，CE/PPL 应解释其不同条件信息。
- 未来同步/异步需保持相同的块输出、随机数和停止规则，再测首 token 延迟、吞吐、端到端耗时与显存。

## 11. 检查点与复现

V2 保存 `format_version=2`、`architecture=latent_block_v2` 和 `decoder_type=block_parallel`，包含优化器、混合精度状态、随机状态、数据版本与 token 计数。临时文件写完后替换 `last.pt`；第一次 Ctrl+C 在完整参数更新后保存退出。

V1 无架构标记的历史权重仍通过 `generate.py` 和 `compare_latent_generation.py` 自动加载旧实现。V1 不能直接作为 V2 完整续训断点，新训练使用 `*-block-v2` 目录，避免混合不同架构的结果。

## 12. 后续路线

先完成文本模型的质量与依赖检验，再实现异步队列与速度对照。当前自动化与 GPU 短测只验证代码流程，不宣称新结构已经优于 V1。

多模态阶段再扩展轻量图像、音频还原器，并训练主干生成对应细节充分的表示。主干负责内容，decoder 负责还原是职责目标；小 decoder 和确定性输出并不自动证明所有知识已集中到主干，也不能以极少 latent 无损表达任意复杂信号。

视觉生成训练是否改善主干的世界理解，要通过共享场景的生成、理解、状态预测和语言回答检验。静态画面逼真不能代替物理或因果理解的证据。相关文献与 V1 阶段的原始讨论保留在历史设计中。
