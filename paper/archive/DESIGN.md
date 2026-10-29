# Latent AR：Energy 分布头 V4

更新日期：2026-10-29。当前实现是固定文本 codec、因果 latent 主干和单步随机生成头，使用 Energy loss 训练下一块连续 latent 分布。V3 确定性预测设计已归档到 [architecture-v3.md](docs/architecture-v3.md)。

## 1. 内容如何选择

```text
真实前序块 → 冻结 encoder → 历史 latent → 因果 Transformer → h
                                                              │
随机输入 ε ~ N(0,I) ───────────────────────────────────────────┤
                                                              ↓
                                                   两层残差 MLP 生成头
                                                              ↓
                                                   标准化候选 latent
                                                              ↓
                                                       撤销标准化
                                                      ↙          ↘
                                             回灌主干             冻结 decoder
                                                                 ↓
                                                      一层块内双向 attention
                                                                 ↓
                                                       四个 token 的词表分数
```

同一个上下文 h 和不同 ε 定义不同候选，生成头的采样过程隐式定义 `p(z_next | history)`。随机性在选择具体 latent 时引入；decoder 只读取选定的单个 z。主干与生成头一起训练。

## 2. 保留的 codec 与因果边界

Codec 保留 V3 的真实四 token 重建结构，embedding / 词表权重共享，全部冻结并保持 eval。Decoder 将一个 512 维 latent 展开成四个 256 维槽位；槽位内双向 attention，块间在 batch 维隔离。它不读取文本历史、prompt 旁路或其他块。

训练对真实后续块编码出 z1…zT，主干输入 `[U,z1,…,z_(T-1)]`，目标依次为 z1…zT。因果 mask 阻止预测读取当前和未来目标，所有训练位置可一次并行计算；padding 不参与注意力条件或损失。具体逻辑位置与缓存规则继承 V3，并保留回归测试。

逐维 μ、σ 只从 4,096 个训练块估计，σ 下限为 0.1，之后冻结：`n(z)=(z-μ)/σ`。生成头输出标准化坐标，解码与反馈前用 `z=n(z)×σ+μ` 撤销标准化。

## 3. 单步随机生成头

默认 `noise_dim=128`、内部宽度 512、2 个残差块。噪声先线性投影，各残差块将主干上下文投影加入当前状态，经 RMSNorm / SwiGLU 后残差更新，最后归一化并投影到 512 维输出。

训练每个上下文配 N=8 个独立随机输入；上下文在候选维广播，只重复生成头，不重复主干。每个候选一次前向完成，没有扩散时间步或多轮推理优化。

| 模块 | 参数量 | 阶段二是否更新 |
|---|---:|---|
| Codec | 4,266,240 | 否 |
| 因果 Transformer | 27,271,680 | 是 |
| Energy 生成头 | 2,428,416 | 是 |
| 合计 | **33,966,336** | **29,700,096 可训练** |

## 4. Energy loss

对一个有效块，目标为冻结 encoder 给出的标准化真实向量 y，候选为独立噪声生成的 x1…xN：

```text
d(a,b) = ||a-b||₂ / sqrt(latent_dim)
attraction = (1/N) Σ_i d(x_i,y)
spread = [1/(N(N-1))] Σ_(i≠j) d(x_i,x_j)
L = attraction - 0.5 × spread
```

这是 Energy score 对应损失的样本估计。使用欧氏距离，不是平方距离；配对项排除同一个候选自身，不能用 N² 作为分母。`1/sqrt(dim)` 是固定正比例缩放，不改变优化目标的分布性质。

先排除 padding，再用 FP32 计算范数与候选距离。整个累计 batch 按有效块总数平均，不把长度不同的 microbatch 等权平均。梯度通过候选回到随机生成头和主干，真实编码过程没有梯度。

**V4 主干只优化 Energy。** 文字 CE 只对第一个候选作 no-grad 诊断；不让所有候选分别用 CE 追同一份参考文本。报告里的 CE 因此不是当前训练目标，也不是隐式联合分布的完整似然。没有 latent MSE 项。

Energy、attraction 和 spread 必须一起解释；降低 Energy 不是单独放大候选距离。它在当前固定 codec 下评价连续表示分布，不保证生成文本自然，也不能跨不同表示空间直接比较数值。

## 5. 推理与纯 latent 反馈

推理每个位置只采样一个 ε，通过生成头得到 z，并直接作为下一步 Transformer 的输入。主干不等待文本解码或重新编码。Decoder 可以在拿到 z 后独立还原四个 token。

参考代码先生成预算内 latent，再并行局部解码，仍是同步实现，尚无异步队列或加速结论。生成中屏蔽 PAD/BOS，首个 EOS 后停止该行输出，支持显式最短生成长度。当前训练目标长度最多 64 token。

`--seed` 控制随机 latent。`--temperature 0` 只表示 decoder 贪心选词，latent 仍然随机。latent 与选词使用两个派生随机数流，改变选词温度不改变同一种子下生成的 latent。

阶段三诊断给真实历史、自主历史两条路径提供相同的逐位置噪声，第一块应一致，之后检查偏移、有限值、重复和短续写。向量距离只是定位工具，不是语义质量分数。

## 6. 初始化、断点与预算

默认配置 `configs/prior.json`：batch 128×累计 2，每窗口最多 128 prompt / 64 目标 token，10,000 步，500 步 warmup，每 100 步保存、250 步验证。每步仍是 256 个故事窗口，8 个候选是每块额外的小头计算。

可以直接复用 `runs/codec-v3/last.pt`。主干初始化仍先检查 2,048 个 dev 重建块，默认要求 token ≥99%、完整四 token 全对率 ≥95%。

`--init-prior` 显式复制已训练旧 prior 的 backbone 和归一化统计，要求数据、codec 权重完全相同，主干参数形状严格匹配。新生成头、优化器、数据 RNG、学习率及步数重新开始；保存源路径、step 与 SHA-256。`--resume` 则恢复同一架构的完整状态，不能与初始化参数同时使用。

训练噪声使用可保存的 torch/CUDA RNG；验证和固定样例使用独立局部 RNG，不干扰训练随机轨迹。正确/错配条件用相同随机候选来比较 Energy。CPU 的安全中断/恢复已验证最终模型逐张量一致，GPU 也已完成真实保存恢复。

新架构标记 `latent_codec_energy_v4`。V1/V2/V3 和早期 flow 检查点继续严格按历史类型加载；旧 V3 续训保留其原 MSE+CE，不静默替换目标。历史确定性配置在 `configs/prior-ar-v3.json`。

## 7. 论文依据与实现边界

分布训练依据 [CALM §3.3](https://arxiv.org/html/2510.27688v1) 和 [EAR](https://proceedings.mlr.press/v267/shao25b.html) 的随机生成头 / Energy score 思路。本实现不是完整 CALM 复现：保留当前确定性 codec、每块一个固定编码目标，采用纯 latent 回灌；CALM 原文采用变分 codec、多 posterior 目标采样和离散文本反馈。

本机 codec 对随机扰动已有较高鲁棒性，但这不证明表示容易建模，也不代替对生成误差的检查。多模态扩展、长链稳定性及同预算语言质量仍待后续实验。论文对照和本地噪声测量见 [研究笔记](docs/latent-generative-models-review.md)，实施验证见 [VALIDATION.md](VALIDATION.md)。
