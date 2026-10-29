# 连续 latent 生成模型：与当前实现的对照

查阅日期：2026-10-29。本文区分论文方法、本项目实测和下一步建议；查阅时尚未改训练；后续已接入 Energy 变体（见 [V4 设计](../DESIGN.md)），CE-only 实验没有启动。

## 1. 核心区别

当前主干是 `z_next = A(F(z_history))`，输出确定的连续向量，用 latent MSE 和解码 token CE 监督。文本 decoder 的四个位置分别给出词概率，但主干没有用于选择整个具体块的随机机制。

下列论文的共同方向是学习 `p(z_next | history)`。因果 Transformer、真实历史 teacher forcing、KV cache 都可以保留；变化集中在表示空间与输出分布。把 MSE 去掉可以做消融，但不会自动给主干添加一个连续块的条件分布。

## 2. 论文采用的解法

| 工作 | 输出端与训练目标 | 与本项目目标的关系 |
|---|---|---|
| [CALM](https://arxiv.org/html/2510.27688v1)（2025） | 随机输入的单步 MLP 生成头，Energy loss；codec 使用变分正则与 dropout | 四 token 压成一个向量、轻量还原器，非常接近；但原文最终回灌解码后的文本，不能直接证明纯 latent 回灌有效。重点见 §2、§3.3。 |
| [GIVT](https://arxiv.org/html/2312.02116)（2023/2024） | 输出混合高斯的均值、方差、混合权重，用 NLL 训练，再采样连续 latent | 无迭代去噪，直接连续回灌；实验主要是图像等任务，不能据此宣称已验证我们的文本 codec。见 §3.2、附录 C.1。 |
| [EAR](https://proceedings.mlr.press/v267/shao25b.html)（ICML 2025） | 用严格适当评分规则训练连续视觉自回归模型，重点是 Energy score | 是 CALM 生成头所沿用的框架来源，说明显式似然并非训练生成分布的唯一选择。 |
| [Large Concept Models](https://arxiv.org/html/2412.08821v1)（2024） | 比较 MSE 基线、条件扩散、量化 latent 后的 CE | 明确处理“相同前缀有多个合理下一句”；也对带噪表示训练更鲁棒的 decoder。粒度是句子，decoder 职责比我们大。见 §2.3、§2.5.2。 |
| [MAR / Diffusion Loss](https://arxiv.org/html/2406.11838v1)（2024） | Transformer 状态作为条件，小型去噪头生成连续 token | 主干的上下文计算与连续输出分布分开；推理增加小头迭代，不等于每轮重跑大主干。附录 D.2 的纯 L2 图像消融效果很差，这不是文本任务的直接结论。 |
| [LatentLM](https://arxiv.org/html/2412.08635)（2024 初版） | 因果主干 + next-token diffusion / flow matching；连续向量直接供下一步使用 | 接近未来图像、音频共享主干目标；该论文的文本仍用离散 token + softmax，不是所有文本也压成四 token latent。见 §2.1–2.3。 |
| [AURORA-LM](https://arxiv.org/html/2608.02602v1)（2026） | 冻结文本 codec 后，块因果 flow matching 生成 latent，完成块留作后续条件 | 明确建模纯连续文本表示分布；使用噪声日程、输入瓶颈及轨迹一致性。其 decoder 可读取对应 latent 前缀，不等于我们的单块独立 decoder。见 §3.2–3.5。 |
| [BLT](https://arxiv.org/html/2412.09871v1)（2024） | 全局 patch 主干 + 局部自回归 byte decoder，按 byte 的 CE 训练 | 说明 latent 主干可以配 CE，但局部 decoder 读取已生成 bytes 和跨注意力，承担内容选择；不满足我们要求的独立块翻译方式。见 §3.3。 |

CALM 的[官方实现](https://github.com/shaochenze/calm)提供两阶段训练及 Energy / diffusion / flow 版本。其训练语料、模型和预算远大于本项目，不能照搬性能数字。

## 3. 不用多步扩散的两个选择

### 单步随机生成头 + Energy loss

以主干状态 h 和不同随机输入 ε，生成候选 latent `z = G(h, ε)`。标准 Energy loss 的形式是：

```text
L = E[||z - z_target||] - 0.5 × E[||z - z_other||]
```

两个候选须在相同条件下独立采样。距离是欧氏距离，不是平方距离。第一项使样本接近数据，第二项防止所有候选被压成同一个点；它不是可以任意加权的“多样性奖励”。只给普通 MSE 预测器加随机数不会得到这个分布训练目标。评分规则框架见 [EAR](https://proceedings.mlr.press/v267/shao25b.html)，文本适配与具体采样估计见 CALM §3.3.2–3.3.3。

### 混合高斯输出头 + NLL

```text
h → {π_k, μ_k, σ_k}
p(z | h) = Σ_k π_k × Normal(z; μ_k, diag(σ_k²))
L = -log p(z_target | h)
```

先采样分量，再采样完整向量，随后由冻结 decoder 还原。GIVT 每块输出 `2×分量数×latent维度 + 分量数` 个参数。该分布比单点回归灵活，但有限分量、对角协方差及高维输出规模都有限制，需要数值稳定的 NLL 和方差下限。它与“对解码后的四个 token 算 CE”是不同训练对象。依据：GIVT §3.2、附录 C.1。

## 4. 我们现有 codec 的抗扰动实测

CALM 等工作提示：高重建率不足以保证 latent 容易生成。不过本项目是否已经因脆弱表示失败，需要实测，不能直接套用论文解释。

CPU / FP32 评价 codec step 10000，固定 2,048 个 dev 块、7,419 个有效 token，其中完整四 token 块 1,660 个。扰动为 `z + s × train_std × N(0,I)`；训练集标准差按 prior 初始化相同方法估计，各强度共用随机方向：

| 标准化噪声强度 s | token 还原率 | 完整四 token 全对率 |
|---:|---:|---:|
| 0 | 99.43% | 97.83% |
| 0.1 | 99.42% | 97.77% |
| 0.3 | 99.35% | 97.47% |
| 0.5 | 99.26% | 97.05% |
| 1.0 | 98.53% | 94.22% |

结果不支持“轻微随机扰动就完全无法解码”的说法。随机高斯扰动不等于主干有结构的预测误差，也不能证明这个空间的概率分布容易学习。当前优先问题仍是主干如何产生包含具体块内容的 latent。

详细结果：[codec 抗扰动记录](../runs/codec-v3-noise-robustness.json)。此前 [step 1500 主干诊断](../runs/prior-v3-diagnostic-step-1500.json)是另一批窗口，不直接混合其准确率。

## 5. 研究建议与后续实现

保留因果主干和轻量双向 decoder，先把“确定性向量回归”升级成可采样的下一块分布。优先比较单步 Energy 头与 GIVT 式混合高斯头，CE-only 留作消融。若使用 Energy 路线并保留纯 latent 回灌，这是结合不同工作的研究变体，需要自行验证；CALM 原文的文本反馈不能被省略后仍声称已复现。

评价同时看 codec 抗扰动、真实历史条件下的分布质量和自主回灌文本；与唯一参考续写的 token 准确率只是定位工具，不能作为开放式生成的唯一评分。论文中的扩散/flow 也会使用平方误差，但其输入包含噪声状态与时间，训练的是分布生成过程，不能与当前直接回归下一块的 MSE 混为一谈。

后续实施选择了单步随机生成头 + Energy loss，保留现有 codec 和纯 latent 回灌；这仍是结合不同工作的变体，不是完整 CALM 复现。实现及短训结果见 [DESIGN](../DESIGN.md) 与 [VALIDATION](../VALIDATION.md)。
