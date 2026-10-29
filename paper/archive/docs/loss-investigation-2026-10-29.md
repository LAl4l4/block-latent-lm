# Loss 停滞与生成质量实测

日期：2026-10-29。原实验保留在 `runs/prior-energy-v4-check/last.pt` step 1404，没有覆盖。新实验仅使用 train/dev，没有查看 test。

## 原因定位

实际目标是连续 latent 的 Energy score，日志的样本 CE 不反传。Energy 是合法的分布评分，但不是文字正确率，也没有应当降至零的通用标准；真实下一块存在不确定性。

第 1000–1404 步的 405 条更新，梯度范数中位数 0.04299，学习率范围 0.0002940–0.0002982。没有发现学习率归零、梯度完全断开或累计 batch 分母错误的证据。基本公式正确：非平方欧氏距离，候选间去除自配对。

固定 seed 10042、128 个 dev 窗口、BF16 的 step 1404 检查：

| 检查 | 实测 |
|---|---:|
| 真实 codec latent 的 token 还原率 | 99.40% |
| 真实 codec latent 的参考 CE | 0.0770 |
| 模型 Energy | 0.6749 |
| 无条件经验 latent 抽样 Energy | 0.7120 |
| 标准高斯 Energy | 0.7138 |
| 零向量 Energy | 1.0096 |
| 错配条件 Energy | 0.7520 |
| 真实历史下单样本参考 CE | 9.8659 |
| 自主 latent 回灌单样本参考 CE | 11.1263 |
| 真实/自主首块距离 | 0 |
| 真实/自主平均标准化 RMS 距离 | 0.3663 |

无条件经验基线来自这批 dev 的 latent，仅作诊断，不能当作训练或公平的留出生成器。高斯和经验分布取得接近的分数，说明在这个 512 维表示空间中，单看距离评分难以区分文字结构。模型利用了一部分条件，但真实历史下的文字仍较差，自主回灌又进一步偏离。

因此，目前的主要瓶颈是表示/目标与文字分布的差距，以及自主历史分布偏移，不能只归因于优化器或回灌中的某一个问题。这是当前配置的实验证据，不是对所有 latent 模型的不可行性结论。

单样本参考 CE 不是普通 LM 的 PPL；其他合理续写也会有高 CE，不能仅凭其上升就断言生成质量下降。这里结合自由文本、词组分布评分和回灌对照判断。

复查：

```powershell
& D:\Code\venv313\Scripts\python.exe diagnose_prior.py `
  --checkpoint runs/prior-energy-v4-check/last.pt
```

原始记录：`runs/energy-diagnosis.json`。

## 对照一：有限 latent 混合 NLL

新增 `latent_ar/mixture.py`：一次主干给出 8 个 latent 与分量权重，每个分量经冻结 decoder 定义四个分类分布的乘积。

`P(block | true_history) = sum_k pi_k * product_s P(token_s | latent_k)`。

用 logsumexp 精确边缘化分量，不要求每个候选分别匹配同一答案。训练均匀抽块并按比例加权；验证计算全部有效块。推理抽一个 latent 并直接回灌，codec 冻结。

继承原 step 1404 主干，重新初始化输出头和优化器，128×2，完成 1000 步：

| step | 固定 dev 联合 NLL/token |
|---|---:|
| 10 | 7.8219 |
| 250 | 5.7536 |
| 500 | 5.5817 |
| 750 | 5.5122 |
| 1000 | 5.4580 |

这是给定真实 codec 历史的块联合条件概率。纯 latent 自主生成使用不同历史，不能把这个值冒充整个自由生成过程的精确 PPL。

相同 128 个 dev 窗口、每条件 32 次抽样，文字 Brier reward（越大越好）：

| 模型 / decoder | Brier-1 | Brier-2 | Brier-3 | Brier-4 |
|---|---:|---:|---:|---:|
| 原 Energy / greedy | 0.08871 | 0.02499 | 0.01030 | 0.00673 |
| 新混合 / greedy | -0.07287 | -0.17948 | -0.21553 | -0.22401 |
| 原 Energy / softmax | 0.08827 | 0.02485 | 0.01128 | 0.00768 |
| 新混合 / softmax | 0.07908 | 0.01845 | 0.00563 | 0.00396 |

本评分只计块内连续 n-gram，不跨块；不是论文 BrierLM 的完整复现，也没有对可能为负的分数取几何均值。

混合头的 greedy 解码显著加重碰撞；softmax 更符合训练概率模型，但该短训检查点的词组评分仍低于原模型，自由续写也未达到连贯标准。**这是 loss 下降却没有通过质量检验的消融，不推荐作为质量已提升的方案。**

检查点：`runs/prior-mixture-v5-check/last.pt`。完整评分与八段未筛选续写：`runs/text-distribution-comparison.json`。

## 对照二：Energy 加可微文字 Brier

`latent_ar/energy_text.py` 实现 Energy + 4×块内 1..4 阶 Brier loss。它使用冻结 decoder 的完整 softmax 概率，奖励参考词组命中、惩罚不同候选的碰撞；去除自配对，保留反传，不通过不可导 argmax 训练，也不要求全部候选输出同一答案。

`train_energy_text.py` 从原 Energy 全部权重初始化独立实验，学习率 0.0001、50 步 warmup，完成 1000 步。固定 dev、八候选的直接文字损失从 -0.14930 到 -0.15658（越低越好），样本 CE 从 9.8936 到 9.7741，Energy 从 0.6743 到 0.6908。

文字评分的小幅改善需要更多独立抽样和自由续写检查；没有据此宣称连贯性达标。该实验另写入 `runs/prior-energy-text-v5-check/`，通过独立入口恢复：

```powershell
& D:\Code\venv313\Scripts\python.exe train_energy_text.py `
  --resume runs/prior-energy-text-v5-check/last.pt --steps 2000
```

追加检查采用另一 seed 20042 的 256 个 dev 窗口、64 次独立抽样。softmax decoder 的点估计如下（越大越好）：

| 模型 | Brier-1 | Brier-2 | Brier-3 | Brier-4 |
|---|---:|---:|---:|---:|
| 原 Energy | 0.09019 | 0.02125 | 0.00608 | 0.00324 |
| 加文字 Brier | 0.09555 | 0.02312 | 0.00718 | 0.00304 |

1–3 阶有所改善，4 阶没有改善，greedy decoder 的 2–4 阶也没有改善；没有计算置信区间。八段固定 prompt/seed 的 greedy 自由续写仍杂乱。结论是文字概率监督有局部收益，但**尚未解决纯 latent 自主生成质量**。完整记录：`runs/text-brier-comparison.json`。训练对应完整 softmax 分布，改变成 greedy 或 top-k 会改变被评价的分布。

## 普通 token AR 基线

`token_ar_baseline.py` 是普通因果 token LM，6 层、宽 256、7,212,288 参数，可训练 embedding，输入/输出词表权重共享。同一 10 万篇故事和窗口规则，只监督目标 token。它用于区分数据/优化问题与 latent 路线的瓶颈。

它逐 token 回灌，不能将其速度作为纯 latent 异步方案的速度。不同模型的参数量、初始化和训练预算不同，属于诊断对照，不是同预算架构排名。

已完成 2000 步，batch 32×累计 2，学习率 0.0003、100 步 warmup，BF16：

| step | 固定 dev CE |
|---|---:|
| 0 | 9.0864 |
| 250 | 4.1021 |
| 500 | 3.5494 |
| 750 | 3.2799 |
| 1000 | 3.1204 |
| 1250 | 2.9911 |
| 1500 | 2.9011 |
| 1750 | 2.8216 |
| 2000 | 2.7451 |

验证覆盖固定 128 个 dev 故事窗口，目标合计 5370 token。每次从训练全集随机取窗口，没有限制成四条故事记忆任务，也没有训练 dev。相同 seed、窗口抽样顺序可复查。CE 与 Energy 的数值不可直接比较。

自由续写保留两个固定手写 prompt×两个 seed，以及上述 Brier 检查的四个 dev prompt×两个 seed，合计十二段不同续写，未筛选；文件可能另有复查时追加的相同续写。例：

> 输入：how to be a friend. She wished she could change. She wished she could say sorry
>
> seed 43：to the little girl. But she was too late. The girl was very sad and sad.

相比原 latent 模型已经出现完整句子及部分承接关系，仍有重复、逻辑跳跃、指代及语法错误。这支持“常规 token AR 是目前可确认改善 loss 和文字流畅性的路线”，不能宣称它已经达到高质量故事生成，更不能宣称 latent 架构也已因此修好。

检查点：`runs/token-ar-control/last.pt`；完整续写及复查记录：`runs/token-ar-control/samples.jsonl`；全部指标：`runs/token-ar-control/metrics.jsonl`。

生成与验证：

```powershell
& D:\Code\venv313\Scripts\python.exe token_ar_baseline.py `
  --resume runs/token-ar-control/last.pt --eval-only `
  --prompt "Once upon a time, a little girl found a strange box." --seeds 42 43
```

继续同一基线（累计到 10000 步，自动每 100 步保存；第一次 Ctrl+C 完成本步后保存）：

```powershell
& D:\Code\venv313\Scripts\python.exe token_ar_baseline.py `
  --resume runs/token-ar-control/last.pt --steps 10000
```

新模型分别按架构严格加载；原 Energy 检查点未迁移。新增测试覆盖混合联合概率/归一化、padding、双峰分工、Brier 公式和梯度、冻结 codec、因果性、KV cache、训练数据打包及中断恢复。

Python 3.13 / PyTorch 2.11.0+cu128 环境下，最终全套 **75 项测试通过**。命令：`python -m pytest tests -q --basetemp .tmp/pytest-loss-investigation-complete`，pytest 来自 `.tmp/dev-tools`。训练与测试工具统一使用 `D:\Code\venv313\Scripts\python.exe`。

## 研究依据和取舍

[CALM 原文](https://arxiv.org/html/2510.27688v1)使用带变分与噪声正则的 codec，并将生成的离散文字参与反馈。当前纯 latent 反馈及 codec 训练均有实质区别，采用 Energy loss 不等于完整复现。论文也区分 latent 评分与语言分布评价。

[EAR](https://proceedings.mlr.press/v267/shao25b.html)提供 Energy 分布评分的理论依据；适当评分规则不保证有限模型、有限数据与特定高维表示下，分数下降一定带来可用文字。[Mixture of Softmaxes](https://arxiv.org/abs/1711.03953)提供分类混合边缘化的参考，本项目的四 token latent 混合是自行实现的对照。

优先以普通 token AR 的固定 dev CE 和未筛选自由续写建立可用语言基线。若坚持单步头与纯 latent 回灌，再分别检验低维、带正则的 codec、噪声/生成历史训练与头部容量；这些目前仍是待验证方法，不能承诺有效。

每个方案都须同时报告固定 dev、多 prompt/seed 的完整自由续写、重复及提前 EOS。单独 loss 下降、小样本记忆或生成更长，都不构成质量达标。
