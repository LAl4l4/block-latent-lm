# 训练目标与回灌接口对照：2026-10-29

## 结论

这轮效果最好的文字基线，是主模型的小输出头先完成四个 token 的联合选择，再用冻结 encoder 编码成 latent，供主干继续推进；双向 decoder 只还原这个已选定内容。纯 Energy 没有改善；单纯换成 CE 虽然降低验证损失，却有严重生成重复。混合文字概率比前两种更有希望，但直接反馈的连续向量仍有问题。

这个结果不证明纯连续 latent 采样已经成功。新基线明确包含离散文字代码的选择，结构与之前的随机连续 MLP 头不同；多模态路线仍需后续验证。默认 `configs/prior.json` 没有被替换，独立配置为 `configs/prior-token-head.json`。

## 第一组：严格控制三种损失

起点是原 Energy step 1404，三组主干、随机生成头和 codec 的初始权重完全相同。每组额外 300 次更新，batch 128×累计 2，每块 8 个随机候选，训练窗口与候选随机输入分别固定到同样的 RNG 序列。AdamW 从头开始，lr=3e-4、前 50 步 warmup，之后恒定。

数据仍为原先选出的 100,000 篇 TinyStories，不改词表，不访问 test。每次验证抽取固定 64 个 dev 窗口；自主生成用固定 8 个 dev 前缀×3 个种子，每段最多 64 token。

- Energy-only：欧氏距离的 Energy score。
- 每候选文字 CE：对每个候选分别计算参考文字 CE，再平均；它可能压缩随机性。
- 整块混合 NLL：先将同一候选内四个位置的概率相乘，再平均候选概率、取负对数。有限候选估计有偏，不是默认 argmax 生成的精确似然。

三组更新数与训练窗口相同，实际 FLOPs 不同。CE / 混合 NLL 通过冻结 decoder 反传，使用激活重算控制显存；不同时保存八份完整词表激活。没有更新 codec，也没有修改用户原来的 step 1404 断点。

## 自主生成比较

辅助语言评分来自已有 V1 小模型，对实际生成结果进行条件 CE 评分，越低表示更符合该小模型的语言规律；它不是语义、事实或上下文一致性的权威评分。重复率是样本内重复 token 三元组比例的平均，平均长度不包括 EOS。

| 方案 | 辅助生成 CE↓ | 重复三元组比例↓ | 平均新 token 数 |
|---|---:|---:|---:|
| Energy-only | 9.62 | 0.1% | 37.3 |
| 每候选文字 CE | 8.27 | 40.6% | 56.5 |
| 候选整块概率混合 NLL | 8.10 | 3.0% | 42.6 |
| 小头内选定 token，再编码 latent（300 步） | 3.59 | 0.8% | 34.7 |

表中前三行来自严格同起点损失对照。最后一行是结构对照：主干与 codec 起点相同，小输出头重新初始化，参数略多；它还引入了块内自回归条件，不能把收益全部归因于单一因素。相同评分器在这 8 个前缀的真实参考续写上约为 **2.24**，仅作尺度参考。

完整指标与 24 个实际样例分别保存在：

- [三目标对照](../runs/objective-ablation-v1/report.json)
- [Energy 300 步输出](../runs/objective-ablation-v1/energy/samples-step-300.json)
- [CE 300 步输出](../runs/objective-ablation-v1/token_ce/samples-step-300.json)
- [混合概率 300 步输出](../runs/objective-ablation-v1/block_mixture/samples-step-300.json)
- [新结构 300 步输出](../runs/token-head-ablation-v1/token_head/samples-step-300.json)

CE 组的典型失败是反复出现 `Lily happy`、`how happy` 和标点循环；更低的 teacher-forced CE 没有转化为稳定自主推进。Energy 的样本有变化，但词句仍不连贯。混合概率减少了重复，句法与长链推进仍弱。

## 第二组：只替换反馈接口

在相同权重与随机输入下，临时将每步反馈向量先解码成 token，再重新编码到真实 codec 空间。当前块的展示仍用原向量，仅改变下一步主干的输入。

| 方案 | 直接回灌的辅助 CE | 解码/重编码反馈的辅助 CE |
|---|---:|---:|
| Energy | 9.62 | 9.23 |
| CE | 8.27 | 6.32 |
| 混合概率 | 8.10 | 5.70 |

这表明反馈表示的差异确实影响结果，不只是日志选择的问题。不过这条诊断路径等待 decoder，违背当前异步目标，不能直接当作正式解法。样本也仍不完全连贯。原始结果在 [反馈诊断](../runs/objective-ablation-v1/feedback-diagnostic.json)。

随后从同一混合概率 step 300 起点，再各训练 200 步，比较不加约束与添加 `||z - E(argmax D(z))||²`（权重 1）。只在完整块上计算一致性目标，目标编码停止梯度，推理不新增 decoder 依赖。

结果：无约束辅助 CE 7.90，一致性约束 8.31。一致性误差降到 0.81，但实际文字没有改善；这个权重 1 的简单做法不推荐。报告在 [一致性对照](../runs/objective-cycle-ablation-v1/report.json)。

## 第三组：先选定内容，再发布 latent

新基线的路径是：

```text
历史 latent → 大主干 → h
                      ↓
           小输出头内部依次选择四个 token 代码
                      ↓
              冻结 encoder 编成 z
                 ↙            ↘
          主干直接回灌         双向 decoder 还原
```

小头是 256 维单层 GRU，只有块内最多 3 个已选 token 的局部历史；每块重新从主干状态初始化，不保留跨块的文本历史。跨块信息必须来自大主干。训练用块内 teacher forcing 的精确联合文字 CE，推理在小头内部进行四次轻量分类选择。

这不是将输出 decoder 改回文本 AR：输出 decoder 仍是一层双向 attention，且不参与内容选择；反馈给主干的是选定代码的实际编码，不是一个代表多个潜在词句的连续均值或模糊候选。小头属于主模型的输出侧；主干不等待外部 decoder。代价是放弃当前文字输出侧的纯连续采样，以及每块新增四次小头采样。

总参数 **34,161,408**，可训练 **29,895,168**，只比 Energy 模型总参数多约 0.6%。其小头需要重新学习可训练词表 embedding，所以结构对照不是单纯更换一个 loss。

新结构保持同样 batch / 窗口与起始主干，先训练 300 步，再恢复到 600 步。

600 步时，固定 dev 文字 CE **3.454**；生成辅助 CE **3.106**；重复比例 **1.04%**。这是稳定改善，但文本仍有角色混乱、语义跳跃与语法错误，不能称为故事生成已解决。

又选了此前未用于这些方案选择的 dev 索引 128～135，step 601 时正确上下文 CE 为 **3.61**，错配主干上下文为 **5.85**，说明小头确实使用主干条件。对该组 512 个计划生成 token，decoder 与小头选择的代码一致率为 **100%**；这是小样本还原检查，不代表整个词表完全可逆。

完整新窗口样例在 [检查报告](../runs/token-head-ablation-v1/new-window-check.json)。其中一个续写片段是：

> His mom was so proud.

完整文本仍会转移话题或混淆角色，应查看原始报告，不应只挑选这句话评价效果。

## 可复查与继续运行

训练入口已支持独立 `token_head` 配置，检查点按 `latent_codec_token_head_v6` 严格加载。原训练数据、原 codec、Energy 及其他实验检查点保留；默认配置没有静默迁移。

现成小头基线已经从实验脚本的 600 步检查点通过正式 `train_prior.py` 恢复到 601 步。若继续该实验，建议先跑到 1,500 步再检查实际续写：

```powershell
& D:Codeenv313Scriptspython.exe train_prior.py `
  --resume runs/token-head-ablation-v1/token_head/last.pt `
  --stop-after 1500 --eval-batches 4
```

仍为 batch 128×累计 2，每 100 步保存。实验断点沿用 warmup 50 步后恒定 3e-4 的计划；`--stop-after` 只限制本次运行，不把总计划改成 1,500。新建实验可用 `--config configs/prior-token-head.json --init-prior runs/prior-energy-v4-check/last.pt`，其新小头将从头训练，并使用该配置的学习率计划。

复现第一组对照：

```powershell
& D:Codeenv313Scriptspython.exe experiment_prior_losses.py `
  --source runs/prior-energy-v4-check/last.pt `
  --output runs/my-objective-ablation --steps 300
```

这轮主要说明：当前问题同时涉及分布目标、块内联合选择和实际反馈内容的一致性。已有结果支持先使用可读的文字基线继续研究，不支持继续将 Energy-only 的下降当作语言生成成功，也不证明纯连续多模态主干已经跑通。
