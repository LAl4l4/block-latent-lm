# 当前架构约定与实现快照

更新：2026-10-31。本文用于保存当前已经对齐的设计，后续讨论、实现与实验先核对本文。代码现状与待验证方案分别标注，不能把接口存在当作实验成功。

实验进度、暂停位置与接续安排见 [PROGRESS.md](PROGRESS.md)。当前讨论建议训练尝试同一h配8份u、推理使用1份；尚未完成正式1份/8份对照，不能记为8份更好的实测结论。

## 1. 当前明确的研究目标

大主干按16-token块更新；小解码器在块内生成文字。小学生提前预测这次文字经历史编码器得到的反馈，用它提前启动下一块。只向前猜一步，每两块同步并用真实反馈修复主干缓存。

**最新对齐的设计：学生与GRU在块开始时读取同一份、实际影响选词的随机数。学生预测的是这一次生成的对应反馈。**

写成公式：

\[
u_t\sim\mathcal U([0,1)^{16}),\qquad
x_t=D(h_t,u_t),\qquad
f_t=E(x_t),\qquad
\hat f_t=P(h_t,u_t).
\]

`u_t`不是两边分别抽取的随机数，也不是生成结束后才从文字中提取的答案。它在解码之前准备好，同时交给D与P。当前目标是配对预测准确，不以学生独立采出另一个合理未来替代这一目标。

## 2. 统一符号

| 符号 | 含义 | 典型形状 |
| --- | --- | --- |
| B | 跨块因果Transformer主干，读取历史反馈，输出当前块的条件状态 | 输入/输出每位置512维 |
| D | 完整16-token解码链：局部GRU选词、VAE均值编码、VAE decoder还原、实际词表选择 | `h,u → 最多16个实际token` |
| E | 历史编码器，把实际文字块压缩成送给B的反馈 | `最多16个token → 512维` |
| P | 反馈学生，目前是小残差MLP | `h,u → 512维预测反馈` |
| h_t | B读完前缀后，用于生成第t块的上下文状态 | 512维 |
| z_t | GRU选定的原始token块，经VAE均值编码得到的内容latent | 512维 |
| f_t | 实际输出文字经E编码得到的真实反馈 | 512维 |
| u_t | 该块预先抽取的16个均匀随机数 | 16维 |

**h、z、f虽然同为512维，但不是同一种表示，不能直接互换。** P的目标是f，不是z，也不是下一步h。

历史文档的C通常就是这里的P，G通常就是这里的E。旧文档里“decoder”有时仅指VAE decoder；本文的D包含完整选词与编解码过程，计时时也按完整链计算。

## 3. 当前真实模型结构

现有教师架构名为 `latent_calm_token_v1`。词表8192，PAD/BOS/EOS分别为0/1/2，块长16。以下参数量由当前冻结教师实例按独立参数统计，已扣除各模块内部的权重共享。

| 模块 | 结构 | 参数量 |
| --- | --- | ---: |
| B | 宽512、8层、8头、前馈宽1536的因果Transformer | 27,271,680 |
| E | 独立token embedding＋按块拼接的MLP历史压缩器 | 13,126,144 |
| 局部GRU选词头 | 宽256、1层GRU，词表输出与其输入embedding共享权重 | 2,623,488 |
| VAE codec | 编码器、后验投影、块内双向decoder | 7,949,568 |
| 教师合计 | B＋E＋局部选词头＋VAE | **50,970,880** |
| P，greedy版 | 宽512、2个残差MLP块，不读取噪声 | **1,579,008** |
| P，共享u版 | 相同MLP，增加16维噪声投影 | **1,587,712** |

### 3.1 大主干B

- B每16个实际生成token推进一个位置，不逐token维护全局KV。
- KV cache对应输入B的历史反馈位置，包括prompt的块表示与生成块的反馈。
- 它不保存局部GRU每个token的hidden，也不读取局部GRU的token级KV。

### 3.2 历史编码器E

`CalmModel.encode_history`使用独立的8192×512 token embedding。每16词的embedding展平成8192维，再拼接16个有效位置标记，得到8208维输入：

```text
token IDs + 有效长度
    → 16 × 512 embedding（padding位置清零）
    → flatten后拼接16个有效标记：8208维
    → Linear(8208,1024) → SiLU
    → Linear(1024,512) → LayerNorm
    → f
```

E与VAE encoder是两个不同模块。真实回灌标签必须取实际文字的 `encode_history` 结果。

### 3.3 局部GRU及实际解码D

每块开始时，GRU hidden由 `tanh(Linear(h_t))` 初始化；第一个输入为BOS，之后输入本块刚选出的词。

GRU只在当前16词块内递推：**没有跨块attention，不携带上一块的GRU hidden。** 跨块依赖由B提供。

当前实现的完整路径为：

```text
h_t + u_t
    → 局部GRU依次选出16个原始token
    → VAE encoder的后验均值 → z_t
    → VAE decoder一次输出16个位置的logits
    → 实际token选择、EOS截断 → x_t
    → 历史编码器E → f_t
```

VAE decoder宽256、1层、4头、前馈宽768，只在当前块内做双向attention，不读取文字历史或其他块。VAE输出投影与其encoder embedding共享权重。推理中的VAE编码使用后验均值，不额外抽取VAE后验高斯噪声。

VAE重建不是绝对无误差，所以GRU原始token与最终实际输出x_t可能不同。P的标签始终由最终实际x_t计算，不能用原始GRU选词直接替代。

**“大主干16词 → 中间小主干每4词更新 → GRU”尚未实现。** 当前D没有额外的4-token级Transformer；那是另一个架构提案，不能混入现有实验描述。

### 3.4 反馈学生P

当前MLP学生结构：

```text
h：512维 → LayerNorm → Linear(512,512) ──┐
u：16维 → 减0.5 → Linear(16,512) ───────┤ 相加
                                      ↓
          2 × 残差块：[LayerNorm → Linear → SiLU → Linear]
                                      ↓
                         Linear(512,512) → 预测反馈
```

greedy旧版没有u分支。学生不读取当前块token、当前真实f或解码后的z；这些在预测时还不可用。`h_t`已经包含由B处理过的过去历史。

另有窗口attention学生：宽256、2层、4头，读取当前和前两块h，含prompt历史版为1,844,736参数。该方案已测但未见明确改善；当前共享噪声方案先以MLP为基线。窗口attention实现当前不支持噪声输入，不能直接称为共享噪声学生。

## 4. “同一份噪声”在代码中具体是什么意思

当前GRU采样接口已支持在块开始时提供整个 `u_t=[u_1,...,u_16]`。第j词的操作为：

1. 根据h和本块已选前缀算logits。
2. 屏蔽PAD/BOS，按温度缩放并作top-k筛选。
3. 用 `u_j` 在累计概率中作逆CDF选词。

P同时读取原始同一个16维u，内部减0.5只是居中，不改变它与GRU噪声的配对关系。

- 温度0：GRU直接argmax，u不影响结果；这就是已经报告的greedy实验。
- 采样实验：现已采集的数据使用温度0.8、top-k 50，u确实影响选词。
- 当前没有把u额外注入GRU初始hidden；“共享”的实际实现是共享选词所用的随机数。
- 同u只规定随机选择的配对，不天然编码“小猫/小狗”这样的语义计划。

旧CALM配置里的 `noise_dim=64`、`energy_candidates=8` 等来自历史Energy生成头，**不决定当前TokenHead的选词随机数维度**。当前 `draw_noise` 由块长决定，实际为16维。

## 5. 两块一组、只提前一步

把当前组的块记为A和B，避免与主干符号B混淆。组前已有真实状态h_A和干净cache。

```text
                    ┌→ D(h_A,u_A) → x_A → E → f_A ──────┐
h_A，干净cache ──────┤                                  │
                    └→ P(h_A,u_A) → f_hat_A             │
                          → 主干预测路径 → h_hat_B       │
                          → D(h_hat_B,u_B) → x_B → E → f_B
                                                       ↓
                           从组前干净cache重放 [f_A,f_B]
                                                       ↓
                                           下一组真实状态
```

同组的共享关系是：**P预测A反馈时使用u_A，D解码A也使用u_A；D解码下一块B使用新的u_B。** 不是把u_B错给预测A的P，也不是让A和B共用同一块的抽样序列。

当前已实现的batch模式可以写成：

```python
# 以下为概念伪代码，不是直接调用接口。
cache_before_pair = clean_cache
u_A, u_B = draw_noise(), draw_noise()

f_hat_A = P(h_A, u_A)
h_hat_B, provisional_cache = B.step(cache_before_pair, f_hat_A)

# 两个条件状态齐备后拼batch，不依赖A的实际文字。
x_A, x_B = D.batch([h_A, h_hat_B], [u_A, u_B])
f_A, f_B = E(x_A), E(x_B)

# 丢弃预测路径的缓存，从组前真实前缀开始重放两个实际反馈。
h_next, clean_cache = B.replay(cache_before_pair, [f_A, f_B])
```

重放使用一次带因果掩码的两位置前向。必须保留位置号、有效长度和EOS mask；结束行不能复活。prompt尾部的部分块保留原来的块边界，不能把prompt与生成文字直接拼接后重新分块来冒充相同前缀。

当前batch模式等待h_hat_B准备好才开始双块解码；不是双CUDA stream。串行双块模式也已实现，作为相同调度语义的对照。

**修复缓存只让下一组状态对应实际已经输出的文字，不能撤销第二块已经生成的内容偏差。** 当前实现没有阈值拒绝或重算，始终保留预测第二块，属于有损实验，不保证原教师分布。

## 6. 训练监督：配对关系必须保留

基线教师先用token CE训练B、E与局部GRU，VAE冻结；蒸馏阶段冻结整个教师，只更新P。

共享随机数训练的基本样本是：

\[
(h_t,u_t,\underbrace{E(D(h_t,u_t))}_{f_t}).
\]

固定h和u后，该教师的对应目标是确定的。这个问题不能再解释成“同一个输入有多种未来，所以任何配对回归都必然取平均”。是否采用MSE，要通过预测精度与下一块行为的实测判断。

可比较的监督有：

- 配对反馈损失：让 `P(h,u)` 接近同一份u对应的f。
- 下一块行为一致性：固定相同真实历史，分别把真实f与预测f送进B，再在同一下一块词前缀上比较局部词分布。

下一块KL的梯度可以穿过冻结B与冻结局部GRU传回P，但不更新教师参数。现有原型比较原始GRU词分布，损失可加MSE锚；具体权重属于实验设置，尚未确认为最终方案。teacher生成词的CE不是自然语料真值CE。

## 7. 八份噪声与Energy：约定及未定事项

若对同一个h准备8份噪声，必须保留8组配对：

\[
u_i\longrightarrow\left(P(h,u_i),\ E(D(h,u_i))\right),\qquad i=1,\dots,8.
\]

这里是同一个P重复使用8次，可沿batch计算；不是默认训练8个独立模型，也不是P和D各自生成无关的8个未来。

**只匹配两组输出分布，不能保证同下标样本对应。** 即使P把不同u对应的未来互相交换，分布指标也可能很好，但反馈配对仍然错误。Energy loss不能单独当作“共享决策已学会”的证据。

Energy可以作为配对监督之外的辅助或对照。具体估计方式、配对项、抽样独立性和权重尚未确定，不能写成已经落地的最终损失。当前P没有完成Energy训练；历史 `runs/calm-16` 的Energy主模型不是这次的反馈学生实验。

## 8. 已实现、已测试与待测试

| 项目 | 当前状态 |
| --- | --- |
| 16-token全局B＋局部GRU＋实际文字E回灌 | 已训练、已有教师检查点 |
| P(h) greedy MLP | 已训练并评估；一步精度未达到保质要求 |
| 1k/4k/16k故事与158万/525万学生对照 | 六组单seed训练完成；更大数据改善dev MSE，扩容量未相应改善 |
| 当前/前两块h的attention学生 | 已训练与评估，未见明确改善 |
| 两块一组、真实反馈重放修复cache | 已实现，CPU/CUDA核验及greedy GPU评估完成 |
| 双块batch解码 | 已实现；固定64词有约20%耗时下降，质量未过关 |
| 双CUDA stream流水线 | 未实现 |
| 阈值拒绝、草稿重算、无损验证 | 未实现；只有离线阈值诊断 |
| GRU/P共享u接口 | 已实现，CPU随机采样流程检查通过 |
| 随机教师的配对u数据 | 已采集，见下节；不是共享噪声学生训练结果 |
| 正式共享u与隐藏u学生训练对照 | 尚未执行完成，没有可报告的效果结论 |
| P的KL行为微调 | 代码和CUDA反传检查已完成；正式对照未完成 |
| P的Energy与8份配对噪声实验 | 待实现/待验证 |
| B、D的反馈误差鲁棒训练 | 未开始 |
| 16→4→GRU三级结构 | 未实现 |

## 9. 固定权重与数据位置

| 内容 | 路径 |
| --- | --- |
| 教师原始训练检查点 | `runs/calm-repair-token/last.pt`，step10000 |
| 蒸馏用冻结教师快照 | `runs/feedback-distill-greedy/data/teacher.pt` |
| 4k greedy学生 | `runs/feedback-study/scan/n4096-h512-s2032/best.pt`，step300 |
| 16k greedy学生 | `runs/feedback-study/scan/n16384-h512-s2032/best.pt`，step1100 |
| 扩充greedy数据 | `runs/feedback-study/greedy-data` |
| 随机教师配对u数据 | `runs/feedback-study/stochastic-data` |

冻结教师SHA256：`1a09997a091049b4a7708054d854c84c97d4557a8c736be5e0deacd152aa89bc`。

greedy数据为16,384篇train/33,439个有效反馈块，512篇dev/972块，128篇独立test/269块。随机数据为4,096篇train/8,858块，512篇dev/1,028块，128篇test/269块，温度0.8、top-k 50。随机数据的train/dev/test采样流分别使用2031、1002031、2002031；不复用同一串u作泛化验证。

数据保留h、r（即f）、u、故事号、块号、实际过去反馈、源prompt和下一块实际文字。只把用于继续推进的完整非EOS块作为反馈训练样本。manifest记录教师、数据、词表、温度与来源；加载时以manifest里的实验温度为准，不能只看原训练配置中的0.8。

## 10. 已知结果和当前边界

greedy两块batch在当前RTX 4060 Laptop上，固定64新词的耗时约从86–88 ms降到69–70 ms；包含prompt prefill、实际编码和修复，但始终保留预测第二块。第二块整段与教师一致为4k学生0/107、16k学生1/107。

固定真实历史、只替换一次反馈时，test第一词argmax约94%的位置改变；平均相对反馈误差仍超过60%。这不是“单步已经解决，只剩多步漂移”的结果。也不能由此判定MSE是唯一原因。

上述均为greedy结果，**不能当作最新共享噪声设计失败的证据**。共享u方案尚无正式训练效果结论。

## 11. 代码入口与相关文档

- B与E：[latent_ar/calm.py](latent_ar/calm.py)。
- GRU教师：[latent_ar/calm_token.py](latent_ar/calm_token.py)、[latent_ar/token_head.py](latent_ar/token_head.py)。
- VAE：[latent_ar/vae.py](latent_ar/vae.py)。
- P与连续回灌：[latent_ar/feedback_student.py](latent_ar/feedback_student.py)。
- 两块真实修复：[latent_ar/pair_feedback.py](latent_ar/pair_feedback.py)。
- 监督研究入口：[experiments/feedback_study.py](experiments/feedback_study.py)。
- 行为损失：[experiments/feedback_dynamics.py](experiments/feedback_dynamics.py)。
- [两块调度和成本口径](docs/pair-feedback-protocol.md)、[两块GPU实测](artifacts/pair-feedback/results.md)。
- [MSE/共享噪声诊断方案](docs/feedback-supervision-study.md)、[Block Transformer文献核对](docs/block-transformer-feedback-audit.md)。

所有Python命令统一使用 `D:\Code\venv313\Scripts\python.exe`。后续修改结构、共享噪声定义、损失或验收规则时，同步更新本文与CHANGELOG，并将新实验存入独立目录。
