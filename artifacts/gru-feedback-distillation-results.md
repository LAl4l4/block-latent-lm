# 16-token GRU基线与第一轮反馈蒸馏实测

日期：2026-10-30。结论：**GRU基线已经能学，第一轮反馈学生尚不能可靠替代真实反馈。** 本轮完成了实际GPU采集、3000步学生训练和32/64-token闭环评估，没有实现或测量异步加速。

## 1. 固定结构与教师

主干每16个token运行一次；局部GRU每块从主干状态h重新初始化，不带跨块attention，也不携带上一块hidden。跨块历史由主干与实际输出文本的压缩反馈提供。

教师为 `runs/calm-repair-token/last.pt`，已核实检查点step10000、最终dev记录和seed42/43两条样本。主干与局部GRU通过文字联合CE训练，冻结VAE。该基线继承早先Energy训练的主干，不是从零训练；未来比较计算预算时必须计入这部分预训练。

固定dev曲线（CE越低越好；条件优势为打乱历史后CE减去正确历史CE）：

| 训练step | dev CE/token | 条件CE优势 |
| --- | ---: | ---: |
| 0 | 9.1015 | 0.0024 |
| 1000 | 3.5036 | 0.7763 |
| 2000（恢复时） | 3.2368 | 1.0221 |
| 4000 | 3.0340 | 1.2835 |
| 6000 | 2.9151 | 1.4088 |
| 8000 | 2.8475 | 1.5441 |
| 10000 | 2.7873 | 1.6252 |

step2000在恢复前后有两次验证，CE相同，随机打乱历史导致条件优势略有差异；此处采用恢复记录。CE是teacher-forcing下局部选词的指标，不包含自由生成时VAE编码/解码的全部误差，不能直接当成实际输出质量。

这些结果支持“本结构可以学习条件语言建模”。实际教师仍会跑题、重复，不能称为高质量故事模型。最终随机样本中有“have a snack as a good day inside”等错误；本轮greedy教师还存在明显套话和重复（见后文）。

## 2. 蒸馏目标、数据与版本

本轮固定greedy教师，学习：

`C(h) ≈ G(D(VAEmean(GRU_greedy(h))))`

其中GRU先选出16个token，VAEmean编码它们，D解码出实际展示的文字，G压缩实际文字作为主干下一步反馈。C直接读解码前的h，试图提前给出这个反馈。标签不是某个未知未来的latent坐标，而是固定教师在给定h下实际产生的反馈。

- 冻结教师：`runs/feedback-distill-greedy/data/teacher.pt`，step10000。
- 快照SHA256：`1a09997a091049b4a7708054d854c84c97d4557a8c736be5e0deacd152aa89bc`。
- 训练4096个prompt，得到**8392个有效反馈块**。
- dev 512个prompt，得到**972个有效反馈块**。
- 来自不同train/dev故事划分；仅保留会继续推进主干的完整、非EOS块。64-token预算的末块不作为下一步反馈标签。
- 快照、数据与词表哈希记录在 `runs/feedback-distill-greedy/data/manifest.json`，训练和评估会校验版本。

学生为宽512、两层残差MLP，共**1,579,008参数**，输入h，无噪声。batch256，AdamW，3000步，损失为反馈MSE。随机教师的共享噪声版本尚未实验，不能由此推断随机采样效果。

## 3. 单步拟合：有泛化收益，但很快过拟合

| step | train MSE | dev MSE | dev cosine |
| --- | ---: | ---: | ---: |
| 100 | 0.2002 | 0.2030 | 0.7369 |
| 300（best） | 0.1788 | **0.1947** | **0.7497** |
| 500 | 0.1632 | 0.1983 | 0.7455 |
| 1000 | 0.0860 | 0.2410 | 0.7000 |
| 2000 | 0.0234 | 0.2935 | 0.6526 |
| 3000 | 0.0132 | 0.3073 | 0.6421 |

恒定输出训练集平均反馈的dev MSE为**0.2693**。best学生的dev MSE较它低**27.69%**，说明学生确实学到了超出均值的信息。然而继续训练主要是在记训练集，最后一步dev误差已经差于均值基线。

闭环采用验证集挑选的 `student/best.pt`（step300），没有使用step3000的last.pt。学生SHA256为 `862352af54a964d1d7d1e34c5bb3c79d6cc655414c9c3e8fd077eaa31f93e03c`。

## 4. 闭环：首块正确，之后没有保持住教师输出

从相同128个dev prompt出发，分别用真实反馈和C反馈推进主干；32与64-token是同一批prompt上的两个长度预算，不是两份独立测试集。

| 指标 | 最多32 token | 最多64 token |
| --- | ---: | ---: |
| 首块完整一致 | 128/128 | 128/128 |
| 平均共同前缀/token | 14.3125 | 14.3125 |
| 逐位置token一致率，按较短长度 | 59.87% | 42.51% |
| 逐位置token一致率，按较长长度 | **56.69%** | **33.99%** |
| 教师实际总token | 3231 | 5184 |
| 学生实际总token | 3270 | 5097 |
| 教师平均长度 | 25.24 | 40.50 |
| 学生平均长度 | 25.55 | 39.82 |
| 教师遇EOS的prompt | 55/128 | 84/128 |
| 学生遇EOS的prompt | 53/128 | 86/128 |
| 学生访问状态上的有效反馈块 | 97 | 225 |
| 这些状态的反馈MSE | 0.2090 | 0.2113 |
| 这些状态的反馈cosine | 0.7285 | 0.7255 |

首块未使用学生反馈，因此100%一致是实现应满足的检查，不能作为蒸馏成功的证据。总体一致率也包含这些免费一致的首块。共同前缀均值含提前EOS的短输出，因此14.31不能解读为所有样本都在第14个token出错。

较长长度归一化把单侧提前EOS后缺失的位置也计入分母。长度和EOS总量相近，但不代表相同prompt在相同位置结束。反馈误差仅统计真正用于推进的完整、非EOS块，不混入最后预算块。

学生访问状态的MSE略高于teacher轨迹dev MSE，和分布偏移的担忧一致；这两组样本不完全相同，**不能只凭0.1947与0.2113证明唯一失败原因是分布偏移**。也没有测量每个反馈维度对下一步文字分布的敏感度。

### 实际文字（64-token评估文件中的原文片段）

dev故事1，教师：

> Lily was so scared and ran away. She never saw the dog again and again. The end.

学生：

> Lily was so scared and ran away. She never saw the dog again and again wished she had listened to her mom. She wished she had listened to her mom remembered to be careful when playing with her toys.

dev故事2，教师：

> he had to go to the pond. He said, "I want to be a friend, I want to play with you."

学生：

> he had to go to the pond. He said, "I want to be.

后续学生文字还出现 `I'm sorry, I didn't mean to scare you.,` 和 `all day long,,`。这类不完整句子和续接错误支持闭环反馈没有保持稳定的判断。另一例中教师自身连续重复 `I'm sorry`，这属于教师质量问题，不能归因于学生。

token不一致不自动意味着语义更差：生成不同但合理的续写是可能的。本轮同时看到了明显句法和续接错误，因此不只是“改写导致精确匹配下降”。但这只是前16条导出样本的人工检查，没有盲评或系统语义质量评分。

## 5. 可以得出的结论与下一步

1. **已经有可继续研究的GRU基线。** CE持续下降且依赖正确上下文；教师仍需改进连贯性，不能直接当成高质量生成系统。
2. **第一轮C(h)反馈蒸馏未通过保质检查。** 最佳单步MSE有收益，闭环仍明显偏离；不能用很低的train MSE宣称替代成功。
3. 当前只测试一个教师、一次学生训练、一个greedy策略；best选择与闭环prompt共用dev划分，因此不是最终独立测试集结果，更不能据此断言该方法必然失败。
4. **尚无异步加速结果。** 目前仍串行计算真实GRU、实际反馈和C，目的仅是诊断质量。计时与加速结论应等并行调度实现后再做，并包含实际文字全部完成的耗时。

建议下一轮先做小规模、可归因的对照，不直接重复3000步离线MSE：

- 固定当前best为离线基线，保留早停；在相同新数据预算下比较“更多teacher轨迹”与“学生闭环访问状态”。后者在学生状态h上运行冻结教师GRU和G，给出该状态的真实反馈标签，再继续训练C，检查闭环误差能否改善。不能只新增on-policy数据而没有等量off-policy对照。
- 如果MSE进一步下降但文字仍不稳定，比较加入**冻结主干下一步局部词分布的蒸馏损失**：对同一历史分别注入真实/预测反馈，固定相同局部token前缀，对齐下一块词分布。此项直接检查反馈的使用效果；可离线训练时增加教师计算，不改变生成时学生规模。它是待测方案，不是已证实的修复。
- 新留出故事只用于最终闭环测试；同时报告每个块后的指标、长度/EOS、实际续写，以及多训练seed的不确定性。质量维持后再实现异步调度并测端到端速度。

第一轮到此完整汇报，没有启动新一轮训练，也未修改已有基线权重。

## 6. 结果文件

- 基线：`runs/calm-repair-token/metrics.jsonl`、`samples.jsonl`、`last.pt`。
- 数据与来源：`runs/feedback-distill-greedy/data/manifest.json`、`teacher.pt`、`train.pt`、`dev.pt`。
- 学生：`runs/feedback-distill-greedy/student/metrics.jsonl`、`best.pt`、`last.pt`。
- 闭环原始指标与16组文字：`artifacts/feedback-distill-greedy-32.json`、`artifacts/feedback-distill-greedy-64.json`。
- 执行说明：`docs/gru-baseline-distillation-run.md`。
