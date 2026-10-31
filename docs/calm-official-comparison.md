# 当前实现与 CALM 官方实现的再次核对

结论：单次随机生成头及 Energy 目标的核心已经对齐；整个实验并不是官方同条件复现。不能从本项目16词小模型失败推断CALM的一次连续采样无效，也不能将局部GRU对照成功称为CALM原方法成功。

## 已对齐的关键机制

- 独立可训练的历史token embedding及两层压缩器，将实际生成文字回灌。
- U(-0.5,0.5)噪声和上下文分别线性映射、LayerNorm；四层条件门控残差MLP；最后输出层零初始化。
- 每块8个预测、100个真实VAE后验目标样本；2倍平均预测—目标欧氏距离减去非对角预测间距离。当前没有平方距离、逐维标准化或sqrt(dim)缩放。
- 冻结VAE训练主干，主干的原Energy路径不使用文字CE。
- VAE后验采样，输入/latent各15% dropout，逐维KL floor0.5，KL权重0.001。

官方证据：[生成头与目标](https://github.com/shaochenze/calm/blob/main/models/modeling_energy.py)、[VAE](https://github.com/shaochenze/calm/blob/main/models/modeling_autoencoder.py)。

## 仍然存在的实质差异

| 项目 | 官方公开主配置 | 本项目 |
|---|---|---|
|块长/latent|4词/128维|16词/512维|
|主干示例|16层、宽1024|8层、宽512|
|VAE结构|两阶段残差FFN encoder/decoder，hidden512|拼接MLP encoder，decoder为一层双向attention，embedding256|
|训练序列|拼接文本后切8192 token，2048 latent步|单故事窗口，最多128历史+64目标token，最多12 latent块|
|预热|主干2000步|100步|
|AE数据预算|约15B token|当前VAE运行约50M目标token，有重复采样|
|主干训练|论文250k步，每步约2M token|10k步，累计113.6M目标token，有重复采样|
|生成温度|基础单样本；低温使用候选池选择|仅基础单样本|

上述tokenizer、监督区域和数据统计口径不同，token量不是严格等价的算力比较。官方语料约230B独立token；本项目训练子集约22M token。

官方没有报告16词块。论文测试K=1/2/4/8，并报告K8出现更大性能下降，作者认为可能涉及容量限制。这是本实验最重要的任务差异，但尚无控制实验能证明它是唯一原因。

来源：[论文§7.1–7.2](https://arxiv.org/html/2510.27688v1)、[训练脚本](https://github.com/shaochenze/calm/blob/main/train/train_energy.sh)、[数据打包代码](https://github.com/shaochenze/calm/blob/main/train/train_calm.py)。

数值和边界适配也不同：本项目额外加入slot有效位、短尾mask、log_std[-8,4]裁剪，噪声先以FP32采样再进入autocast，距离与posterior exp用FP32。官方按hidden dtype生成噪声，VAE CE按完整块计算。完整块下我们的VAE损失为官方损失除K，短尾比例较高时权重不严格一致。没有证据支持将这些数值差异直接判为失败主因。

## 一次采样与200候选的区别

官方T=1：一次随机头输出z，decoder一次并行argmax得到K词。官方BrierLM评价的是基础采样分布，因此论文结果不能解释为全部由200候选筛选获得。

官方默认生成T=0.5：在相同上下文抽200个z并解码，将完全相同的K词块计数，按组合数权重抽样。如果没有重复块，算法降低匹配要求并退回候选选择。这增加了生成头与解码成本，不能算作只采一个z。

来源：[官方temperature_sampling](https://github.com/shaochenze/calm/blob/main/models/modeling_calm.py)。

本项目实测：最终Energy检查点，在32个固定dev上下文各抽200个完整16词候选，**32/32候选池均没有重复整块**，全部进入T=0.5算法的回退情况。该测试保留本项目PAD/BOS屏蔽，没有EOS截断，以完整16词元组计重复；单次候选池，没有误差条。结果见`artifacts/calm16-pool200-diagnostic.json`。

这说明当前模型上不能简单照加200候选就期待解决问题；它没有在这些候选池中获得重复整块所提供的选择优势。不等于已经完整评价所有温度或更大候选池。

## 作者直接回应过类似问题

[issue #18](https://github.com/shaochenze/calm/issues/18)中，有人报告自行训练产生不连贯文字。作者回应建议生成温度0.5或更低，并强调充足训练token（举例100B）与更长预热。该回应支持连续生成训练较困难，不能作为本项目具体根因的证明。

官方现已发布[CALM-M/L/XL及Autoencoder权重](https://huggingface.co/collections/cccczshao/calm)。因此接下来的直接验证路径应是先运行官方已训练K4模型，分别检查基础采样与官方低温采样；再以K4、原VAE结构作为从头训练对照。16词扩展应放在该对照能工作以后，而不是把缩小规模又扩到16词的实验称为原配方复现。

本次完成源码/论文/作者回复核对和200候选离线推理诊断，没有启动新训练或修改已有模型权重。
