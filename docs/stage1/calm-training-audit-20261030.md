# CALM 官方训练配方核对（2026-10-30）

结论：下一次基线应先复现完整训练配方，再研究反馈蒸馏。当前 prior-16-feedback 是有限混合文字 NLL 实验，并非 CALM 的复现；此前 Energy 实验同样没有复现 CALM 的 VAE 及输入模块。

## 阶段一：鲁棒 VAE

官方 encoder 输出 mean 与 log_std，重参数采样 z=mean+exp(log_std)*eps。decoder 一次输出整块词表分布，用真实 token CE 训练。官方 decoder 是 FFN，当前仓库的一层双向 attention 属于实现差异。

官方训练损失为每块重建 CE 的和，加 0.001 倍逐维 KL 的和；KL 每维下限 0.5。源码中的 CE 先按 token 平均，再乘 patch_size，移植时需保持相同归一化，不能直接将 KL 权重套到另一种 CE 分母上。

输入 token 随机置零概率为 0.15；latent dropout 也是 0.15。两个 dropout 只在 AE 训练启用；后续语言模型仍对 VAE 后验采样，但不启用上述 dropout。

来源：
- https://github.com/shaochenze/calm/blob/main/models/modeling_autoencoder.py
- https://github.com/shaochenze/calm/blob/main/models/configuration_autoencoder.py

论文 §2.2 报告纯重建表示难以用于其连续语言模型，因而引入这些措施。这是作者实验结论，不能据此断言当前16-token实验失败的唯一原因也是表示鲁棒性；我们尚未做对应控制实验。

来源：https://arxiv.org/html/2510.27688v1

## 阶段二：冻结 VAE，训练主干与随机生成头

主干读取实际历史 token 的可训练 embedding，将一整块拼接后用独立的两层 compression MLP 压成一个主干输入。这里不是冻结 VAE encoder 的输出；输入模块与目标表示分别承担不同职责。

生成头接收上下文 h 和 U(-0.5,0.5) 噪声，各自线性映射与 LayerNorm，再通过条件残差 MLP 生成 z，最后输出层零初始化。官方脚本使用4层生成 MLP。

默认每个目标块采8个预测 z，同时从该真实文本块的 VAE 后验采100个目标 z。100个目标是同一块的后验采样，不是100种真实未来。

最小化的目标为 2倍平均预测—目标欧氏距离，减去预测样本之间的平均非对角欧氏距离；默认距离幂为1。它不要求随机预测样本都命中同一组词。主干训练不经过 decoder 的文字 CE，decoder 用于文本评估。

来源：
- https://github.com/shaochenze/calm/blob/main/models/modeling_energy.py
- https://github.com/shaochenze/calm/blob/main/models/configuration_calm.py
- https://github.com/shaochenze/calm/blob/main/train/train_energy.sh

## 生成与评估

生成时 z 经 decoder argmax 得到文本块，再用独立 compression MLP 回灌。T=1路径只抽一个z；官方默认T=0.5路径使用200个候选块进行频次选择，计时不能将其算成只解码一个候选。

评估需包含解码文本的 BrierLM 与自由生成。当前混合文字 NLL、单个随机候选对参考文字的 CE，均不能替代 CALM 的原评估。

若继续反馈蒸馏，其目标应改写为 C(z)≈G(D(z))，G是已经训练完成的历史压缩器。G与目标VAE encoder不能混用。训练C之前需固定教师的G、主干、生成头和D。

来源：https://github.com/shaochenze/calm/blob/main/models/modeling_calm.py

## 复现范围

官方公开脚本主配置为4-token块、128维latent；AE隐藏512，主干隐藏1024/16层。AE脚本30000步，主干脚本250000步，均为8卡训练，AE约15B tokens。我们不能把小规模TinyStories训练称为其结果复现。

建议先做小主干的4-token配方验证，固定VAE、Energy及输入模块等关键机制；能学到条件生成后，再将块长扩到16、latent扩到512。这些缩规模及扩块实验必须单独标记。现有16-token纯重建codec保留作对照，不能作为已经符合官方VAE配方的检查点。

来源：
- https://github.com/shaochenze/calm/blob/main/train/train_autoencoder.sh
- https://github.com/shaochenze/calm/blob/main/train/train_energy.sh
- https://github.com/shaochenze/calm

## 为什么暂不直接复现 Cola2

Cola2在token侧继续更新VAE/decoder，仅对Flow Matching路径阻断编码器梯度，并采用动态分段、可读取历史文本的masked diffusion decoder。其项目页目前仍标注Code Soon、Model Soon。它不能作为现有轻量、仅看单个z的decoder可直接替换的官方实现。

来源：
- https://arxiv.org/abs/2610.31305
- https://hongcanguo.github.io/Cola2/

本次仅完成源码核对与配方记录，未更改训练配置、模型结构或启动新训练。
