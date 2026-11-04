# 16-token VAE → CALM 式主干

这条独立实验沿用 CALM 的 VAE 正则、后验目标采样、随机生成头和 Energy 目标。块长16、latent宽512、小主干、块内双向attention decoder、短尾有效位置标记和训练预算是本项目的改编，不代表复现论文原规模结果。

## 先训练 VAE

```powershell
& D:\Code\venv313\Scripts\python.exe train_vae.py `
  --config configs/vae-16.json `
  --run-dir runs/vae-16 `
  --device cuda
```

默认30000步，batch128×累计2，每100步保存、250步验证。学习率3e-4，100步warmup后固定。旧的确定性codec检查点不能直接用于这条VAE训练。

encoder输出每块512维均值和对数标准差，采样z后送入一层双向decoder。训练输入token遮盖概率15%、latent dropout15%；KL权重0.001、逐维free bits下限0.5。log_std限制[-8,4]用于数值稳定。损失为CE总和+0.001×有效块KL总和，再按整个累计batch的有效token数归一化；完整块时等于官方每块损失除16。

训练时被遮住的词不一定能从剩余内容确定，因此不要要求训练准确率达到纯重建codec的99%。验证不遮盖输入、不开dropout，但仍采样后验；重点观察其token准确率与完整16词块准确率。原计划门槛为token 99%、完整块95%。用户接受约89%的完整块结果作为探索性主干试验起点后，当前配置保持token 99%，将完整块启动门槛改为88%，给固定抽样验证留少量余量；仍检查2048块。这不代表原95%目标已经达到。

恢复或单独验证：

```powershell
& D:\Code\venv313\Scripts\python.exe train_vae.py --resume runs/vae-16/last.pt --device cuda
& D:\Code\venv313\Scripts\python.exe train_vae.py --resume runs/vae-16/last.pt --device cuda --eval-only
```

## 再训练主干

VAE采样重建达标后运行：

```powershell
& D:\Code\venv313\Scripts\python.exe train_calm.py `
  --config configs/calm-16.json `
  --run-dir runs/calm-16 `
  --device cuda
```

默认10000步、batch128×累计2，每100步自动保存。固定VAE；独立的历史token embedding、两层压缩MLP、8层主干以及4层随机生成MLP共同训练。历史输入不是冻结VAE encoder的输出。

每个有效目标块：生成头以相同上下文配8组独立U(-0.5,0.5)噪声生成预测z，冻结VAE为真实目标块提供100个高斯后验样本。目标为2×平均预测—目标距离，减去不同预测之间的平均距离；不使用平方距离，不逐维标准化或除sqrt(dim)，不通过decoder计算主干文字CE。

**全部有效目标块参与损失**，不继承旧mixture的64块抽样策略。`energy_batch_blocks=32`只控制能量计算分组；分组使用反向重算节约显存，不减少监督目标。实际GPU显存和速度尚需运行确认。

验证报告Energy、打乱上下文的Energy优势和1至4词前缀Brier奖励（两次独立抽样的命中项减碰撞项，越高越好）。250步一次固定prompt自由续写写入`samples.jsonl`。不能仅凭Energy下降判断生成成功。

```powershell
& D:\Code\venv313\Scripts\python.exe train_calm.py --resume runs/calm-16/last.pt --device cuda
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/calm-16/last.pt `
  --prompt "Once upon a time, Lily found a little box." `
  --max-new-tokens 64 --seed 42 --show-info
```

当前实现一次抽z、decoder按位置argmax解码16词，再将实际输出送入已训练的历史压缩器G回灌。EOS、特殊token、短尾和已结束样本沿用严格的实际文本反馈处理。不是CALM默认温度0.5的200候选筛选路径；这里对应单次基础采样。

后续反馈蒸馏目标为`C(z)≈G(D(z))`。C尚未实现，应先验证主干自由生成；该接口的历史表示与VAE目标表示承担不同职责。

## 验证与来源

CPU测试包含KL逐维裁剪、随机后验、因果边界、冻结VAE、零初始化生成头、分组反向梯度、检查点加载及两阶段训练恢复。另用完整16-token VAE配置在真实数据上做了一步CPU训练/验证/保存检查。这些只验证实现，不代表模型已收敛。正式GPU长跑未启动。

- [官方VAE源码](https://github.com/shaochenze/calm/blob/main/models/modeling_autoencoder.py)
- [官方Energy头与损失](https://github.com/shaochenze/calm/blob/main/models/modeling_energy.py)
- [官方生成与评估](https://github.com/shaochenze/calm/blob/main/models/modeling_calm.py)
- [前期逐项核对](calm-training-audit-20261030.md)

变更记录（2026-10-30）：新增VAE/CALM模型、独立训练器、两套配置与测试；公共生成入口支持新检查点，文本反馈接口支持可训练历史压缩器。旧实验配置和检查点继续使用原入口。

本轮21项模型与反馈测试通过；另1项两阶段集成测试在指定仓库临时目录中通过，包含VAE与CALM两阶段恢复训练对比连续训练的逐项完全一致性。默认系统临时目录发生权限错误后已改用仓库临时目录运行；不是模型断言失败。

后续门槛调整：VAE第13750至14500步验证完整块约88.77%–89.31%，第14500步token 99.23%。按用户选择以此继续试验，将主干完整块门槛降至88%。这些是采样后验重建指标，不是主干生成质量；实际初始化会重新验证并将成绩存入主干检查点的vae_source。

混合精度修复：文本回灌缓冲区改为按历史编码的dtype与维度分配，避免BF16预测z与FP32历史表示写入冲突。补充不同dtype回归检查，相关13项测试通过，并用实际CALM检查点完成CUDA/BF16跨两个16-token块的生成验证。
