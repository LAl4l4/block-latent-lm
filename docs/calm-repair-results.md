# CALM-16 排查与实测修复

结论：当前预算下，直接学习16词联合连续latent分布的接口训练效果差；主干、历史压缩器和VAE并没有失去拟合能力。局部GRU按文字联合CE训练是本次实测能训练、能使用主干的工作基线，但放弃了“单次并行选完整块”的要求。它不是原单步随机MLP方案已经成功的证明。

## 排查证据

1. 相同prompt与噪声下，真实历史并行计算和自由生成首块的上下文/latent差约4.8e-7。未发现足以解释坏文本的首块错位或断梯度。
2. 64个验证完整块的真实VAE后验解码token准确率99.22%、整块89.06%。预测latent的范数/协方差规模已接近真实分布，但解码文字仍差，说明匹配这些整体统计不能保证语言质量。
3. 固定32个训练块、冻结主干/VAE，只更新原Energy生成头：100步后token重建98.71%、整块79.69%；1000步为98.95%/83.20%。这是记忆容量检查，不是泛化或语言建模成绩。它支持基本梯度和解码通路可用。
4. 修复一项验证指标问题：变长目标的历史打乱原来可能换入padding query，现改为同一块位置的有效行互换。这影响条件优势解释，不改变正常训练目标，也不足以解释原本差的续写。

这些证据定位到当前连续分布学习接口的监督效率、联合依赖建模与训练预算组合；没有控制实验能把失败唯一归因于Energy理论、512维或16词中的某一个。

## 实际训练对照

两个新头均从原CALM第10000步继承同一冻结VAE、历史压缩器及主干，独立初始化新输出头，再使用相同数据流、batch128×2训练2000步；训练所有有效目标块。不是从零训练的规模对照。原检查点保持不变。

| 方法 | 实际采样 | 结果 |
|---|---|---|
|原Energy|单次随机MLP|10000步后续写仍差|
|条件Flow Matching|16次Euler头调用|追加2000步速度MSE 1.72→1.11，但未解决续写|
|局部GRU文字联合CE|块内16次顺序选词，再VAE均值编码和双向D还原|追加2000步，固定验证CE 9.10→3.24，生成改善|

统一评估为128个固定验证窗口、376个可配对块，每个条件16次采样，12条最多64-token自由续写。以下Brier是块内所有起点的n-gram奖励，不能与旧日志只统计块起点的前缀奖励直接混用。

| 方法 | 2-gram Brier奖励（高优） | 4-gram奖励（高优） | 辅助AR模型对实际输出CE（低优） |
|---|---:|---:|---:|
|原Energy|0.001192|-0.0000158|8.5314|
|条件Flow|0.001676|0|8.7577|
|局部GRU|0.006365|0.0008843|2.9260|

辅助AR评分只是流畅度代理，不是语义真值，12条续写也是小样本；没有估计置信区间。模型间训练目标不同，不直接比较Energy、速度MSE与文字CE。

## 主干确实有作用

使用另一组固定128个dev窗口（seed2029），5520个相同目标token，只干预主干传给小头的上下文，保留块内真实前词：

| 上下文 | CE/token |
|---|---:|
|正常|3.0830|
|同块位置有效行错配|4.0695|
|置零|3.9155|

块首词CE为正常3.427、错配9.762、置零10.261；后续位置仍有较小但可测差异。冻结VAE无梯度，主干梯度范数0.252，历史压缩器0.160，小头3.568。主干参数相对初始检查点RMS变化0.01252，VAE变化0。参数变化本身不是质量证据；这里与heldout条件消融一起说明小头确实使用了主干信息。

## 可用命令

本次有效检查点：`runs/calm-repair-token/last.pt`，step2000。继续到第10000步（新增8000步；不覆盖原Energy实验）：

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.train_calm_repair `
  --method token `
  --run-dir runs/calm-repair-token `
  --resume runs/calm-repair-token/last.pt `
  --steps 10000 --device cuda
```

生成：

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/calm-repair-token/last.pt `
  --prompt "Once upon a time, Lily found a little box." `
  --max-new-tokens 64 --show-info --device cuda
```

新实验每100步保存；保存先于续写评估。已在独立临时目录验证从2000恢复到2002，不改动上述用于评估的2000步检查点。

## 仍然存在的限制

实测续写有可读句子，但仍有语法错误、重复、实体漂移和不合理寓意；不能称为已经训练好的高质量语言模型。小GRU总是先选择16词，VAE mean编码后D再次还原，实际发出的词可能被VAE重建误差改变；heldout小头CE没有涵盖这个误差，统一生成评估则包含它。

主干仍每16词运行一次，decoder仍为一层块内双向注意力。新增的局部选词头顺序运行16次；没有实测端到端速度优势。本次找到的是可训练基线，不是原单次连续latent路线或反馈蒸馏已经完成。

## 结果文件

- `artifacts/fit-head-energy/report.json`：固定样本拟合。
- `artifacts/calm-investigation-decoder.json`：VAE/预测分布及首块对齐诊断。
- `artifacts/eval-repair-energy-original/report.json`：原方法统一评估。
- `artifacts/eval-repair-flow16/report.json`：条件流统一评估。
- `artifacts/eval-repair-token/report.json`：有效基线指标及全部12条实际续写。
- `artifacts/token-context-diagnostic.json`：上下文消融、梯度和参数变化。

新增实验代码位于`experiments/train_calm_repair.py`、`latent_ar/conditional_flow.py`、`latent_ar/calm_token.py`；可复用诊断入口见对应同名实验脚本。
