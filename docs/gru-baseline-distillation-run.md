# GRU基线与第一轮反馈蒸馏执行说明

约束固定：16-token块，小GRU没有attention；每块hidden由主干上下文h重新初始化。GRU的块内状态不传到下一块。跨块信息只由主干和实际文本压缩回灌提供。块间隔离和固定噪声采样已测试。

当前基线：`runs/calm-repair-token`，从既有2000步检查点继续到10000步。主干与GRU联合文字CE训练，VAE冻结，batch128×2，每100步保存。它继承早期Energy主干，来源与训练预算记录在原实验配置；不是从零训练的干净规模对照。之后学生与精确反馈基线使用同一个固定教师，不混用教师版本。

第一轮学生采用greedy教师，目标为C(h)≈G(D(VAEmean(GRU_greedy(h))))。G是可训练后已冻结的历史压缩器。学生输入h，而不是GRU选词之后的z；随机教师的共享噪声版本已有模型接口，但本轮先不使用。

## 基线完成后的顺序

确认检查点step10000、最后dev记录及续写均已生成，再冻结教师采集轨迹：

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.distill_gru_feedback collect `
  --teacher runs/calm-repair-token/last.pt `
  --output runs/feedback-distill-greedy/data `
  --train-prompts 4096 --dev-prompts 512 --batch-size 32 --device cuda
```

采集保留独立train/dev故事索引和裁窗来源；只用会继续推进的完整非EOS块。冻结教师权重快照、词表和哈希一并保存。实际有效训练块数取决于教师生成长度，应在报告中写明，不能把prompt数当作有效样本数。

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.distill_gru_feedback train `
  --data runs/feedback-distill-greedy/data `
  --run-dir runs/feedback-distill-greedy/student `
  --steps 3000 --batch-size 256 --device cuda
```

512宽、两层残差MLP拟合反馈MSE，记录train/dev误差、cosine和训练集平均反馈基线；每100步自动保存best/last、优化器及随机状态。若训练提前退出，可带`--resume .../last.pt`恢复，`--steps`仍表示最终总步数。

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.distill_gru_feedback evaluate `
  --data runs/feedback-distill-greedy/data `
  --student runs/feedback-distill-greedy/student/best.pt `
  --output artifacts/feedback-distill-greedy-64.json `
  --prompts 128 --max-new-tokens 64 --device cuda
```

同时运行32和64 token评估，分别保存新文件。128-token闭环属于超过原64目标长度的外推，应单独标记，第一轮不据此外推失败作主要结论。

## 判断标准与边界

先看固定教师的验证文字CE、上下文消融和实际文字；其质量仍可能有限。学生的单步dev MSE需与均值基线比较，不能只报训练误差。闭环报告教师/学生实际输出、长度/EOS、共同前缀及两个长度分母下的token一致率，以及学生访问历史中的反馈误差。首块理论上应一致，因为还没有使用学生反馈。

当前闭环实现仍按顺序计算GRU分支及学生，保留真实反馈作诊断，**不是异步实现，不可据此宣称加速**。通过质量检查后再实现并行调度，最终计时必须包含实际文字全部完成。

代码的CPU端到端检查已跑通collect→train→evaluate；仅使用4个训练故事、4个dev故事和2步学生训练，证明接口可执行，不是蒸馏效果证据。正式蒸馏要等待10000步教师冻结后执行。
