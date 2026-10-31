# 滑动窗口反馈学生

## 目的

原学生只用当前主干状态h，经约158万参数MLP预测实际文字的encoder回灌向量。第一轮单步MSE有所下降，但连续生成没有保持住教师输出。

新实验让学生读取当前h及同一轨迹前两块h，用小型因果attention预测相同的回灌目标。输入不依赖当前块解码完成，仍可以用于研究提前预测反馈。

对固定greedy教师，理论上当前h已经决定教师输出。历史窗口提供的是更容易利用的显式历史，并不保证能消除预测误差；是否有效以闭环实测为准。

## 窗口定义

预测第t块反馈时输入 `[h_(t-2), h_(t-1), h_t]`，只包含当前和过去。轨迹开始处左侧补空位并屏蔽，不借用其他故事、不读取未来h。

正式窗口包含prompt预填充时的过去h：预测第一块反馈时，当前h前面已经可以有两个prompt状态。短prompt不够两块时才左侧padding。推理时复用主干已有的prefill输出，之后每次生成新主干状态再更新窗口；缓存不跨prompt调用保留。

首个试跑仅积累生成阶段的h，第一块只有当前h。这个版本保留为generated-only诊断，检查点的 `use_prompt_history` 默认为false；含prompt历史的新版本显式设为true，不会悄悄改变旧检查点的推理输入。

旧采集数据生成64token，仅保存会继续推进的完整、非EOS块，因此监督位置最多为0、1、2。只使用生成阶段历史时，它足以覆盖前两块，无法完整覆盖前三块。正式窗口会额外重算并保存prompt的前序状态，不需要重新采集或改变既有h/r标签。

## 对照

- 原MLP结果保留，不覆盖。
- 新attention结构，历史块数0：只有当前h。
- 同样attention结构，历史块数2：当前h与前两块h。
- 额外保留“包含prompt历史”的窗口2版本，检验第一次回灌起就提供历史的情况。
- 两组使用同一冻结greedy教师、原train/dev轨迹、seed、batch和训练步数，分别选dev MSE最低的检查点，再比较32/64token闭环。

attention与MLP结构不同，因此只有attention的0/2历史对照可以直接检验增加窗口的作用。两组位置编码参数可能随窗口长度有少量差别，应同时报告准确参数数目。

## 结构和运行

输入/输出均为512维；attention内部宽256、2层、4头，前馈层宽1024。只读窗口最右端的输出，窗口本身截止当前块。历史块数2时有1,844,736参数，历史块数0时有1,844,224参数。padding由独立有效掩码屏蔽。

以下为本轮已经执行的训练命令；已有结果目录会拒绝覆盖：

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.distill_gru_feedback train `
  --data runs/feedback-distill-greedy/data `
  --run-dir runs/feedback-attention-h2/student `
  --student-type attention --history-blocks 2 `
  --steps 3000 --batch-size 256 --device cuda
```

对照把历史块数改为0，结果目录改为 `runs/feedback-attention-h0/student`。每100步记录dev指标、保存last，并保留dev MSE最好的best。继续训练需带 `--resume`，学生类型、历史窗口、设备类型和batch保持一致，`--steps`指定续训后的总步数。

正式含prompt历史的数据先由以下命令生成新目录；原目录保持不变：

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.distill_gru_feedback add-prompt-history `
  --data runs/feedback-distill-greedy/data `
  --output runs/feedback-attention-prefix/data `
  --history-blocks 2 --batch-size 32 --device cuda
```

随后训练命令的数据目录换成 `runs/feedback-attention-prefix/data`，结果目录换成 `runs/feedback-attention-prefix-h2/student`。新目录保持相同teacher快照、prompt、h/r及划分，只增加历史特征并更新文件哈希与溯源信息。

闭环使用同一命令入口，自动识别旧MLP和新attention检查点：

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.distill_gru_feedback evaluate `
  --data runs/feedback-distill-greedy/data `
  --student runs/feedback-attention-h2/student/best.pt `
  --output artifacts/feedback-attention-h2-64.json `
  --prompts 128 --max-new-tokens 64 --device cuda
```

结果与局限见 [本轮实测报告](../artifacts/feedback-attention-results.md)。

## 验证边界

检查窗口内因果性、跨故事隔离、左padding、截断、检查点恢复和混合EOS。质量指标继续包括单步dev MSE/均值基线、共同前缀、首块一致、两种长度归一化的token匹配、EOS/长度、学生访问状态反馈误差，以及实际文字。

首块一致是实现检查，因为首块尚未使用预测反馈。token匹配不是语义质量评分。当前评估仍串行保留真实解码及反馈作为诊断，不宣称异步加速。
