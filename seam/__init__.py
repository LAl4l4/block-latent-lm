"""倒序错开解码（reverse-staggered decoding）的调度代价研究。

  schedules.py     可见关系：每个调度下"谁能看到谁"（visibility、计划码时刻、采样轮次）
  models.py        双流 Transformer（TwoStreamLM）与带块级计划的 PlanLM
  data.py          定长窗口数据
  train_mask.py    mask 实验：各调度只差 attention mask，同预算训练
  train_plan.py    计划实验：每块加 K 个离散计划码
  sample_eval.py   按各调度自身顺序采样，由独立 causal 评委打分
  plan_samples.py  计划模型采样与按块打印
  summarize.py     汇总多 seed 结果

想法与结果见 docs/idea-reverse-staggered-decoding.md。
"""
