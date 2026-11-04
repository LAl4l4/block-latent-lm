"""阶段 2：倒序错开解码（reverse-staggered decoding）。说明见 stage2/README.md。

  schedules.py   可见关系：每个调度下"谁能看到谁"（visibility、计划码时刻、采样轮次）
  models.py      双流 Transformer（TwoStreamLM）与带块级计划的 PlanLM
  mask/          各调度只差 attention mask：train、sample_eval、examples、summarize
  plan/          每块加离散计划码：train、samples
"""
