"""块级并行解码中"衔接处"的研究代码（阶段 2 和阶段 3）。

公共部分
  common.py        Transformer 组件、混合精度、JSONL 日志、安全中断
  schedules.py     可见关系：每个调度下"谁能看到谁"（visibility、计划码时刻、采样轮次、窗口）
  models.py        TwoStreamLM、PlanLM、LatentLM、WindowLM
  data.py          定长窗口数据

阶段 2：倒序错开解码（论文 paper/reverse-staggered）
  mask/            各调度只差 attention mask：train、sample_eval（采样 + causal 评委）、
                   examples（论文附录样例）、summarize（多 seed 汇总）
  plan/            每块加离散计划码：train、samples

阶段 3：三级解码
  latent/          连续块 latent：train、samples、one_block（真实上下文里重采样一块）
  window/          强 z + 小窗口解码器：train、samples

运行方式：python -m seam.mask.train ...；说明见 docs/stage2/idea.md 和 docs/stage3/idea.md。
"""
