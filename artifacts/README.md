# 固定研究制品

`checkpoints/main-step3100.pt` 是恢复后的完整第 3100 步不可变备份，包含模型、优化器及随机状态，由 Git LFS 保存。

实际继续训练用 `runs/main/last.pt`。不要把这个归档目录作为训练输出目录，避免 Git 状态检查读取正在更新的权重。旧实验制品的索引位于 `paper/artifacts.json`。
