# 离线工具与对照

从项目根目录使用 `python -m experiments.<模块名>`，不要直接切换到本目录运行脚本。

- `diagnose_backbone_learning`：主干权重回退、上下文抹除和错配。
- `evaluate_continuous_prior`：固定 dev 上的 teacher/free 对照和全部自由文本。
- `evaluate_rollout`：逐块真实/自由回灌漂移。
- `evaluate_text_distribution`：文字 n-gram 分布评分。
- `experiment_prior_losses`、`train_energy_text`：历史损失对照。
- `token_ar_baseline`：普通逐 token LM 基线。
- `compare_latent_generation`、`diagnose_prior`、`benchmark_batch`：表示/性能诊断。
- `restore_artifact`：从本机 Git/LFS 快照恢复旧实验文件，验证哈希且不覆盖现有文件。

当前模型训练仍使用根目录 `train_prior.py`；默认配置是小输出头。
