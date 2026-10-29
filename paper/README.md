# 论文材料索引

## 当前版本与事实记录

- 方法：[docs/method.md](../docs/method.md)。主干按四 token 块推进，小输出头内部含离散选择；不是纯连续生成，也没有已验证的异步加速。
- 当前模型：本地 `runs/main/last.pt`，已恢复第 3100 步；固定备份 [main-step3100.pt](../artifacts/checkpoints/main-step3100.pt)。
- 最新已有验证记录是 step 3000：dev NLL/token **2.802499**，21435 个有效 token，5485 个块；错配上下文为 **6.197730**。第 3100 步只有训练记录，不能冒充同一步的验证结果。
- [环境](environment.json)、[数据元信息](data/metadata.json)、[数据来源与校验](data/source.json)、[数据卡](data/DATASET_CARD.md)。
- [继续训练和复查命令](../docs/reproduce.md)。

## 结果证据

`experiments/runs/` 保存整理前全部轻量结果，包括配置、指标、自由续写、诊断和实验报告；去掉每个实验重复的 tokenizer 副本，原词表保留在 `data/prepared/tokenizer.json`。新训练输出仍写入根目录 `runs/`，不会自动改写论文证据快照。

建议从以下记录取数：

| 论文内容 | 证据 |
|---|---|
| 当前训练曲线 | [main/metrics.jsonl](experiments/runs/main/metrics.jsonl) |
| 主干/上下文干预 | [block-head-verified-backbone.json](experiments/runs/block-head-verified-backbone.json) |
| step 1000 完整自由续写 | [block-head-verified-evaluation.json](experiments/runs/block-head-verified-evaluation.json) |
| 固定 latent MSE 与去噪对照 | [continuous-prior-comparison-fp32.json](experiments/runs/continuous-prior-comparison-fp32.json) |
| Energy、CE、块混合目标消融 | [objective-ablation-v1/report.json](experiments/runs/objective-ablation-v1/report.json) |
| 一致性约束消融 | [objective-cycle-ablation-v1/report.json](experiments/runs/objective-cycle-ablation-v1/report.json) |
| 小头早期训练对照 | [token-head-ablation-v1/report.json](experiments/runs/token-head-ablation-v1/report.json) |
| Codec 还原与噪声鲁棒性 | [codec-v3-noise-robustness.json](experiments/runs/codec-v3-noise-robustness.json) |

历史研究笔记、设计、阶段说明和旧配置位于 `archive/`；它们是历史材料，不代表当前默认方法。原始路径可通过 Git 快照和 [artifacts.json](artifacts.json) 追溯。

## 论文中需要区分的结论

1. MSE 点回归会趋向条件均值。本项目已观察到 loss 下降而自由生成失败，不能将其作为生成成功。
2. 去噪 loss 的输入含带噪目标，不能与纯历史预测 MSE 直接排名。
3. 输出头的联合 NLL、单次采样的参考 CE、本地 LM 流畅性代理不是同一指标；本地代理也不是语义质量评测。
4. 训练预算、初始化及验证批数在部分实验间不同。绘制曲线/表格时须读取相应 config、tokens 和 blocks，不能把它们混成严格同预算比较。
5. 尚缺正式未参与调参的 test 评价、多次种子训练、统一预算基线、端到端性能及真实异步调度实验。当前证据支持原型有效性，不能写成完整质量或速度结论。

## 文件保留策略

活动目录保留当前模型、codec、关键 latent 对照、主干初始化来源和本地语言评估基线。其余试运行、旧权重和缓存已移出工作目录；研究权重/数据仍可从 Git/LFS 历史恢复，清单见 [artifacts.json](artifacts.json)。清理记录见 [cleanup.json](cleanup.json)。

整理后的全套 **91 项测试通过**。其中模拟了 Windows 短暂读锁和持续锁定，验证训练可另存完整版本并继续；第 3100 步的严格模型/优化器加载、数据哈希及备份一致性见 [recovery-verification.json](recovery-verification.json)，没有执行主模型参数更新。归档恢复工具也已从本地 LFS 恢复一份 offsets 文件并验证 SHA-256 一致。
