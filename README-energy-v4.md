# Latent AR：单步随机生成头 V4

后续实测与建议：多组损失和反馈接口对照见 [2026-10-29 测试报告](docs/training-ablation-results-2026-10-29.md)。当前文字效果最好的是独立 `token_head` 基线，配置 `configs/prior-token-head.json`；它含离散内容采样，尚不是纯连续方案。默认 Energy 配置保留作研究对照。

2026-10-29 的 [loss 与生成质量实测](docs/loss-investigation-2026-10-29.md)已补充：原 Energy 目标的诊断、两个没有通过完整质量检验的新 loss 对照，以及实际训练到 2000 步的普通 token AR 基线（固定 dev CE 9.0864 → 2.7451）。原实验和默认配置保留；这不代表纯 latent 生成质量已经解决。

当前方案：冻结已训练的四 token codec，主干读取真实历史 latent，用带随机输入的小生成头预测下一块的连续分布。**训练主干只使用 Energy loss；decoder 保持一层块内双向注意力，负责将选定 latent 还原成四个 token。**

设计见 [DESIGN.md](DESIGN.md)，变更见 [CHANGELOG.md](CHANGELOG.md)，论文依据见 [研究笔记](docs/latent-generative-models-review.md)。旧确定性 V3 的说明已归档到 [V3 用法](docs/usage-v3.md) 和 [V3 设计](docs/architecture-v3.md)。

## 1. 直接训练

本机已有 `runs/codec-v3/last.pt`，不需要重训 codec。所有命令在项目目录执行，统一使用 Python 3.13 环境。

继承旧 prior 已训练的主干，重新训练随机生成头和优化器：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --init-prior runs/prior-v3/last.pt
```

默认输出 `runs/prior-energy-v4/`。程序会检查旧 prior 的数据、冻结 codec、主干结构是否匹配，只复制主干和 latent 归一化统计。新实验从 step 1 开始，学习率重新 warmup；旧线性头和旧优化器不继承。

想让主干也从头开始，直接运行：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py
```

两种启动方式请分别用新的 `--run-dir`，已有非空实验目录需要通过 `--resume` 继续。

默认 **batch 128 × 累计 2 = 每步 256 个故事窗口**，每个窗口最多 128 个 prompt token、64 个目标 token。每块取 **8 个候选**，只重复小生成头，不把故事 batch 再乘 8。主干每个 microbatch 只前向一次。

计划 10,000 步，前 500 步 warmup；**每 100 步自动保存**，每 250 步验证。配置在 `configs/prior.json`。

## 2. 暂停与继续

按一次 `Ctrl+C`，程序完成当前参数更新、保存后退出。恢复完整实验：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-energy-v4/last.pt
```

本次已完成的 101 步 GPU 短训在独立目录，继承的是旧 prior step 2202 的主干。也可以直接接着它训练：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-energy-v4-check/last.pt `
  --eval-batches 4
```

两个命令选择一个实验继续即可。`--resume` 恢复其原架构、配置、优化器、数据和随机状态；恢复旧 V3 检查点会继续旧 MSE+CE 目标。切换到 Energy 必须用新实验或 `--init-prior`，不能把旧断点直接当成新输出头续训。

## 3. 损失与日志

```text
真实历史 latent → 因果主干 → h（只计算一次）
                               ├ + ε1 → 小生成头 → z1
                               ├ + ε2 → 小生成头 → z2
                               └ ... 共 8 个独立候选

Energy = 平均“候选到目标”的距离 - 0.5 × 平均“不同候选之间”的距离
```

距离使用欧氏距离，再除以 `sqrt(latent_dim)`，不使用平方距离。训练按整个累计 batch 的有效块数平均，padding 不参与。

- `Energy`：实际优化目标，越低越好；只在相同 codec、标准化和评价协议下比较。
- `spread`：同一上下文下不同候选的平均距离，观察模型是否忽略随机输入；不单独追求越大越好。
- `样本 CE`：第一个候选经冻结 decoder 后与参考文字的 CE，**仅诊断，不反传**。它不是隐式分布模型的完整似然，也不能当作普通 LM 的 PPL。
- 验证新增 `块混合 NLL`：先在每个候选内计算整块文字概率，再平均候选概率、取负对数。它比单候选 CE 更贴近分布诊断，但只有 8 个候选，估计有偏；对应 softmax decoder，不是贪心生成的精确似然，仍不能代替实际续写评价。
- `条件 Energy 优势`：将上下文错配后的 Energy 减去正确条件的 Energy；两条分支使用相同噪声。正值说明该批上正确上下文有帮助，不等于续写质量达标。

训练没有 latent MSE，也没有把所有候选分别拉向同一参考文字的 CE。冻结 codec 依然提供真实 latent 目标，decoder 不参与主干的 Energy 优化。

完整模型 **33,966,336 参数**，其中可训练主干和生成头 **29,700,096**；冻结 codec **4,266,240**。

## 4. 生成与回灌检查

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/prior-energy-v4/last.pt `
  --prompt "Once upon a time, a little girl found a strange box." `
  --max-new-tokens 64 --seed 42 --show-info
```

默认 `--temperature 0` 是贪心选词；**latent 本身仍然随机**，由 `--seed` 控制。同一 checkpoint、设备与设置下，同一种子可复查输出；更换种子会改变 latent 抽样。词采样和 latent 采样使用分开的随机数流。

推理每块只抽一个 latent，一次小生成头前向完成。这个 latent 一路回灌主干，另一路交给 decoder；不等待解码文字回灌。当前参考实现仍是同步执行，尚未实现异步队列。

```powershell
& D:\Code\venv313\Scripts\python.exe evaluate_rollout.py `
  --checkpoint runs/prior-energy-v4/last.pt `
  --output runs/rollout-energy-v4
```

报告比较真实历史分支和自主回灌分支，两者使用相同的逐位置随机输入；第一块预测应一致。真实历史分支只作诊断，不作为真实生成结果。先检查 64 token 内的多个种子，向量距离不是语义质量分数。

原始 latent / 时间均值对照继续可用：

```powershell
& D:\Code\venv313\Scripts\python.exe compare_latent_generation.py `
  --checkpoint runs/prior-energy-v4/last.pt `
  --output runs/energy-v4-latent-comparison
```

该脚本为每个 prompt 固定一份 latent 抽样，再改变 decoder 的随机种子，结果是离线扰动诊断。

## 5. Codec、旧实验与数据

只有尚无 codec 时才需要先训练重建器：

```powershell
& D:\Code\venv313\Scripts\python.exe train_codec.py
```

codec 仍使用 `configs/codec.json`，默认 `runs/codec-v3/`，每步 256 个至多四 token 的小块。进入主干训练前重新检查固定 2,048 个 dev 块，要求 token 还原率 ≥99%、完整四 token 全对率 ≥95%。本机正式 codec 已通过，不必重训。

保留历史 V3 的确定性对照配置 `configs/prior-ar-v3.json`；V1/V2/V3 的生成按原架构严格加载。实验数据和检查点不迁移、不覆盖。

沿用 100,000 篇 TinyStories（21,997,870 个含 EOS token），dev/test 各 10,995 篇，8,192 词表仅用训练子集拟合。完整原始 Parquet 和准备数据保留在 `data/raw/`、`data/prepared/`；下载与准备说明见 [V3 用法](docs/usage-v3.md)。

## 6. 验证与代码阅读

61 项测试通过，包括 Energy 公式、双峰分布学习、噪声作用、冻结与梯度、纯 latent 回灌、相同噪声的首块一致性、随机状态恢复和旧架构兼容。GPU 以 128×2 完成 100 步并恢复到第 101 步；只确认训练运行和初步损失变化，尚无正式语言质量结论。详细数值见 [VALIDATION.md](VALIDATION.md)。

| 文件 | 内容 |
|---|---|
| `latent_ar/energy.py` | 条件随机 MLP、Energy loss、候选与采样 |
| `latent_ar/staged.py` | 共用真实历史构造、KV cache 回灌、生成与 EOS |
| `latent_ar/staged_training.py` | 两阶段训练、质量检查、主干初始化、日志与保存 |
| `configs/prior.json` | 默认 Energy 训练配置 |
| `evaluate_rollout.py` | 固定噪声的真实历史 / 自主回灌对照 |

```powershell
$env:PYTHONPATH = Join-Path $PWD '.tmp/dev-tools'
& D:\Code\venv313\Scripts\python.exe -m pytest tests -q `
  --basetemp .tmp/pytest-manual-energy
```

pytest 会清空 `--basetemp` 指定的专用临时目录。

后续实测：step 1404 的全语料训练仍没有产生可用续写；应记录为真实语料尚未跑通，不能把 Energy 下降当作语言质量。隔离的 4 个固定真实训练窗口可以很快被记住，这不代表泛化成功。详见 VALIDATION 的后续诊断。
