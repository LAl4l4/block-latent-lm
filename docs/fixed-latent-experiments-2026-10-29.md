# 固定 latent 目标与后续改法：实测结论

本轮按用户要求，先试确定性下一 latent 回归，训练不使用 Energy。随后试连续去噪；两者未解决自由生成后，再核验并延长项目中的块内小头对照。所有实验使用同一冻结 codec 和 train/dev 数据，未使用 test，没有覆盖原 Energy 检查点。

## 1. 严格按固定 latent 目标训练

配置：`configs/prior-latent-mse.json`；检查点：`runs/prior-latent-mse-check/last.pt`，1000 步。

- 输入：真实前序块的冻结 codec latent。
- 输出：确定性的下一 latent，没有随机生成头。
- 目标：真实下一块的标准化 codec latent，按有效块平均 MSE。
- `latent_mse_weight=1`、`token_ce_weight=0`。CE 在 no-grad 下仅作诊断，decoder 不参与更新。
- 继承旧 step 1404 主干，线性输出头和优化器重新开始，128×2，学习率 0.0003。

训练 MSE 从 1.1098 降到 0.9143。固定 128 个 dev 窗口：

| step | latent MSE | 参考 token CE（不训练） |
|---|---:|---:|
| 250 | 0.9352 | 7.8792 |
| 500 | 0.9303 | 7.8834 |
| 750 | 0.9296 | 7.9167 |
| 1000 | 0.9260 | 7.9329 |

另一组固定 32 个 dev 窗口、FP32 的完整预测检查：零向量基线 MSE 1.02069，预测 MSE 0.92692，错配历史后 1.10963。主干利用了条件，预测比零向量改善约 9.2%，但自由文本仍杂乱且重复。

同批真实 latent 的标准化 RMS 为 1.0103，预测只有 0.3204。该结果与点回归的平均化现象一致：平方误差的最优点预测是条件均值，不保证落在某个具体可解码文字块附近。这里不把“loss 能下降”误报为“已经能生成”。

记录：`runs/latent-mse-evaluation.json`、`runs/latent-mse-rms.json`。

复查与继续：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-latent-mse-check/last.pt --eval-only --eval-batches 1
```

## 2. 连续去噪回归：有局部收益，仍未解决

实现：`latent_ar/denoising.py`；配置：`configs/prior-denoising.json`。继承前述 MSE 主干，新增小去噪头，真实 latent 仍为固定监督目标，不使用 Energy、没有 token 回灌。

训练输入为 `(1-t)*noise+t*target`，预测 clean target，用 MSE 更新主干与头。生成时从独立高斯噪声开始，固定主干条件后迭代小头 32 次，得到一个 latent 再回灌。主干不读取当前真实目标进行推理。该方案增加了小头推理次数，不能宣称仍是单步头。

验证去噪 MSE：250 步 0.4441，500 步 0.4114，750 步 0.3946，1000 步 0.3840。最新保留检查点为 step 1100：`runs/prior-denoising-check/last.pt`。

**去噪验证能看到带噪的当前目标，不能将 0.384 与普通下一 latent 预测 MSE 直接比较。** 独立噪声自由生成的本地 LM 流畅性代理分数有所改善，但整体文字仍不连贯，因此暂不推荐作为已解决生成质量的方案。

这一方向参考 [MAR / Diffusion Loss](https://arxiv.org/html/2406.11838v1) 将连续输出分布的建模放在小去噪头中的做法；这里是本项目的 clean-latent 回归变体，不是论文原样复现，图像实验也不证明本文本模型会成功。

## 3. 已确认主干有效学习的方案：块内离散小头

实现：`latent_ar/token_head.py`；配置：`configs/prior-token-head.json`。这是项目已有的结构对照，本轮重新核验，并从 step 601 继续训练至 1000，保存到独立目录 `runs/prior-block-head-verified/`。

路径明确如下：

1. 主干读取历史 latent，每块运行一次，输出条件状态。
2. 小 GRU 头在内部按顺序选择四个词。
3. 冻结 codec encoder 把这四个词编码为 latent，立即回灌主干。
4. 冻结 decoder 从该 latent 还原文字；回灌环不等待它的输出。

因此它保留块级主干和 latent 接口，但**包含块内离散词选择，目标改为小头的联合文字 CE，不再是完全连续的固定 latent 回归**。与普通逐 token 主干相比，主干仍四 token 推进一次；尚未实现或验证实际异步加速。

固定 128 个 dev 窗口的 NLL/token：step 601 为 3.3674，750 为 3.2807，1000 为 3.1614。训练目标没有 Energy，codec 冻结。

为区分“主干学了”和“小头独自学会局部语言”，固定 step 1000 输出头、codec、数据和协议，进行干预：

| 主干条件 | dev NLL/token |
|---|---:|
| 训练后的主干 | **3.1614** |
| 换回原 step 1404 的主干权重 | 4.2397 |
| 抹去主干输出，仅保留小头的块内条件 | 5.3635 |
| 将训练后上下文在故事之间错配 | 5.7891 |

这确认输出依赖主干的更新与正确历史。恢复初始权重也改变了头与主干的配合，因此它不是“同预算冻结主干重训”的严格消融，不据此推断全部收益来源。

干预记录：`runs/block-head-verified-backbone.json`。可独立复查：

```powershell
& D:\Code\venv313\Scripts\python.exe diagnose_backbone_learning.py `
  --checkpoint runs/prior-block-head-verified/last.pt `
  --initial runs/prior-energy-v4-check/last.pt `
  --output runs/block-head-recheck.json
```

## 4. 自由生成核验与限制

固定另一组 32 个 dev prompt，seeds 42/43/44，每条最多 64 token，FP32 推理；使用已经训练的 7.2M token AR 模型只作本地流畅性代理，分数越低越好：

| 方法 | 自由文本本地 LM CE | 重复 trigram 比例 |
|---|---:|---:|
| 确定性 latent MSE，step 1000 | 9.118 | 16.61% |
| 连续去噪，step 1100 | 7.466 | 0.35% |
| 块内小头，step 1000 | **2.608** | **1.39%** |

三者初始化和训练预算不同，这是诊断对照，不是公平架构排名。低代理分数不是语义质量保证。全部未筛选文本分别保留在 `runs/continuous-prior-comparison-fp32.json` 和 `runs/block-head-verified-evaluation.json`。块内小头生成已出现完整句子，但仍有语义跳跃、重复和指代问题，尚未达到高质量故事生成。

所有分支从历史及独立噪声开始预测；未将真实当前 target 或其带噪版本传进自由生成。FP32 的真实/自主首块最大差在约 1e-6–6e-6，符合数值误差。BF16 下较小的概率变化可能触发不同离散采样结果，因此不要求其随机文字逐元素相同。

## 5. 直接使用已验证检查点

```powershell
& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/prior-block-head-verified/last.pt `
  --prompt "Once upon a time, a little girl found a strange box." `
  --max-new-tokens 64 --seed 42 --show-info
```

继续训练同一块级主干模型：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/prior-block-head-verified/last.pt
```

如果必须保留“纯连续、单次确定性输出、目标仅为真实 latent”这三个条件，目前本轮只证明了有限的预测学习，没有证明可用生成。块内小头是经过验证的替代取舍，不能当成原来三个条件全部满足。

新训练默认配置已改为纯 latent MSE；原 Energy 配置另存为 `configs/prior-energy-v4.json`。最终全套 87 项测试通过（Python 3.13 / PyTorch 2.11.0+cu128，16.79 秒），包括新阶段的中断恢复逐张量一致、冻结与梯度，以及 decoder 不进入块内小头回灌环的验证。
