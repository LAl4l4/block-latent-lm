# 当前方法：块级 latent 主干与块内离散输出头

四个 token 经固定 codec encoder 编码为一个 512 维 latent。8 层、宽 512 的因果 Transformer 读取历史 latent，给出下一块的条件状态。宽 256 的小 GRU 输出头在块内顺序选择四个 token，随后用同一个冻结 encoder 将其编码为下一步回灌的 latent。冻结 decoder 只把 latent 还原成文字，主干回灌不读取 decoder 输出。

这包含块内离散采样，训练目标为四 token 联合文字 CE，不是纯连续 latent 回归。主干每块推进一次，不等于系统已实现异步并行：目前参考实现仍为同步循环，没有正式速度收益结论。

训练使用真实前序 latent；当前块的小头只能读取先前槽位的真实 token，不能读取当前或未来槽位。生成时块内使用自身选出的 token。Padding 不参加损失，全部 codec 参数冻结。总参数 34,161,408，可训练 29,895,168。

数据来源、划分和词表校验见 [论文数据记录](../paper/data/metadata.json)。实际训练环境见 [environment.json](../paper/environment.json)。

核心实现：`latent_ar/token_head.py`、`latent_ar/staged.py`、`latent_ar/staged_training.py`。其他生成头保留在包内用于加载和复现论文对照，不作为当前默认训练方案。

已有证据与局限见 [论文材料索引](../paper/README.md)。
