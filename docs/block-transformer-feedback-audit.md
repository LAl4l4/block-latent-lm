# Block Transformer：反馈预测到底做到了哪一步

核对日期：2026-10-31。核对NeurIPS 2024正式版§4.2、arXiv v2和官方公开实现。

## 已有的直接证据

正式版§4.2确实讨论了预测下一输入块embedding，让token解码器并行处理多个块；并报告用MSE或对比损失会降低性能，同时指出连续块表示的误差累积问题。该段没有给出这种预测式生成的完整质量、误差、接受率及端到端速度对照。

因此不能说作者完全没做：原文报告了负面尝试，代码也有相应损失。但也不能把它描述成已经完成了稳定、保质的预测反馈并行生成系统。

来源：[正式论文，第9页§4.2](https://proceedings.neurips.cc/paper_files/paper/2024/file/579b5b84311e122584dedab1d3b7613f-Paper-Conference.pdf#page=9)。

## 代码证据

- [`model/block_decoder/base.py`](https://github.com/itsnamgyu/block-transformer/blob/main/model/block_decoder/base.py#L24)：`compute_loss`将主干输出与后移一块的输入embedding对齐；支持FP32 MSE和归一化相似度对比损失，目标分支detach。
- [`pretrain_block_transformer.py`](https://github.com/itsnamgyu/block-transformer/blob/main/pretrain_block_transformer.py#L36)：训练入口确实连接了块预测损失开关和权重。
- [`conf/trainer/block_main_b4_5.yaml`](https://github.com/itsnamgyu/block-transformer/blob/main/conf/trainer/block_main_b4_5.yaml#L58)：所核对主配置开启token损失，关闭block预测损失。
- [`model/block_transformer.py`](https://github.com/itsnamgyu/block-transformer/blob/main/model/block_transformer.py#L332)：公开生成循环先调用token decoder产生当前块，再把实际生成token经embedder编码，作为下一轮主干输入。这里没有用预测反馈跳过这项依赖。

论文的10–20倍吞吐量结果不能当成“预测反馈流水线”的加速结果。公开主生成路径仍依赖实际token回灌。

## 与本项目的区别

他们的公开实现可在主干上加入块embedding预测损失；本项目当前实验则冻结B、D、E，单独训练外部P来拟合E(D(h))。两者的梯度路径和目标变化风险不同。

因此他们的负面报告是相关风险的证据，不能直接证明本项目的MSE必然是唯一问题，也不能证明本项目学生的单步预测已经准确。

本项目最新的“两块一组、组末用真实反馈从干净cache重放”的调度，还需要独立测量质量与延迟。所查这篇论文和上述公开代码没有给出同样协议的完整结果；这不等于已证明整个领域没有其他先例。
