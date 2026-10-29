# 原始 latent 与均值 latent 的短文本生成对比

检查点：step 101；1 个 prompt × 1 个随机种子；最多 16 token。

均值定义：对每个 prompt 的完整生成 latent 序列取时间均值，再重复到全部生成位置。保留原 prompt latent。两边采样设置和随机种子完全相同；不强制最低生成长度。

这项离线诊断比较逐步变化的 latent 与其均值；均值仍包含该 prompt 的信息，不能等同于没有 latent，也不能据此单独判断主干的质量收益。

完全相同的输出：0 / 1。

第一个 token 即不同：1 / 1；平均续写长度：原始 16.00 token，均值 16.00 token。

## Prompt 1 / seed 42

**输入**

Once upon a time, a little girl found a strange box.

**原始 latent**

 The could always alwaysy smiled away enough t t come P. ". it

**均值 latent**

y N It alwaysy N It alwaysy N It alwaysy N It always

相同前缀：0 token；输出长度：16 / 16 token。
