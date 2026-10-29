# V3 自主 latent 回灌检查

检查点 step 3；codec 在本批真实目标上的还原准确率 93.75%。

真实历史分支读取正确的前序 target latent，仅用于诊断；自主分支只读取 prompt 与自身采样的 latent。差异不等于生成质量下降，需人工检查下列短续写。

全部自主 latent 为有限值：True。

## 样本 1 / seed 42

**输入**

 so excited to have her very own library card. She picked out lots of books and brought them home to read. As she read, she noticed that some of the stories had harsh words in them. She asked her mommy why some stories were mean.

Mommy explained, "Sometimes, stories have harsh words

**真实 latent 历史（非实际推理）**

 said kept would store be Suddenlyâ egg

**自主回灌**

 said kept would store be feltle world

## 样本 2 / seed 42

**输入**

 to help. He walked over to the flower and picked it for her. Lily was so happy and hugged her daddy tightly.


**真实 latent 历史（非实际推理）**

 Lily thin medicine car while this it watched

**自主回灌**

 Lily thin medicine car friendship eyes it for
