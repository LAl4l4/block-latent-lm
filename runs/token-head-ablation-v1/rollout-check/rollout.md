# 自主 latent 回灌检查

检查点 step 601；codec 在本批真实目标上的还原准确率 98.11%。

真实历史分支读取正确的前序 target latent，仅用于诊断；自主分支只读取 prompt 与自身预测的 latent。差异不等于生成质量下降，需人工检查下列短续写。

全部自主 latent 为有限值：True。

## 样本 1

**输入**

 so excited to have her very own library card. She picked out lots of books and brought them home to read. As she read, she noticed that some of the stories had harsh words in them. She asked her mommy why some stories were mean.

Mommy explained, "Sometimes, stories have harsh words

**真实 latent 历史（非实际推理）**

 it is very proud things when we can will try to be make more careful when

**自主回灌**

 it is very proud of himself."

Timmy felt so happy that she would

## 样本 2

**输入**

 to help. He walked over to the flower and picked it for her. Lily was so happy and hugged her daddy tightly.


**真实 latent 历史（非实际推理）**


After a few they heard a big bird's dark. and she was a

**自主回灌**


After a few minutes, Lily's dad went down and took a little girl

## 样本 3

**输入**

 didn't like him anymore. 

But then, Cow said, "I'm sorry, Oink. I didn't mean it. I just don't like getting dirty." Oink forgave C

**真实 latent 历史（非实际推理）**

 smiled and said, to play together. They were very happy day on, Spot

**自主回灌**

 smiled and said, "I'm glad you too. I might be careful with

## 样本 4

**输入**

 face. The boy stops laughing. He starts crying.

"Ow, ow, ow!" he says. He runs to his mom.

Ben and Lily smile. They are brave. They are friends. They go back to the slide.

**真实 latent 历史（非实际推理）**

（EOS）

**自主回灌**

（EOS）
