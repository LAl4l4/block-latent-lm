# 自主 latent 回灌检查

检查点 step 601；codec 在本批真实目标上的还原准确率 98.11%。

真实历史分支读取正确的前序 target latent，仅用于诊断；自主分支只读取 prompt 与自身预测的 latent。差异不等于生成质量下降，需人工检查下列短续写。

全部自主 latent 为有限值：True。

## 样本 1

**输入**

 so excited to have her very own library card. She picked out lots of books and brought them home to read. As she read, she noticed that some of the stories had harsh words in them. She asked her mommy why some stories were mean.

Mommy explained, "Sometimes, stories have harsh words

**真实 latent 历史（非实际推理）**

 things can catch a things to find a have to be so explore and more careful

**自主回灌**

 things can catch a big tree."

But she was never started to explore

## 样本 2

**输入**

 to help. He walked over to the flower and picked it for her. Lily was so happy and hugged her daddy tightly.


**真实 latent 历史（非实际推理）**


Lily's lesson Lily saw a beautiful bird had to the. Lily said,

**自主回灌**


Lily's lesson to that the end of the things and that she could not

## 样本 3

**输入**

 didn't like him anymore. 

But then, Cow said, "I'm sorry, Oink. I didn't mean it. I just don't like getting dirty." Oink forgave C

**真实 latent 历史（非实际推理）**

 didn't wait to playing together. From 

The moral day on, Spot

**自主回灌**

 didn't wait to tell her mom when she could go from the table. 


## 样本 4

**输入**

 face. The boy stops laughing. He starts crying.

"Ow, ow, ow!" he says. He runs to his mom.

Ben and Lily smile. They are brave. They are friends. They go back to the slide.

**真实 latent 历史（非实际推理）**



From that

**自主回灌**



From that day on, Mom and Mom. They were happy. She
