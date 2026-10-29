# 自主 latent 回灌检查

检查点 step 101；codec 在本批真实目标上的还原准确率 99.33%。

真实历史分支读取正确的前序 target latent，仅用于诊断；自主分支只读取 prompt 与自身预测的 latent。差异不等于生成质量下降，需人工检查下列短续写。

全部自主 latent 为有限值：True。

## 样本 1

**输入**

 so excited to have her very own library card. She picked out lots of books and brought them home to read. As she read, she noticed that some of the stories had harsh words in them. She asked her mommy why some stories were mean.

Mommy explained, "Sometimes, stories have harsh words

**真实 latent 历史（非实际推理）**

 because, b could two its bad he

**自主回灌**

 because, b couldf

## 样本 2

**输入**

 to help. He walked over to the flower and picked it for her. Lily was so happy and hugged her daddy tightly.


**真实 latent 历史（非实际推理）**


 man?" J before is he C is. "u Kl? other. He menu mom permit him inviteog it took with it.nantie. windows me dragged made made " C him's. M.

**自主回灌**


 man?" J before is he C is. "u Kl. otherE very could t Maybe very inviteog it took with it. soantie swam windows me dragged made help " C him's. M.

## 样本 3

**输入**

 didn't like him anymore. 

But then, Cow said, "I'm sorry, Oink. I didn't mean it. I just don't like getting dirty." Oink forgave C

**真实 latent 历史（非实际推理）**

. M. in it bardsy. decidedy see M I other a all said If said found so that buss thatic some kept away always. many watched until other. mom always

**自主回灌**

. M. in it, It could. decidedy see N I other a shine said. thoughty found so that Bs!ic some. again always. ok " going said mill for always M.. laughed watched N M like always gentlyog and car? like is!". " g at.".

## 样本 4

**输入**

 face. The boy stops laughing. He starts crying.

"Ow, ow, ow!" he says. He runs to his mom.

Ben and Lily smile. They are brave. They are friends. They go back to the slide.

**真实 latent 历史（非实际推理）**

 He'slly

**自主回灌**

 He'slly momo teaching
