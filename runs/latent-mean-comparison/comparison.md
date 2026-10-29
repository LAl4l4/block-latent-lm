# 原始 latent 与均值 latent 的短文本生成对比

检查点：step 4112；6 个 prompt × 2 个随机种子；最多 64 token。

均值定义：对每个 prompt 的完整生成 latent 序列取时间均值，再重复到全部生成位置。保留原 prompt latent。两边采样设置和随机种子完全相同；不强制最低生成长度。

这项离线诊断比较逐步变化的 latent 与其均值；均值仍包含该 prompt 的信息，不能等同于没有 latent，也不能据此单独判断主干的质量收益。

完全相同的输出：0 / 12。

第一个 token 即不同：11 / 12；平均续写长度：原始 49.25 token，均值 30.75 token。

## Prompt 1 / seed 42

**输入**

 so excited to have her very own library card. She picked out lots of books and brought them home to read. As she read, she noticed that some of the stories had harsh words in them. She asked her mommy why some stories were mean.

Mommy explained, "Sometimes, stories have harsh words

**原始 latent**

. You should value them."

Lily thanked her mommy and went home. They had a great time together and were always curious. 

The moral of the story is that it will be good to keep others in life. Even when you are sad, it is always a good idea to wear and be kind to

**均值 latent**

 home with her mommy and daddy, and her family all about the bright, bright, new and shiny things. Finally, when it was finished, the little girl knew it was perfect for her family. 

The girl and her mommy were so happy to see the little girl smile and she was no longer bored. She

相同前缀：0 token；输出长度：64 / 64 token。

## Prompt 1 / seed 43

**输入**

 so excited to have her very own library card. She picked out lots of books and brought them home to read. As she read, she noticed that some of the stories had harsh words in them. She asked her mommy why some stories were mean.

Mommy explained, "Sometimes, stories have harsh words

**原始 latent**

 to learn."

The boy was so surprised. He realized that he was sorry, he had not been a good idea. He decided to read a book too, so she took the book away.

Mommy explained that she wanted to read books and read with her family and the words of words of words.

**均值 latent**

 to learn more stories about the world around them.

The moral of the story is that kindness can make a lot of fun and kindness to the world around us.

相同前缀：2 token；输出长度：64 / 34 token。

## Prompt 2 / seed 42

**输入**

 to help. He walked over to the flower and picked it for her. Lily was so happy and hugged her daddy tightly.


**原始 latent**


When it was time to go home, Lily waved goodbye to the flower and took it out. She was so happy and grateful to be back to her family for helping her.

**均值 latent**

 mommy and daddy helped her find the flower. They all loved it and gave it more flowers to help her. Finally, they found the flower in the garden and Lily felt happy again. She hugged her daddy and they became best friends.

相同前缀：0 token；输出长度：36 / 47 token。

## Prompt 2 / seed 43

**输入**

 to help. He walked over to the flower and picked it for her. Lily was so happy and hugged her daddy tightly.


**原始 latent**


From that day on, Lily and her mommy always asked each other for help and they talked about it. Lily's daddy said that it was the best thing to do. They both looked at it and said that it was important to help others and make them happy. Lily learned that being generous and doing good things and

**均值 latent**

 to come back to the library to see the flowers.

相同前缀：0 token；输出长度：64 / 11 token。

## Prompt 3 / seed 42

**输入**

 didn't like him anymore. 

But then, Cow said, "I'm sorry, Oink. I didn't mean it. I just don't like getting dirty." Oink forgave C

**原始 latent**

 and with some value of being kind. From then on, he made sure to be more careful and not throw things in the mud again.

**均值 latent**

 not clean up again and they became friends. The end.

相同前缀：0 token；输出长度：28 / 12 token。

## Prompt 3 / seed 43

**输入**

 didn't like him anymore. 

But then, Cow said, "I'm sorry, Oink. I didn't mean it. I just don't like getting dirty." Oink forgave C

**原始 latent**

, the two friends decided to play with the new peanut. They were happy and played together all day long. The end.

**均值 latent**

 to learn how to jump and dance.

相同前缀：0 token；输出长度：25 / 8 token。

## Prompt 4 / seed 42

**输入**

 face. The boy stops laughing. He starts crying.

"Ow, ow, ow!" he says. He runs to his mom.

Ben and Lily smile. They are brave. They are friends. They go back to the slide.

**原始 latent**



They start to play with the swings. They play with their other toys. They have fun.

But then it is time to rest. Mommy comes home and sees the mess. She is not happy. She is not happy.

"Mom, mom, we are sorry. We broke the slide

**均值 latent**

 to clean up. The boy is happy too. He likes to play with the kids. He gives them more cookies and milk. They try to make them feel better.

They play with their ball and laugh. They have fun.

They are happy. They have each other. They share their toys.

相同前缀：0 token；输出长度：64 / 64 token。

## Prompt 4 / seed 43

**输入**

 face. The boy stops laughing. He starts crying.

"Ow, ow, ow!" he says. He runs to his mom.

Ben and Lily smile. They are brave. They are friends. They go back to the slide.

**原始 latent**

 They say goodbye to the bird.

They go back to the park. They go back to the park. They have fun.

They go back to their toys. They put their bags. They go home. They say goodbye.

They say goodbye to their mom. They go home. They go

**均值 latent**

 to play with their other kids.

Ben and Lily do not want to go on the slide again. They want to stop playing. They do not want to see her friends on the slide. They go tired.

They sit on the bench. They sit down. They sit on the couch. They

相同前缀：0 token；输出长度：64 / 64 token。

## Prompt 5 / seed 42

**输入**

 big.

But then the comet started to shrink. Little by little, it became smaller and smaller until it was gone.

Jacky and Ella looked around, but the comet was nowhere to be found.

"Where did it go?" asked Jacky.


**原始 latent**


"I don't know, I thought it would be so bad," said the comet.

So the two of them went back to their house and never found each one again.

**均值 latent**

 home with little star and they became the best of friends.

相同前缀：0 token；输出长度：38 / 12 token。

## Prompt 5 / seed 43

**输入**

 big.

But then the comet started to shrink. Little by little, it became smaller and smaller until it was gone.

Jacky and Ella looked around, but the comet was nowhere to be found.

"Where did it go?" asked Jacky.


**原始 latent**


The comet replied, "Don't worry, the comet's just a little one. We can stay here and keep it safe."

So, they went inside to show the comet the next week.

The end.

**均值 latent**

 to be careful and enjoy the comet more and more.

相同前缀：0 token；输出长度：47 / 11 token。

## Prompt 6 / seed 42

**输入**

, it's going to be very exciting!"

So, off they went to the wedding of the two birds. The trees were all decorated with colourful streamers and the two birds sang love songs together. Both the bear and the fox were very happy for the two birds and wished them all the best. The bear and the fox clapped when the two birds officially said "I

**原始 latent**

 love that little children and they all come back soon!"

The bear and the bear danced and clapped their hands in joy. They knew this was the best thing they needed. They will never forget each other's wonderful ideas.

**均值 latent**

 the whole little fox and his friends come back soon.

The moral of the story is that even more humble creatures can make friends.

相同前缀：0 token；输出长度：46 / 28 token。

## Prompt 6 / seed 43

**输入**

, it's going to be very exciting!"

So, off they went to the wedding of the two birds. The trees were all decorated with colourful streamers and the two birds sang love songs together. Both the bear and the fox were very happy for the two birds and wished them all the best. The bear and the fox clapped when the two birds officially said "I

**原始 latent**

 love the sun and how the trees make birds singing and singing together!" 

And from that day forward the bear and the birds became good friends. They would go for a walk together and the bear would always remember the day they all enjoyed the sunshine.

**均值 latent**

 to learn their new language, and they all lived happily ever after.

相同前缀：0 token；输出长度：51 / 14 token。
