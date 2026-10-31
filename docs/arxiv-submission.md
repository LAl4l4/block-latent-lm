# arXiv v1 投稿清单

稿件位于 `paper/reverse-staggered/`。论文写作会话正在补第 6 节（块级计划负结果），并整理上传文件夹 `arxiv-upload/` 和元数据 `arxiv-metadata.md`。

## 你需要亲自做的事

1. **注册 arXiv 账号**：https://arxiv.org/user/register
   - 用 UNSW 邮箱注册。arXiv 对部分认可机构的学术邮箱会自动给 endorsement，不保证一定有，但值得一试。
   - 名字按以后想署名的写法填，后面改起来麻烦。
2. **在 Overleaf 编译**：本机没有 TeX。
   - 新建项目，选 Upload Project，上传 `arxiv-upload.zip`。
   - 编译通过后，检查 PDF 里的表格、图和参考文献。
   - 从 Overleaf 的 Submit 菜单下载 arXiv 用的源码包，里面会自动带上 `.bbl`。arXiv 不会替你跑 bibtex，所以必须带 `.bbl`。
3. **填作者信息**：在 `main.tex` 的 `\author{...}` 里填你的名字、`School of Computer Science and Engineering, UNSW Sydney` 和邮箱。
   - 如果导师愿意参与，再商量是否署名。通常你是第一作者，导师排末位。
4. **自己核对关键结论**：答辩、审稿或面试时，你需要能亲口讲清楚这些：
   - 距离下界：看 `main.tex` 第 3 节的证明，以及 `tests/test_seam_schedules.py` 里的穷举验证；
   - mask 的正确性：可见关系满足传递性，不存在信息泄漏；
   - 信息代价和排序代价的拆法：为什么 reverse16 就是"只排序"的代价。
5. **开始投稿**：https://arxiv.org/submit
   - 主分类选 `cs.CL`，交叉分类选 `cs.LG`。
   - 如果系统提示需要 endorsement，会给你一个 endorsement code，把它连同下面的邮件发给导师。
6. **许可证**：推荐 CC BY 4.0，这是常见选择，以后投会议也不冲突。
7. **AI 使用声明**：稿件里已经加了致谢说明，Claude 协助了实现、实验和写作，想法和方向来自作者。投会议时，按会议的规则再检查一遍（ACL 和 NeurIPS 都要求披露）。

## 关于 endorsement

- 在某个分类第一次投稿，需要一位在该分类近期发过若干篇论文的人背书。
- 背书只表示"这是该领域的正常学术工作"，不是审稿，不要求对方同意你的结论。
- 背书流程：
  1. 你开始投稿，系统给出 code；
  2. 你把 code 发给对方；
  3. 对方登录 arXiv，在 https://arxiv.org/auth/endorse 输入 code。
- **怎么选人：** UNSW CSE 里做 NLP、LLM 或生成模型的老师，先在 arXiv 上搜对方的名字，确认最近几年在 cs.CL 或 cs.LG 发过论文。COMP9444 的授课老师也是很自然的人选，他们认识你，方向也对口。
- 也可以请对方的博士生背书，只要他们在 cs.CL 或 cs.LG 发过足够多的论文。

## 给导师的邮件草稿（英文）

邮件写短一点，附上 PDF。正文先说明你是谁、做了什么、想请对方帮什么，最后给对方一个容易答应的选项。

> **Subject:** Request for feedback and arXiv endorsement — reverse-staggered decoding for block-parallel LMs
>
> Dear Prof. [Surname],
>
> I am [Your name], a [degree, e.g. second-year undergraduate / master's] student at UNSW CSE ([zID]). I took [COMP9444 / your course] in [term].
>
> Over the past few weeks I have been doing an independent research project on block-parallel text generation. Pipelined decoders start a block before the previous one is finished, and all existing schemes I could find (e.g. LCLM, LatLM, PSF) write each block left to right. As a result, the first token of a block cannot see the end of the previous block — the "seam" that these papers identify as the main quality problem. My idea is to generate each block in reverse (or in reversed segments) while staggering block starts. The seam token is then written last, after the previous block is complete.
>
> Results so far (small-scale, TinyStories, 10–30M parameters, 3 seeds):
> - I prove that every dropped dependency then spans at least L+s tokens (forward staggering drops distance-1 pairs).
> - In a controlled study where schedules differ only in the attention mask, dropping the same 216 dependency pairs costs 0.016 nats/token when they are distant versus 0.35 when adjacent.
> - At equal parallelism it is the best causal schedule I tested, and it removes the seam failure in samples.
> - There is a remaining ordering cost (~0.25 nats/token) that does not shrink with 3× capacity or 4× training; the draft reports this honestly, together with a negative result on block-level plans.
>
> I have a draft (attached) that I would like to post on arXiv. Since this is my first submission, arXiv requires an endorsement for cs.CL. Would you be willing to endorse it? I would also be very grateful for any feedback, and I would love to continue this work under your supervision if you have capacity (e.g. a research project or a scaled-up experiment).
>
> Disclosure: I used AI coding/writing assistants (Claude) for much of the implementation, experiments and drafting; the idea and research direction are my own, and I have checked the proofs and results myself.
>
> The code is in a private GitHub repository, and I am happy to share access.
>
> Best regards,
> [Your name]
> [zID] · [email] · [GitHub, optional]

### 发邮件的小提示

- 先只发给一位老师。等一周左右没有回复，再发给下一位，不要同时群发。
- 附件放 PDF，再附一页摘要，可以用想法文档的"当前结论"一节翻译成英文。
- 如果对方约你见面，带上三样东西：距离下界的证明（能在白板上讲清楚）、同样 216 对依赖 0.016 对 0.35 的对比、排序代价这个没解决的问题。坦诚讲局限，比夸大更能赢得信任。
- 邮件里的 AI 使用说明建议保留。老师迟早会问，主动说明更好。
