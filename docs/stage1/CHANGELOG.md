# 更新记录

## 2026-10-31

- 记录用户于05:03:50（UTC+11）提出的块内倒序+错开半块解码想法，完成文献核对（ZipAR、DiagD、PAR、PSF、TeCoP等），推导出丢失依赖距离≥L+s；见 [想法记录](../stage2/idea.md)。完成第1步四组pilot，并修复 `--train student` 的cuDNN eval反传问题。
- 新增 [论文计划](paper-plan.md)：冻结P蒸馏为负面基线，暂停8份/1份u对照；第1步改为同预算的同步/延迟反馈端到端对照，预先登记判定阈值，第2步设计u反推。`pair_statistics` 增加错配历史选项，pair前瞻评估新增第二块错配CE，汇总脚本增加以sync运行为基准的跨运行代价比例；pilot改为从冻结teacher.pt初始化并启动。
- 新增 [PROGRESS.md](PROGRESS.md)，核对并保存教师、六组数据/容量探索、多seed基线、一步前瞻结果及未完成对照；记录共享u的训练8份/推理1份建议与等标签数对照，明确它尚无正式效果结论。README和架构文档增加入口；本次仅更新文档，未启动训练。
- 新增根目录 [ARCHITECTURE.md](ARCHITECTURE.md) 作为当前架构约定：明确B/D/E/P、h/z/f、16词GRU与VAE的实际路径、共享同一u的配对监督、双块缓存修复、参数量与实验状态；README增加入口，区分历史四词默认路线。本次只整理文档，未启动训练或修改模型。
- 新增“两块一步前瞻”及组末真实反馈KV重放，包含串行/双块batch实现、单步分布误差、组件耗时与端到端计时。固定64词耗时下降约20%，但现有学生第二块质量未过关，不能称为保质加速；见 [实测](artifacts/pair-feedback/results.md)。
- 核对Block Transformer正式论文和官方源码，区分已经实现的embedding预测损失、主生成路径和未完整报告的预测反馈调度；见 [文献核对](block-transformer-feedback-audit.md)。
- 新增滑动窗口attention反馈学生：从当前主干状态及最近两块状态预测encoder回灌向量，支持配置历史块数。
- 直接按旧轨迹的故事ID和块号重建窗口，屏蔽左侧padding，拒绝重复或缺失的块号；推理窗口在每次新prompt时重置。
- 新增prompt历史特征补充流程，在新数据目录保存prefill过去状态，保持原teacher及h/r标签不变；新学生可以从第一次回灌起读取前序状态，旧检查点仍保留原窗口行为。
- 蒸馏训练新增 `--student-type attention --history-blocks 2`，支持检查点恢复；保留原MLP训练和检查点兼容性。
- 闭环评估记录学生结构、配置与训练步数。新增窗口隔离、恢复、padding和混合EOS检查。
- 窗口0、生成阶段窗口2、含prompt历史窗口2分别完成3000步训练及32/64-token GPU评估；历史窗口尚未解决反馈替代的质量问题，详见 [实测报告](results/feedback-attention-results.md)。

此前研究记录见 [归档更新日志](CHANGELOG.md)。
