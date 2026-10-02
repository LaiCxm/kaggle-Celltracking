# EXP027 架构边界

EXP027 位于候选层之后、生产全局 ILP 之前：

1. 统一节点层保留 Pilkwang、consensus、FOCUS-only；
2. 分裂候选使用 EXP026 选出的 `16/20µm` 几何门；
3. 候选代价只包含几何距离和来源激活代价；
4. 选择时施加父唯一、daughter 入度唯一和事件唯一约束；
5. 真实三维轮廓只保留在上游审计中，不给予额外权重；
6. FOCUS-only 仍是临时节点，不直接写入生产图。

由于本实验只围绕有标签分裂事件建立候选集，它回答的是“来源激活代价和基本结构约束是否能选对候选”，不能代替完整电影上的 continuation、appearance、disappearance 和官方 scorer。
