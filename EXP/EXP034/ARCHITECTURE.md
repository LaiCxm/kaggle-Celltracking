# EXP034 架构边界

- 复用 EXP033 的 24 个视频、31 个分裂 transition 和 16/20 微米候选门；
- 父节点仍由所有 Pilkwang/consensus 节点组成，不使用真实父节点提示；
- daughter 仍来自 Pilkwang 与 FOCUS3D 的联合节点池；
- 新增父节点证据：中心置信度、是否 consensus、FOCUS 实例质量、到上一帧最近节点的距离、下一帧 16 微米邻居数；
- 不使用标签选择特征、不生成 submission.csv；
- 稀疏标签下的“不发生分裂”阈值只作探索性分析。
