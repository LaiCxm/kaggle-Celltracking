# EXP033 架构边界

- 父体候选：事件 transition 中所有 Pilkwang-only 或 consensus 节点；
- daughter 候选：联合节点池；
- 不使用真实 parent 限定候选；
- 排序器按视频留出，训练负例按 transition 选几何 hard negative 与固定种子随机负例；
- “不分裂”由候选得分阈值表示；
- 尚未接入 continuation 图、ILP 或官方 scorer。
