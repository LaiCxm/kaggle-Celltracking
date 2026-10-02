# EXP033：不知道真实父体时的 division 发现

EXP030–032 都在已知真实 division parent 的局部候选中选择 daughter，证明的是局部配对能力，不是完整的分裂发现能力。

EXP033 去掉这个提示：在每个事件窗口的全部 Pilkwang/consensus 父节点上枚举三元组，daughter 仍来自 Pilkwang 与 FOCUS3D 联合节点池。比较距离基线、三特征逻辑回归和加入五个 FOCUS3D 掩膜几何量的逻辑回归。

每个父体只保留最高分候选，并通过分数阈值允许“不发生 division”。阈值扫描只用于诊断，不作为部署阈值；本轮不生成提交文件，不计算官方 CV。
