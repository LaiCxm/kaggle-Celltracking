# EXP035 架构边界

- 复用 EXP034 的候选构造和 24 个视频、31 个真实 transition；
- 不使用真实父体提示；
- 第一阶段只使用无标签父体证据，对每个 transition 的父节点排序；
- 第二阶段使用候选级三特征与 FOCUS3D 掩膜几何排序 daughter 三元组；
- K 是诊断变量，不从验证标签中选择部署值；
- 不生成 `submission.csv`，不计算官方 CV。
