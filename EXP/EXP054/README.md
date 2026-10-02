# EXP054：ILP 后联合分裂选择官方 CV

EXP049 的 `cos060` 只在 ILP 已选中模型第一条 continuation 时，才允许把第二条边晋级为 division。EXP053 证明单独扩展 Top-3 和放宽三类结构门不能增加 division TP。本实验测试更直接的阻断：在 ILP 输出图之后，将 `parent -> daughter1 + daughter2` 作为一个联合候选，允许它替换父节点当前唯一的错误 continuation。这里的“联合”是确定性的后 ILP 冲突选择，不是修改 tracksdata 原生 ILP 求解器。

## 固定项

- 继承 EXP053 的确定性冻结权重、检测、普通边、ILP、DeepCenter、发散余弦和官方 patched scorer。
- 原有概率、父子距离、姐妹距离、距离不对称、下一帧发散、DeepCenter、每帧/全局上限全部不变。
- 不修改已有 fork；只允许当前出度为 0 或 1 的父节点参与联合候选。
- 目标节点发生冲突时删除竞争入边，再写入两条同源 parent→daughter 边。

## 分支

| 分支 | 候选 | 改动 |
|---|---|---|
| `off` | 无 | EXP048 关闭对照 |
| `exp049_cos060` | Top-2 | EXP049 原始后 ILP 晋级，对照 |
| `exp054_joint_top2` | Top-2 | 后 ILP 联合选择，允许替换父节点错误单边 |
| `exp054_joint_top3` | Top-2 + rank-3 | 后 ILP 联合选择并扩大延迟候选 |

本 Notebook 只运行四个完整带标签视频并调用官方 patched scorer，不生成测试集 `submission.csv`。只有联合选择增加 division TP 且普通边没有明显退化，才考虑把方案移植到测试提交。
