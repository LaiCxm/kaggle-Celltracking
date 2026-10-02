# EXP053：Top-3 分裂结构门单变量消融

EXP053 在 EXP049 `cos060` 的确定性预测图上复用 EXP052 的候选捕获和
Top-3 延迟晋级。模型权重、检测阈值、普通 continuation、ILP、DeepCenter
门、发散方向余弦、每帧/全局上限以及官方 patched scorer 全部保持不变。

每个父节点同时保存模型排名第 2 和第 3 的备用边。Top-2 先按 EXP049
规则处理；只有 Top-2 没有晋级且该帧尚未使用晋级名额时，才检查 Top-3。
因此 Top-3 变体不会把第三边放回普通关联竞争。

## 预注册分支

| 分支 | 最小姐妹距离 | 最大距离不对称 | 最小下一帧发散增益 | 含义 |
|---|---:|---:|---:|---|
| `exp049_cos060` | 8.0 | 0.60 | 2.25 | EXP049 Top-2 对照 |
| `exp053_top3_baseline` | 8.0 | 0.60 | 2.25 | EXP052 Top-3 基线 |
| `exp053_top3_sister075` | 7.5 | 0.60 | 2.25 | 只放宽姐妹距离下限 |
| `exp053_top3_asym110` | 8.0 | 1.10 | 2.25 | 只放宽距离不对称上限 |
| `exp053_top3_diverge175` | 8.0 | 0.60 | 1.75 | 只放宽下一帧发散增益 |
| `exp053_top3_joint` | 7.5 | 1.10 | 1.75 | 三个门同时放宽，作为联合诊断 |

另外保留 `off` 分支，用于复核确定性 EXP048 基线。除表中三个参数外，
所有分支共享 `cosine_max=-0.60`、父子距离 7.0 µm、姐妹距离上限
14.0 µm、第一/第二边概率下限 0.60/0.18、每帧最多 1 个、全局最多 64 个。

## 运行和判定

Notebook 只运行四个完整带标签视频，并调用官方 patched scorer 的
`csv_to_geffs + evaluate_pairs + summarise`。它不会生成测试集
`submission.csv`，也不会在隐藏测试集上扫描参数。

重点记录每个分支的正式总分、普通边 Jaccard、division TP/FP/FN、Top-3
结构候选数和最终新增边数。只有在 division TP 增加且普通边没有明显退化时，
才考虑把对应门控组合移植到独立的测试提交实验。

Notebook：`CELL_post_ilp_division_gate_ablation_cv.ipynb`
