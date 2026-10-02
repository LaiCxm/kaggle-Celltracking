# EXP046 分析

## 官方 patched scorer 结果

EXP046 使用 4 个完整训练视频、单一 `off` 图后处理分支，官方口径结果为：

| 指标 | EXP046 | EXP019 参考 |
|---|---:|---:|
| edge Jaccard | 0.913297 | 0.915900 |
| adjusted edge Jaccard | 0.920344 | 0.917988 |
| division Jaccard | 0.200000 | 0.200000 |
| division TP/FP/FN | 1/0/4 | 1/0/4 |
| node recall | 0.978181 | 0.974306 |
| official proxy score | 0.940344 | 0.937988 |

## 结论

第二出边召回没有恢复任何新的真实 division：division TP、FP、FN 完全不变。普通 edge Jaccard 下降，但由于预测节点数量和官方调整项变化，adjusted edge Jaccard 上升，导致总分表面上提高约 `0.00236`。

因此不能把 `0.940344` 解释为候选召回规则的净收益。当前结果缺少同一运行条件下的关闭对照，下一轮先复现相同 EXP046 配置但关闭召回规则，确认分数差异是否确实由候选规则造成。

## 产物

- `EXP019_official_strategy_summary.csv`
- `EXP019_official_strategy_per_movie.csv`
- `EXP019_strategy_stats.csv`
- `focus_strategy_off_geffs/`
