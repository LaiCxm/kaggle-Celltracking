# EXP010 节点与 division 候选审计

审计对象：EXP010 验证集中的 4 个带标签视频。GT 通过官方比赛 GEFF 下载；预测使用 EXP010 `unet_transformer_val` GEFF（safe-div 重放前）。
节点匹配使用逐帧一对一最小距离匹配，物理尺度为 `(1.625, 0.40625, 0.40625)` 微米/体素，半径为 7 微米。

## 重要边界

预测 GEFF 只保存该文件写入时已保留的边，不保存模型内部 ILP 之前的完整候选边池，也不是 safe-div 重放后的最终图。因此本报告的 9.9 微米、Top-10 候选图是独立重构，用于回答候选召回是否可能成立；它不是内部私有候选池的逐边复刻。

## 汇总

- GT division：5；parent 检出：5；daughter1 检出：5；daughter2 检出：5。
- 独立 Top-10 候选图同时覆盖两个 daughter：4 / 5。
- 预测 GEFF 中保存的边形成完整 fork：0 / 5。这不是 safe-div 之后的最终图。

按事件分类：
- 节点未检出：0；
- 节点都检出但独立候选 Top-10 未覆盖：1；
- 候选覆盖但最终边未形成 fork：4；
- 预测 GEFF 中已经保存为 fork：0。

## 逐事件

| 视频 | 事件 | 预测 parent | 两个 daughter 是否检出 | Top-10 候选覆盖 | GEFF 中两条边 | safe-div 重放 | 结论 |
|---|---:|---:|---|---|---|---|---|
| 44b6_12dfb391 | 1 | 32130 | 1/1 | 1/1 | 1/0 | gt_daughter_has_incoming_edge / selected=0 | candidate_generated_but_not_final |
| 44b6_267148e4 | 1 | 649 | 1/1 | 1/1 | 0/1 | gt_daughter_has_incoming_edge / selected=0 | candidate_generated_but_not_final |
| 6bba_062c8d37 | 1 | 5997 | 1/1 | 1/1 | 1/0 | candidate_survived_gates / selected=1 | candidate_generated_but_not_final |
| 6bba_07e24132 | 1 | 14149 | 1/1 | 1/0 | 1/0 | gt_daughter_has_incoming_edge / selected=0 | candidate_not_generated |
| 6bba_07e24132 | 2 | 24663 | 1/1 | 1/1 | 0/1 | divergence_distance_gate / selected=0 | candidate_generated_but_not_final |

详细节点对应关系见 `node_audit.csv`；逐 division 候选排名、GEFF 中保存的边、daughter 入边和 safe-div 重放状态见 `candidate_audit.csv`；机器可读汇总见 `summary.json`。
