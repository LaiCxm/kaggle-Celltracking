# EXP010 4-video division stage audit

审计日期：2026-09-09  
Kaggle kernel：`laicxm/exp010-division-stage-audit-v2`（`COMPLETE`）

## 审计范围

本报告只分析 4 个带标签验证视频中的 5 个 GT division，不包含 Kaggle test submission 中的 94 个 fork。

这里的 TP/FN 用于追踪 GT division 在各图阶段是否形成完整 parent-to-two-daughters fork。Notebook 内置的 4-video proxy 不是官方 patched scorer，尤其会忽略无法映射到 GT parent 的预测 fork，因此本报告不使用其 `fp_official` 字段判断正式 FP 或正式 CV。

## 逐阶段结果

| 图阶段 | 预测 fork | TP | FN | 相对上一阶段的结论 |
| --- | ---: | ---: | ---: | --- |
| raw ILP | 0 | 0 | 5 | ILP 没有产生 fork |
| edge filter | 0 | 0 | 5 | 无变化 |
| motion relink | 0 | 0 | 5 | 没有产生 fork；新增一条错误 continuation，阻止了一个潜在 safe-div TP |
| single-parent / single-child | 0 | 0 | 5 | 无变化 |
| gap close | 0 | 0 | 5 | 无变化 |
| gap2 | 0 | 0 | 5 | 无变化 |
| safe-div selected | 51 | 1 | 4 | 新增全部 51 个 fork，并新增唯一的 TP |
| division geometry filter | 51 | 1 | 4 | 未删除 TP，也未删除 fork |
| prune isolated | 51 | 1 | 4 | 未删除 TP，也未删除 fork |
| short-track | 51 | 1 | 4 | 未删除 TP，也未删除 fork |
| final | 51 | 1 | 4 | 最终保持不变 |

safe-div 内部数量链为：

```text
446  通过早期几何、互近邻和 t+2 divergence 检查后记录的 geometric candidates
 59  通过 DeepCenter/symmetry veto 后的 proposals
 51  最终写入图的 fork
  1  命中完整 GT division
  4  GT division 仍为 FN
```

日志的 `deepcenter_rejected` 字段实际混合了 DeepCenter 与 symmetry 拒绝，不能把 `446 -> 59` 精确拆成两个模块各自的拒绝数。`59 -> 51` 是最终冲突选择/写图阶段造成的差额，不是 cap 拒绝；日志中 `cap_skipped=0`。

## 5 个 GT division 的去向

| 视频 / 事件 | 最终结果 | 首个关键阻断点 |
| --- | --- | --- |
| `44b6_12dfb391` / event 1 | FN | raw ILP 已用错误 parent `32172` 占用 daughter `32646`；正确 parent 应为 `32130` |
| `44b6_267148e4` / event 1 | FN | raw ILP 时 daughter `839` 尚无入边，但 motion relink 新增错误边 `636 -> 839`；正确 parent 应为 `649` |
| `6bba_062c8d37` / event 1 | TP | 通过所有 safe-div gate 并被选中，是唯一 TP |
| `6bba_07e24132` / event 1 | FN | raw ILP 已用错误 parent `14120` 占用 daughter `14487`；正确 parent 应为 `14149` |
| `6bba_07e24132` / event 2 | FN | safe-div 的 `t+2 divergence_distance_gate` 拒绝 |

## 结论

没有 TP 被 safe-div 之后的清理模块删掉。唯一 TP 是 safe-div 新增的。

另外 4 个 GT division 中，2 个被 raw ILP 的错误 continuation 提前占用真实 daughter，1 个被 motion relink 新增的错误 continuation 占用，1 个被 safe-div 的 t+2 divergence 距离门拒绝。当前首要问题不是后置 filter 过严，而是 safe-div 的候选枚举依赖“daughter 尚无入边”，导致上游 continuation 错误会直接使 GT division 无法进入候选。

最终 51 个 fork 中只有 1 个与完整 GT division 重合。不能将 proxy CSV 中的 `fp_official=0` 解读为其余 50 个不是 FP；正式 division precision、FP 和 CV 必须另用官方 patched scorer 计算。

## 产物

- `division_stage_summary.csv`：每视频、每阶段的节点、边、fork、proxy TP/FN
- `division_stage_audit.csv`：5 个 GT division 在每个阶段的结构状态
- `division_candidate_audit.csv`：5 个 GT division 在 safe-div 候选门中的去向
- `exp010-division-stage-audit-v2.log`：Kaggle 完整执行日志
