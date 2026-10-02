# EXP045：阈值前候选边诊断

## 目的

EXP044 证明四个完整训练视频中，5 个真实 division 的 10 条 parent→daughter 边只有 6 条进入 ILP 前候选池，但还不能区分两种原因：

1. 原始边概率本身低于候选阈值；
2. 原始边分数足够，但在 source/target 节点预算中被截断。

EXP045 在阈值和节点预算之前保存每个 source、每个 target 的原始边概率 top-16 并集，随后用训练标签匹配真实 division 边，输出原始概率、source/target 排名、阈值通过情况、预算拒绝情况和最终候选状态。

## 判据

- `final_candidate`：已进入生产候选池；
- `source_budget_rejected`：原始概率超过 0.48，但 source 排名超过每个 source 的 2 条边预算；
- `target_budget_rejected`：原始概率超过 0.48，但 target 排名超过每个 target 的 1 条入边预算；
- `prethreshold_topk_only`：被 top-16 保存，但原始概率没有超过 0.48；
- `not_in_saved_topk`：不在保存的 top-16 并集中，不能仅凭本实验判断其原始分数。

## 口径

这是标签诊断实验，不修改生产图、不生成正式 CV/LB，也不提交竞赛分数。Notebook 使用 EXP010 的冻结双种子推理链和相同的四个完整带标签视频；官方 patched scorer 不适用于本实验的诊断目标。

## 验证

- `tools/validate_exp045_patch.py`：动态补丁执行及运行脚本编译检查；
- `tools/test_audit_prethreshold_candidates.py`：NPZ 读取和状态分类测试（需要 pytest 环境）；
- 运行环境未安装 pytest 时，使用等价直接断言验证核心函数。
