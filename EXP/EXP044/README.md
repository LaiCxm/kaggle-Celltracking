# EXP044：ILP 前候选边审计

## 目的

EXP041–EXP043 已证明 `safe-div` 之后的局部重连没有恢复新的真实 division。EXP044 把观测点前移到检测候选边进入图构建和 ILP 之前，逐个带标签 division 检查：

- parent 与两个 daughter 是否被检测到；
- 两条 parent→daughter 候选边是否真的进入候选池；
- 候选边在该 parent 的概率排序；
- daughter 是否同时受到其他 parent 的候选边竞争。

## 设计

- 底座：EXP010 的 frozen-weight 双种子、harmonic association、检测阈值和后处理配置。
- 唯一新增：`predict_video` 返回前保存已过候选阈值及每源/每目标预算的边池，不改变 ILP 或提交图。
- 评估：四个完整训练视频，GT 仅用于审计；节点匹配半径 7 um。
- 输出：`preilp_candidate_audit/preilp_candidate_audit.csv`、`summary.json`、`REPORT.md`。
- 本实验不把候选覆盖率当作 CV/LB，也不生成用于竞赛评分的结果。

## 判据

- 若真实两条边都在池中但排名低，下一步研究受限的 division-aware 全局选择。
- 若真实边在池中不存在，下一步回到候选生成/检测输出，不再调后置 safe-div。
