# EXP046：受限第二出边召回

## 动机

EXP045 在 5 个真实 division 的 10 条 parent→daughter 边中发现：6 条进入候选池，4 条因原始概率低于 0.48 被排除，0 条因 source/target 预算截断。缺失概率为 `0.188、0.356、0.013、0.406`。

## 方案

保持普通候选边阈值 `0.48` 不变，仅对每个 source 的第二高 target 应用受限召回：

- 第一高边概率至少 `0.60`；
- 第二高边概率至少 `0.18`；
- parent→candidate daughter 物理距离不超过 `14µm`；
- 仍使用原有 source 最多 2 条、target 最多 1 条的度预算。

该实验不改变检测节点、不替换模型、不引入 FOCUS3D；只验证低概率第二 daughter 是否值得进入候选图。使用 4 个完整带标签视频和官方 patched scorer，属于正式口径的本地官方 CV 诊断，不提交排行榜。

## 预期判据

若 official score 和 division TP 改善且普通 edge 没有明显退化，下一轮再研究 target 冲突的局部仲裁；若 score 下降，则冻结“第二出边盲目召回”路线。
