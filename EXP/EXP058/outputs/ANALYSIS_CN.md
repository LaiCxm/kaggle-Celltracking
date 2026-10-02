# EXP058 预注册分析

## 目标

在 EXP049 `cos060` 的官方四视频基准上，只将运动重连紧半径从 `6.0µm` 放宽到
`6.5µm`，检查能否恢复 Edge 或 division。EXP057 的 `5.5µm` 已明确回退，因此本实验
只测试对称方向，不扫描更多半径。

## 固定条件

- 冻结主模型、第二种子、DeepCenter 和所有离线运行时代码不变；
- 检测阈值、双向融合、ILP、gap close、safe-division、cos060 结构门不变；
- 只比较 `off` 与 `exp049_cos060` 两臂；
- 四个完整带标签视频，使用官方 patched scorer；
- 不生成测试集提交。

## 晋级条件

`exp049_cos060` 必须保持 division `TP/FP/FN=2/0/3`，且调整后 Edge Jaccard 高于
`0.9214675166`。否则回退 EXP049 的 6.0µm `cos060`。

## 推送记录

首次 slug 推送因 Kaggle 标题与 slug 清洗结果冲突而被 API 拒绝，未创建内核；随后将标题
改为包含 `V1` 后成功推送：

`laicxm/exp058-cos060-motion-relink-tight65-official-cv-v1`

当前 version 1 已进入 `RUNNING`。本文件将在官方结果拉回后补充逐视频和最终判断。

## version 1 官方结果

version 1 已通过官方 patched scorer，日志确认 `BIOHUB_MOTION_RELINK_TIGHT_UM=6.5` 且
第二种子权重 SHA256 正确。

| 分支 | Edge Jaccard | 调整后 Edge | division TP/FP/FN | 总分 |
|---|---:|---:|---:|---:|
| `off` | 0.9136780651 | 0.9205990662 | 1/0/4 | 0.9405990662 |
| `exp049_cos060` | 0.9144764289 | 0.9213982369 | 2/0/3 | 0.9613982369 |

与 6.0µm 的 EXP049 `cos060` 基准比较：

- division 保持 `2/0/3`；
- 普通 Edge Jaccard 完全相同：`0.9144764289`；
- 调整后 Edge 和总分均下降 `0.0000692797`；
- 5.5µm 的 `1/0/4` 回退被修复，但没有产生 Edge 增益。

逐视频看，6.5µm 的 `44b6_267148e4` 保持 division TP，另外三个视频的 Edge 计数与
6.0µm 基本一致；差异主要来自节点数量调整项，而不是 Edge TP/FP/FN 的改善。`cos060`
在 6.5µm 下接受 5 次结构重连，仍没有超过 6.0µm 的官方分数。

**结论：EXP058 不晋级。** 6.5µm 比 5.5µm 明显更好，说明 5.5µm 的回退确实来自过度
收紧；但相对 6.0µm 仍是中性略负，不能替代 EXP049 `cos060`。运动重连半径不再继续
扫描，保留 6.0µm `cos060` 作为四视频 division `2/0/3` 分析基线。
