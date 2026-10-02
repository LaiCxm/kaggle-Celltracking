# EXP019：FOCUS3D 掩膜利用方式比较

## 目的

EXP019 的本来目的，是在同一批完整验证视频、同一批冻结模型预测和同一官方修订评分器下，回答一个机制问题：FOCUS3D 三维实例掩膜应该以哪种方式参与已有轨迹图，才能改善边关联或 division，而不破坏原有普通轨迹。第一轮只比较机制，不同时扫描距离、分数和优势阈值。

EXP019 是验证实验，不是排行榜提交实验，也不是为了直接测量完整测试集 LB。

## 实验单位与固定条件

- 实验单位：完整视频，不把视频内的边当成独立样本。
- 验证集：按两个胚胎前缀各选 2 个完整训练视频，共 4 个视频；优先包含真实分裂的视频。
- 固定模型：EXP017 的 Pilkwang 双种子冻结检测与关联权重。
- 固定后处理：EXP017 的 base 配置，运动重连严格半径 6.0 微米。
- 正式响应：官方 patched scorer 的 pooled score、adjusted edge Jaccard、division Jaccard 及 TP/FP/FN。
- 阻断：所有策略共享同一批原始预测图，并保留逐视频结果，避免将视频内大量边误当成独立重复。

## 第一轮策略

| 策略 | 对现有图的作用 |
|---|---|
| `off` | 完全不使用 FOCUS3D，内部基线 |
| `broad_rescue` | 复刻 EXP018-A：先选择少量帧，但在已加载帧内广泛搜索并重接 |
| `selected_rescue` | 修复 EXP018 的预算漏洞，只允许预先排序的最多 4 个冲突进入重接 |
| `veto_only` | 不抢占新的子细胞，只删除掩膜形状分数和体积平衡同时强烈反对的 safe-div 新增边 |
| `hybrid_selected` | 严格限量抢救后，再对原 safe-div 新增边做保守否决 |

`selected_rescue` 每个视频最多选择 4 个冲突、最多加载 4 帧。`veto_only` 只允许删除带 `safe_division=1` 标记的新增边；缺失掩膜时保持原图，不做否决。

## 评分纪律

Notebook 不使用旧的四视频手写代理分数进行排序，而是直接加载 `dalloliogm/biohub-official-scorer-patched`：

1. 每个策略生成四视频 CSV；
2. 使用官方 `csv_to_geffs.py` 转回 GEFF；
3. 使用官方 `evaluate.py/evaluate_pairs` 逐完整视频评分；
4. 使用官方 `summarise` 按样本大小加权 adjusted edge Jaccard，并 pooled division TP/FP/FN；
5. 显式检查加载的是 2026-07-17 后的 patched division metric。

产物：

- `EXP019_official_strategy_summary.csv`
- `EXP019_official_strategy_per_movie.csv`
- `EXP019_strategy_stats.csv`
- `focus_strategy_<mode>.csv`

v1 Notebook 是本实验的正式验证 Notebook，不产生竞赛提交文件是正确设计。Kaggle 提交页的 `submission.csv` 要求只适用于排行榜提交，不应反过来改变 EXP019 的验证 Notebook。

## 本地验证

- FOCUS3D 基础掩膜与重接单测：3 项；
- EXP019 策略单测：3 项；
- 官方 patched scorer 来源与聚合公式单测：2 项；
- 合计 8/8 通过；
- Notebook 12/12 代码单元通过 Python 编译。

## EXP019 正式结果

官方 patched scorer 的四视频 pooled 结果：`off`、`selected_rescue`、`veto_only`、`hybrid_selected` 均为 `0.937988`（adjusted edge Jaccard `0.917988`，division Jaccard `0.2`，division TP/FP/FN=`1/0/4`）；`broad_rescue` 降至 `0.911881`，并产生 7 个 division FP。严格限量的 `selected_rescue` 虽接受 5 次局部重接，但官方汇总分数与关闭 FOCUS3D 的基线持平。因此 EXP019 的结论是：在当前接入方式和阈值下，FOCUS3D 没有带来可测量净收益；广泛重接会破坏原有图。

Notebook SHA256：`3f125a9d13501c272085bd90d5f3b647bb280789b6ea4b177d831cd482de1786`。

Kaggle 内核：[`laicxm/exp019-focus3d-strategy-official-compare`](https://www.kaggle.com/code/laicxm/exp019-focus3d-strategy-official-compare)，version 1（诊断比较，不提交）。

修正版验证内核：[`laicxm/exp019-focus3d-official-cv`](https://www.kaggle.com/code/laicxm/exp019-focus3d-official-cv)，version 2。该版本跳过测试集推理，只运行一次四视频验证预测和五种官方评分比较；比较结束后不再重新处理测试集，也不生成排行榜提交文件。version 1 仅完成了首次推送，version 2 才包含“去掉测试集推理”的超时修复。

Version 2 已运行完成并拉回 `outputs/kaggle_cv_v2/`。三份核心 CSV 与原始验证运行逐字节一致，正式结论得到复现：`off`、限量重接、仅否决、混合策略均为 `0.937988`；广泛重接为 `0.911881`。详细分析见 `outputs/kaggle_cv_v2/REPORT.md`。

## 误提交工程记录

后续曾错误地把验证 Notebook 改造成提交版 `CELL_focus3d_strategy_official_compare_submit.ipynb`，同时运行验证比较和测试集推理，导致提交 ref=`56227265` 超时。该提交不是 EXP019 的评分结果，也不改变上述官方验证结论。EXP019 不应再使用这个混合 Notebook 做排行榜提交。
