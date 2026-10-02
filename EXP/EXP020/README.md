# EXP020：提高 FOCUS3D 轮廓证据权重

## 目的

在 EXP018/019 已证明“每视频最多 4 个冲突”只替换约 5 条边、影响过小之后，测试一个预注册的明显更激进配置。该实验仍不使用 FOCUS3D 新增节点，只提高三维轮廓证据在已有节点重接中的决定权。

## 与保守策略的唯一机制差异

| 参数 | EXP019 限量策略 | EXP020 |
|---|---:|---:|
| 每视频最多冲突 | 4 | 64 |
| 每视频最多 FOCUS3D 帧 | 4 | 16 |
| FOCUS3D 分裂分数权重 | 1.0 | 1.75 |
| 相对原连接的领先幅度 | 0.10 | 0.02 |
| 最低原始分裂轮廓分数 | 0.20 | 0.20 |

接受条件为：原始 FOCUS3D 分裂轮廓分数至少为 `0.20`，且 `1.75 × 分裂轮廓分数 > 普通延续轮廓分数 + 0.02`。

## 风险控制

- 每个视频最多检查 64 个预排序冲突，不进入 EXP019 已证伪的全帧广泛搜索。
- 每个视频最多加载 16 帧，整个 Notebook 的 FOCUS3D 全局上限为 80 帧。
- 只做一删一加换边，不新增或删除中心节点。
- 关闭验证集推理，排行榜 Notebook 只处理测试集，避免 EXP019 混合流程导致的超时。
- 不做测试集阈值扫描；全部参数在推送前锁定。

## 预期判据

运行产物 `run_stats.csv` 必须显示比 EXP019 的 5 次换边明显更多的 `focus_accepted`。最终是否有价值只以 Kaggle LB 相对干净基线 `0.946` 的变化判断。

## 执行记录

- Notebook：`CELL_infer_focus3d_aggressive.ipynb`
- SHA256：`A90217B5FB6A9B979F6D5B1FB71BB33B75742B6B74D7A60151497EB444D0F16A`
- Kaggle kernel：[`laicxm/exp020-aggressive-focus3d-evidence`](https://www.kaggle.com/code/laicxm/exp020-aggressive-focus3d-evidence)，version 1
- T4、断网、验证关闭；当前运行中，完成后从该版本提交一次排行榜评分。
