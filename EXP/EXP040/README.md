# EXP040：被占用 daughter 的局部冲突重连

## 目的

验证 EXP010/EXP022 暴露的具体失败模式：真实分裂的第二个 daughter 已被错误 continuation 占用，导致后续 safe-div 无法建立正确 fork。

本实验不继续扩大候选池，也不训练新的分类器，而是在完整预测图上测试一个严格受限的原子修复：

```text
删除：错误父节点 q -> daughter2
新增：真实候选父节点 parent -> daughter2
```

这里的“真实候选父节点”仅表示算法提出的候选，不使用 GT 标签选边。

## 固定约束

- parent 已有一条 continuation 到 daughter1；
- daughter2 当前恰好有一条来自其他父节点的入边；
- parent、daughter1、daughter2 满足三维物理距离门限；
- 两条 daughter 轨迹在下一帧继续发散；
- 每个 parent、daughter2、被替换父节点最多参与一次修改；
- 每帧最多修改 1 次，每个视频最多修改 6 次；
- 每次严格一删一加，不新增或删除检测中心。

## 对照分支

| 分支 | parent-daughter 门限 | daughter-daughter 门限 |
| --- | ---: | ---: |
| `base` | 不重连 | 不重连 |
| `compact_10_14` | 10 um | 14 um |
| `compact_12_14` | 12 um | 14 um |
| `compact_14_14` | 14 um | 14 um |

所有重连分支的下一帧发散增量门限固定为 2.25 um。

## 预注册判据

主要判据是 4 个完整验证视频上的官方 patched scorer 总分相对 `base` 的变化。只有官方总分提高，且没有以明显破坏普通 continuation 边为代价，才认为该机制值得继续。

同时报告：

- 每视频官方分数；
- edge 与 division 的 TP、FP、FN；
- 候选数、接受重连数、拒绝原因；
- 每次一删一加的完整审计日志。

候选召回、候选数量和局部排序指标不能替代官方评分。本 Notebook 仅做 CV，不运行测试集推理，也不生成 `submission.csv`。

## 运行

Notebook：`CELL_official_cv_local_division_conflict.ipynb`

Kaggle 数据源：

- `pilkwang/biohub-tracking-support-pack-50ep-v1`
- `pilkwang/biohub-temporal-unet3d-seed314159-v1`
- `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`
- `qiweiyin/focus3d-nuclei-runtime`
- `dalloliogm/biohub-official-scorer-patched`

## 运行记录

- version 1：工程失败。推送命令漏挂第二个随机种子的冻结权重数据集，Notebook 的 SHA256 完整性检查在推理前主动终止；未产生实验结果。
- version 2：补挂 `pilkwang/biohub-temporal-unet3d-seed314159-v1` 后重新推送，实验逻辑与预注册参数不变。
