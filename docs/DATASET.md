# DATASET — 数据说明

> 最近抓取日期:2026-08-19(见 docs/Idea_Research/weekly-2026-08-19.md)

- 训练集大小 / 测试集大小:未在公开 notebook 中硬编码(train/test 均动态枚举 `*.zarr`)。训练集 40+ 视频级(固定8胚胎用于 Lite CV 的清单见下),测试集 hidden。单个视频节点数达万级(检测上限 max_peaks=40000)。
- 数据目录布局(在 Kaggle 内核中位于 `/kaggle/input/competitions/biohub-cell-tracking-during-development/`):
  ```
  train/{dataset}.zarr/0/...      # 3D+t 图像,T×Z×Y×X,uint16
  train/{dataset}.geff           # 图标注(zarr v3 目录: nodes/ids, nodes/props/t|z|y|x/values, edges/ids)
  test/{dataset}.zarr/0/...
  sample_submission.csv
  ```
- Zarr 结构细节:
  - 数组在 `{dataset}.zarr/0/`,元数据 `0/zarr.json`(shape、data_type、chunk_shape)
  - 分块:每时间点一个 chunk,路径 `0/c/{t}/0/0/0`,blosc2(zstd)压缩;可 `blosc2` 直接解压 + frombuffer + reshape(frame_shape) 高效读取
  - 物理尺度 `SCALE = (1.625, 0.40625, 0.40625)` µm/voxel(z/y/x,z 各向异性 4 倍)
  - 根 `zarr.json` 的 `attributes.image_statistics.quantiles`(q001/q999)可用于归一化
  - `.geff` 的 `zarr.json` 中 `attributes.geff.extra.estimated_number_of_nodes` = metric 用的 T_true
- 字段说明(提交列):
  - `id`:行号;`dataset`:视频名 `{embryo_id}_{field_of_view}`
  - `row_type`:`node` / `edge`
  - node:`node_id`(数据集内从1起)、`t`/`z`/`y`/`x`(整数体素坐标,round+clip≥0)
  - edge:`source_id`、`target_id`(指向 node_id,仅允许相邻帧 t+1)
- 目标列 / 标签分布:节点时空坐标 + 有向边(允许一个 source 最多 2 个 target = 二分裂)。GT 是稀疏标注(只标了部分细胞),所以预测多出的节点不直接扣分(只通过节点数比例罚调整 jaccard)。
- 数据泄露风险点:
  - 测试集图像本身公开(可读 shape/scale/quantiles)——可用于归一化,不构成标签泄露
  - 代码中有按测试集 dataset 名硬编码后处理参数的先例(如 `6bba_05b6850b` min_track=6)= 对公开 LB 的过拟合,需谨慎
  - 曾存在 metric 漏洞被官方修复;`0.908` 被公认为修复后的 clean 基线,metric hack(负时间节点、哨兵坐标、人工 hub)已被判定无效/违规,禁止采用
- 评估 metric 的官方实现要点(公式 + 边界情况):见 OVERVIEW.md;完整实现参考
  `clean-approach-lightweight-local-cv-no-hack.ipynb` 内嵌的 `official_spec_evaluate`(7µm 匹配、division 局部分支匹配、adjusted jaccard),本地复刻时须逐字对照并写单测
- 验证策略(推荐按官方 metric 组织 CV):
  - 公开最佳做法 = **固定 8 个 train 胚胎的官方 spec Lite CV**(不是随机 K-fold):
    ```
    CV_FIXED_DATASETS = ['44b6_0113de3b','44b6_0b24845f','44b6_341df25f','44b6_e57ff5c6',
                         '6bba_05b6850b','6bba_05db0fb1','6bba_969618f6','6bba_fc83837d']
    ```
  - 参考:Notebook 124 fixed-8 Lite CV 分 ≈ 0.8792;138 fixed-4 ≈ 0.8879;对应 LB ≈ 0.908–0.915(有 gap)
  - 推理时对每个 embryo 用同一模型跑 predict + 官方 spec 打分,先写 hidden-test 提交再跑 CV(防止覆盖),CV 后再校验 SHA
