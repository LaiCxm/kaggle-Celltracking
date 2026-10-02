# EXP023：Pilkwang 与 FOCUS3D 已标注实例命中审计

本实验只比较细胞实例检测，不比较连接边和分裂后处理。评估范围是 `EXP021` 已经共同处理的 4 个带标签视频、30 个事件窗口帧。比赛 GT 是稀疏标注，并不包含画面中的所有真实细胞，因此本实验只能可靠测量已标注节点召回和定位误差，不能得到完整实例检测混淆矩阵。

定义：在同一帧内使用物理尺度 `(1.625, 0.40625, 0.40625)` 微米/体素和 7 微米匹配半径做一对一最小距离匹配。

- `TP`：匹配到唯一已标注 GT 细胞实例的预测中心；
- `unmatched_predictions`：没有匹配稀疏 GT 的预测中心，其中包含大量真实但未标注的细胞，不能称作 FP；
- `FN`：没有被预测中心匹配到的已标注 GT 细胞实例；
- `TN`：背景不是有限的实例集合，不定义也不计入。

Notebook 原始产物保留逐视频、逐帧和全局统计。由于 GT 稀疏，其中 `FP` 实际应读作 `unmatched_predictions`，`precision` 和 `F1` 不可用于评价模型；`TP`、`FN`、已标注节点召回率和匹配距离有效。

运行：

```powershell
python tools/push_custom_kernel.py --exp EXP023 `
  --kernel CELL_compare_instance_detection_confusion.ipynb `
  --competition biohub-cell-tracking-during-development `
  --data-source qiweiyin/focus3d-nuclei-runtime `
  --data-source pilkwang/biohub-tracking-support-pack-50ep-v1 `
  --no-internet
```

该实验不生成 `submission.csv`，结果只用于验证实例检测质量。
