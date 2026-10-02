# EXP022：分裂结构理想上限诊断

目标：在 EXP019 的四视频官方 CV 基础图上，用真实标签做验证集专用的理想结构修复，测量“节点已经存在，只是分裂边选错”时的最高潜在收益。

处理组：

1. `baseline`：EXP019 的 FOCUS3D 关闭组，不修改。
2. `occupied_only`：只尝试修复 EXP010 阶段审计中被普通延续边占用子节点的事件。
3. `divergence_only`：只尝试修复被后续发散距离门拒绝的事件。
4. `all_repairable`：修复最终节点集中能被官方 7 微米规则映射的全部事件。

理想修复只允许：

- 删除两个真实子节点上的冲突错误入边；
- 删除真实父节点指向其他节点的出边；
- 添加父节点到两个子节点的正确边；
- 不新增、删除或移动任何预测节点；
- 不修改事件邻域以外的边。

运行：

```powershell
python tools/test_division_oracle.py
python tools/run_exp022_division_oracle.py
```

正式结论只认 `EXP/EXP022/outputs/local_official_cv/` 中 patched official scorer 的结果。本实验读取真实标签，只能用于验证诊断，不生成 `submission.csv`，不得部署到测试集。
