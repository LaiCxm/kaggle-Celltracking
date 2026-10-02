# EXP055：联合 Top-2 / Top-3 测试集提交

本实验固定 EXP054 的后 ILP 联合分裂实现，在隐藏测试四视频上分别运行：

- `exp054_joint_top2`
- `exp054_joint_top3`

两条分支都会生成独立文件：

- `submission_exp054_joint_top2.csv`
- `submission_exp054_joint_top3.csv`

由于 Kaggle 一次内核评分只读取 `submission.csv`，本版本将验证集官方 CV 完全相同且
计算量更小的 `exp054_joint_top2` 复制为最终 `submission.csv`。Top-3 文件和每个分支的
运行统计、SHA256 会保留在 `/kaggle/working`，便于评分后拉回比较。

固定项：冻结权重、检测阈值、双向融合、ILP、DeepCenter、safe-division、确定性种子和
EXP054 候选生成均不变；关闭训练集验证器，不访问 GT，不扫描任何测试集参数。

Notebook：`CELL_infer_exp054_joint_top2_top3_submission.ipynb`

## version 1 修复记录

version 1 因 metadata 漏挂 `pilkwang/biohub-temporal-unet3d-seed314159-v1`，在加载独立
第二种子权重时找不到预期 SHA256，未进入测试推理，也没有生成提交文件。version 2 已补
齐 DeepCenter、第二种子和 support-pack 三个数据源，并修正末尾诊断单元中的旧实验变量。

version 2 已成功加载三套权重，但误保留了 EXP054 的 CV-only 测试入口，日志为
`Found 0 prediction graphs`。version 3 接回 EXP050 已验证的双 GPU 测试推理与合并逻辑，
再执行 Top-2/Top-3 两个后处理分支。
