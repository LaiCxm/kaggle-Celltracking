# EXP050：固定 EXP049 cos060 的测试集提交

EXP049 在四个完整带标签视频的官方 patched scorer 上选出 `cos060`：官方 CV 为 `0.9614675166`，相对确定性关闭基线 `0.9406682102` 提升 `0.0207993064`。

本实验只做一次测试集运行：

- 关闭训练集 validator；
- 只运行 `exp049_cos060`；
- 保持 EXP049 的冻结权重、检测阈值、ILP、后处理和确定性种子不变；
- 由既有 `write_test_submission()` 生成唯一的 `submission.csv`；
- 不在测试集上扫描余弦阈值，也不生成多个候选提交。

目的：检验四视频官方 CV 的提升能否迁移到公开排行榜。结果应与 EXP049 的 `cos060` 代码血缘严格一致；如果提交失败，失败只能归因于提交工程，不改变 EXP049 的官方 CV 结论。

## 运行修复记录

version 1 误保留了 EXP019 official-CV Notebook 的 CV-only 逻辑，日志显示 `skipped test-set inference`，因此 `write_test_submission()` 找不到四个测试视频的预测图并报错 `Expected 4 graphs, found 0`。这不是模型或提交文件格式错误。

version 2 已恢复 EXP018 中已验证的测试集 GPU 分片推理与 `.geff` 合并流程，但拼接时误留下一个字符，形成 `pstart_time`，末尾读取 `start_time` 时触发 `NameError`。

version 3 已将计时变量修复为 `start_time`，并增加单元测试验证变量定义顺序、四视频合并、固定 `exp049_cos060` 调用和 `submission.csv` 存在性。该版本已推送运行。
