# EXP017 - 公开 0.947 候选复现

## 来源

- Kaggle Notebook: `sjlee101/biohub-lf-dctta`
- 原始版本 ID: `348619543`
- 本地入口: `CELL_infer_public_0947_dctta.ipynb`

## 当前结论

这是当前公开 Notebook 排序中高于已明确标注 `0.946` 方案的最高可信候选，且原作者运行产物完整。社区传播成绩约为 `0.947`，但 Kaggle CLI 元数据不提供该 Notebook 对应提交的排行榜分数，因此在本项目自行提交前，不能把 `0.947` 记作已复现 LB。

已排除的更高标题方案：

- 两个 `0.948 reproduction` Notebook 的源码与复跑记录不一致，其中修改后的配置曾得到 `0.935`，不能视为可靠的 `0.948` 方案。
- `the-0-950-cluster-what-happens-after-the-rescore` 是旧 division metric 漏洞分析，不是可执行的 `0.950` 推理方案。

## 主要变化

- 保留 Pilkwang 冻结双种子检测和关联权重。
- 使用双向调和边融合。
- 对 DeepCenter 启用测试时增强：`BIOHUB_DEEPCENTER_TTA=1`。
- 在留出视频上扫描少量后处理参数，最终选择 `tight55`，即运动重连严格半径从 `6.0` 收紧到 `5.5` 微米。
- 内置代理分数从 `0.949038` 提升到 `0.951095`；该代理不是官方 patched scorer 的正式 CV。

## 保存的参考产物

`outputs/public_sjlee_dctta/` 保存原作者运行产生的参数扫描结果、运行统计、完整性报告和参考提交文件。它们只用于复现核对，不代表本项目已获得相同排行榜成绩。

本地文件哈希：Notebook SHA256=`95F08BB82388E9206F45C86DAF926BB3D7A7FFF9889C89B5FA9596F8E13C3BD4`；参考 submission SHA256=`A69C78229C6556D06D5FE9FF050074254B4A326E83249EFBF578B843AEC7A848`。
