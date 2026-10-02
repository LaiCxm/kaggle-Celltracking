# EXP028：候选级 OOF 排序与结构选择诊断

EXP027 已经证明 `16/20µm` 联合候选门可以把 18 个开发 division 全部放进候选池，但几何贪心选择只选对 5/18。EXP028 固定 EXP027 的 union 候选池，不再扫描 FOCUS-only 激活代价，测试候选排序是否能改善结构选择。

比较三组固定几何代价和一组按视频留出的逻辑回归排序：

- `parent + 0.5 × daughter`；
- `parent + 0.5 × daughter + 4 × FOCUS-only 数量`；
- `parent + daughter + 4 × FOCUS-only 数量`；
- 仅使用父子距离、双子距离、FOCUS-only 数量的按视频留出逻辑回归。

评价同时报告事件级 Top-k 排名和每个留出视频内的结构选择结果。标签只用于训练视频和最终审计，不参与固定排序臂的分数计算。实验不生成提交文件，不运行官方 scorer。
