# EXP057：cos060 + motion relink tight 5.5 um

EXP056 证明 Edge 保护门可以保持 EXP049 cos060 的 division 2/0/3，但没有提高
官方 Edge 分数。本实验回到原始 cos060，只改变一个已有公开方案使用过的运动重连参数：

- MOTION_RELINK_TIGHT_UM: 6.0 -> 5.5
- 其他检测、融合、ILP、DeepCenter、safe-division 和 cos060 结构门全部锁定
- 官方 patched scorer，比较 off 与 exp049_cos060
- 不运行隐藏测试，不扫描相邻阈值

晋级条件：exp049_cos060 的 division 仍为 2/0/3，且调整后 Edge Jaccard 高于
0.9214675166。否则回退到 EXP049 cos060。

