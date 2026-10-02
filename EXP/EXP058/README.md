# EXP058：cos060 + motion relink tight 6.5 um

EXP057 将运动重连紧半径从 6.0 收紧到 5.5 微米后，官方 patched scorer 显示 Edge 和
division 同时下降。本实验只把该单旋钮改为 6.5 微米，其他模型、双种子融合、ILP、
DeepCenter、safe-division、cos060 结构门和评分流程全部锁定。

比较分支：`off` 与 `exp049_cos060`。

晋级条件：`exp049_cos060` 保持 division `2/0/3`，且调整后 Edge Jaccard 高于
EXP049 的 `0.9214675166`。否则回退 EXP049 `cos060`，不继续扫描运动半径。
