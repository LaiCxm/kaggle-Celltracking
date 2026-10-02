# EXP057 预注册分析

## 本次运行结果与配置审计（2026-09-28）

本次 Kaggle version 1 已完成官方 patched scorer，但**5.5µm 处理没有生效**，因此它不能回答预注册问题。

- Notebook 设置了 `MOTION_RELINK_TIGHT_UM=5.5`，实际推理配置读取 `BIOHUB_MOTION_RELINK_TIGHT_UM`；日志回显 `MOTION_RELINK_TIGHT_UM 6.0`，确认本次使用默认 6.0µm。
- `off`：adjusted edge Jaccard `0.9206682102`，division `1/0/4`，总分 `0.9406682102`。
- `exp049_cos060`：adjusted edge Jaccard `0.9214675166`，division `2/0/3`，总分 `0.9614675166`。
- cos060 与 EXP049/056 的逐视频节点、边和官方计数一致，属于原配置复跑；不能表述为“5.5µm 无收益”。

**处置：** EXP057 version 1 记为参数未生效的工程失败，不晋级、不生成测试提交。下一次运行前将变量修正为 `BIOHUB_MOTION_RELINK_TIGHT_UM=5.5`，并加运行前断言/配置回显校验；只有日志确认为 5.5 后，官方 CV 才能用于判断 Edge 是否提高且 division 是否保持 `2/0/3`。

## 修正版 version 2（2026-09-28）

已修正环境变量名并加入配置门禁：Notebook 现在写入并校验
`BIOHUB_MOTION_RELINK_TIGHT_UM=5.5`，若推理代码读取到其他值会在正式计算前直接失败。
修正版已推送到 Kaggle 内核
`laicxm/exp057-cos060-motion-relink-tight55-official-cv` version 2，等待官方 patched scorer 完成。
在日志确认解析值为 `5.5` 之前，不对该旋钮的 Edge 或 division 效果下结论。

## EXP055–EXP057 路线复盘（推送修正版前）

已核对最近三个实验的完整记录：

- **EXP055**：测试集上的联合 Top-2/Top-3 提交均为 `0.943`，两份提交文件逐字节相同；后 ILP 联合分裂选择没有可迁移收益，路线暂停。
- **EXP056**：Edge 保护门与 EXP049 `cos060` 的官方 CV 都为 `0.9614675166`、division `2/0/3`；保护门只改变局部图，不改变官方计数，不能称为 Edge 增益，路线暂停。
- **EXP057 v1/v2**：v1 实际仍解析为 `6.0µm`，是无效复跑；v2 已确认解析为 `5.5µm`，但在推理前因未挂载第二种子权重而失败，尚未产生评分结果。

当前路线判断：

1. 后 ILP 联合选择和额外 Edge 保护已分别完成否定/中性验证，继续堆叠相似结构门没有证据支持。
2. EXP057 仍是唯一预注册的单变量 Edge 实验，变量只应是运动重连紧半径 `6.0→5.5µm`；本次重新推送只补齐缺失的数据源，不改变代码、权重或评分口径。
3. 若修正版保持 division `2/0/3` 且调整后 Edge 高于 `0.9214675166`，才考虑保留；否则回退 EXP049 `cos060`，不再扫描相邻半径。

## 修正版 version 4 推送记录

已补挂缺失的数据集 `pilkwang/biohub-temporal-unet3d-seed314159-v1`，其余数据源、代码、
冻结权重、环境变量和官方 scorer 均未改变。Kaggle 内核
`laicxm/exp057-cos060-motion-relink-tight55-official-cv` version 4 已成功推送并进入
`RUNNING`；在拉回完整官方汇总前不对 5.5µm 的效果下结论。

## version 4 失败审计

version 4 已完成第二种子挂载、SHA256 校验和四个验证视频的双 GPU 预测；日志确认：

- `BIOHUB_MOTION_RELINK_TIGHT_UM=5.5`；
- 第二种子权重 SHA256 为预期值；
- 四个验证视频均生成原始预测图并缓存完成。

失败发生在官方评分单元加载阶段：

```text
RuntimeError: EXP019: patched official scorer dataset is not mounted
```

原因是重新推送时只补挂了第二种子，没有同时保留原 Notebook 元数据中的官方 scorer
数据源。该错误与 5.5µm 变量、模型权重和推理图无关。下一次 version 5 只补齐
`dalloliogm/biohub-official-scorer-patched`（并恢复原流程的 FOCUS3D 数据源），不改变
实验变量或评分代码。

## version 5 推送记录

已补齐五个数据源：主模型、DeepCenter、第二种子、FOCUS3D 运行时和官方 patched scorer。
Kaggle 内核 version 5 已成功推送并进入 `RUNNING`。本次仍只比较 `off` 与
`exp049_cos060`，没有生成测试集提交，也没有改变 5.5µm 单旋钮。

## version 5 官方结果

version 5 已通过官方 patched scorer，且日志确认解析值为 `5.5µm`、第二种子 SHA256
正确。结果如下：

| 分支 | Edge Jaccard | 调整后 Edge | division TP/FP/FN | 总分 |
|---|---:|---:|---:|---:|
| `off` | 0.9121199500 | 0.9190896913 | 1/0/4 | 0.9390896913 |
| `exp049_cos060` | 0.9121199500 | 0.9190896913 | 1/0/4 | 0.9390896913 |

与 EXP056/EXP049 的 6.0µm `cos060` 基准（总分 `0.9614675166`、调整后 Edge
`0.9214675166`、division `2/0/3`）相比：

- 调整后 Edge 下降 `0.0023778253`；
- division 少命中 1 个，`2/0/3 -> 1/0/4`；
- 总分下降 `0.0223778253`。

逐视频看，`44b6_267148e4` 是主要回退点：Edge `255/37/22 -> 251/44/26`，并丢失唯一额外
division TP；另外两个视频只有极小的 Edge 计数改善，无法抵消该回退。5.5µm 下
`exp049_cos060` 仍产生 16 个结构候选、接受 3 次重连，但这些局部变化没有改变官方
计数，且没有恢复 division。

**结论：EXP057 不晋级。** 5.5µm 单旋钮同时损害 Edge 和 division，不能替代 6.0µm
`cos060`；该路线停止继续扫描相邻运动半径。生产回退仍为已验证的 EXP018-A（LB
`0.946`），四视频诊断上保留 EXP049 `cos060`（official CV `0.9614675166`）作为
division `2/0/3` 的分析基线。

## 目标

在 EXP049 `cos060` 的 division `2/0/3` 基准上，只收紧运动重连半径
`MOTION_RELINK_TIGHT_UM`：`6.0 -> 5.5µm`，寻找普通 Edge 的增益。

## 固定条件

检测阈值、双向融合、ILP 权重、DeepCenter、gap2、safe-division、cos060 结构门和随机性
全部保持 EXP056/EXP049 配置；只运行 `off` 与 `exp049_cos060` 两个官方 CV 分支，不运行
隐藏测试，不扫描相邻阈值。

## 晋级条件

`exp049_cos060` 必须保持 division `TP/FP/FN=2/0/3`，且调整后 Edge Jaccard 高于
`0.9214675166`。否则回退 EXP049 cos060，不把局部候选数或 proxy 指标当作收益。
