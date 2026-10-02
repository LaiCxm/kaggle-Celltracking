# EXP021 V1 架构冻结记录

日期：2026-09-16

这份记录冻结第一版 FOCUS3D-first 重构的研究问题和边界。它不是排行榜方案，也不宣称已经优于 EXP018 的 0.946。

## 研究假设

FOCUS3D 的潜在增益来自三维实例轮廓、实例质量和同帧实例关系，而不是另一套可直接替换的中心点检测器。要验证这个假设，FOCUS 必须在节点候选和事件候选生成阶段参与，而不是只在旧图末端换一条边。

## 三层系统

### 1. 统一观测层

每一帧保留两类原始观察：

- Pilkwang：中心坐标、检测置信度、双种子/增强一致性和学习边证据；
- FOCUS3D：实例编号、质心、裁剪掩膜、体积、包围盒、逐体素置信度均值和低分位数、边界截断比例。

FOCUS 实例编号只在单帧内有效，跨帧引用必须使用 `(t, label)`。中心和实例采用物理距离门控的一对一匹配，匹配歧义不能静默丢弃。

节点提案分为：

- `consensus`：中心与实例匹配；
- `pilkwang_only`：只有中心观察；
- `focus_only`：只有实例观察，第一版标记为 `provisional`，不直接写生产图。

### 2. 统一事件候选层

候选生成不读取旧 ILP 的已选边，因此不会因为 daughter 已被错误 continuation 占用而消失。

- 普通延续：枚举 `t -> t+1` 的节点对，在物理距离范围内保留候选；若两端有 FOCUS 实例，计算裁剪掩膜运动补偿后的重合和体积比；
- 分裂事件：枚举 `parent(t), daughter1(t+1), daughter2(t+1)`，两个 daughter 同时进入一个显式三元组。只使用候选距离预筛，不要求 daughter 当前没有入边；
- 缺帧事件和全局选择器尚未在 V1 实现，不能由 V1 输出推断正式分数。

所有 FOCUS 几何量是软特征，不是已校准概率，也不允许直接乘任意权重后与 learned edge probability 比较。

### 3. 全局求解层（下一阶段）

后续求解器一次性选择节点、普通延续和分裂三元组，并施加：

- child 入度不超过 1；
- 普通 parent 至多一个 child；
- division parent 的两个 daughter 必须成组选择；
- 时间单调、无环；
- 节点激活代价和 appearance/disappearance 代价。

在这一层完成前，不恢复旧的 `motion_relink` 整图覆盖、`safe-div` 末端补边或 FOCUS 局部换边。

## V1 已实现范围

- FOCUS runtime 一帧只推理一次；同时保留 `instance_map`、`confidence_map` 和 `log_info`；
- 实例摘要使用裁剪布尔掩膜，而不是长期保存 Python 体素集合；
- 统一节点提案和显式 continuation/division 候选；
- 跨帧编排器保证候选来自完整节点池，而不是旧图已选边；
- CPU 合成自检。

## V1 明确不做

- 不替换 EXP018/0.946 生产流程；
- 不运行 ILP、运动重连、gap/gap2、safe-div、DeepCenter veto 或坐标平滑；
- 不把 FOCUS-only 节点直接写入提交；
- 不在四个已反复分析的视频上选择阈值；
- 不把候选数量或 proxy 分数写成正式 CV。

## 证据和消融顺序

1. 固定完整视频和官方 patched scorer，复现 FOCUS off 基线；
2. 比较 Pilkwang、FOCUS、union 的节点端点召回和完整 division triple 召回；
3. 固定 Pilkwang 节点和候选集，只加入 FOCUS 普通延续特征；
4. 固定节点，使用显式 division 三元组与无 FOCUS/有 FOCUS 对照；
5. 最后才测试激活高质量 FOCUS-only 节点。

必须保留的反事实对照：FOCUS 质心-only、真实掩膜、保持质心和体积但随机置换掩膜、置信度置乱。只有真实掩膜优于质心且置乱后优势消失，才能归因于轮廓。

## 晋级和停止

- union 没有新增标注端点或完整 division triple：停止 FOCUS-only 节点支路；
- 开发面板官方 pooled score 至少 `+0.002`，两个胚胎前缀均不下降；
- division 阶段必须真实增加 TP，不能只增加 FP；
- 普通边调整后分数下降超过 `0.001`：停止该分支；
- 任何提交前确认实验都必须使用预先锁定的完整视频面板和视频级 bootstrap。

生产回退始终为 EXP018 的 `0.946`，不因 V1 候选数量增加而改变。
