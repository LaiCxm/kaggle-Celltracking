# EXP018-A：FOCUS3D 原始实例掩膜的局部分裂证据

## 最终结果（version 3）

- Kaggle submission ref：`56202599`
- Public LB：`0.946`
- 最终提交：240988 行，SHA256=`77531ca2fefc9a65a7ea5b3bc8762bb095585d4a5c3626ee90eabdd4662f498f`
- FOCUS3D：加载 15 帧，匹配 4132 个节点，检查 4716 个局部比较，接受 909 次重接。

血缘需要特别区分：Notebook 源自 EXP017，但 version 3 为控制运行时间关闭了留出集扫描，最终保留 `MOTION_RELINK_TIGHT_UM=6.0` 的 `base` 配置，没有应用 EXP017 通过扫描选择的 `tight55=5.5`。原 Notebook 将该有效基础配置标为 public 0.939。这个 0.939 不是本账号此前 ref=`56134873` 的 0.939；后者来自 EXP014，采用 EXP010 系配置、检测阈值 0.96，并关闭了 safe-div 的 DeepCenter 否决。因此两者不能构成干净对照。本次 0.946 相对原 Notebook 声明基础线是强正面信号，但在完全相同代码和运行环境下关闭 FOCUS3D 的消融完成前，不能把全部 `+0.007` 严格归因给 FOCUS3D。

另一个实现审计结论是：`max_conflicts=4` 目前只用于选择需要运行 FOCUS3D 的帧；进入这些帧后，`local_mask_rewire` 仍会扫描并修改所有满足门控的冲突，所以本次不是“全测试集只改 4 个冲突”的极保守实验。后续必须把选中的冲突标识传入重接函数，才可进行可解释的 1/2/4/8 冲突预算消融。

## EXP018-B 干净对照

对照 Notebook 为 `CELL_infer_focus_mask_control.ipynb`。它直接从取得 0.946 的 EXP018-A v3 源码生成，唯一有效配置差异为：

```python
os.environ["BIOHUB_FOCUS_MASK_ENABLE"] = "0"
```

验证流程仍关闭，运动重连严格半径仍为 6.0，数据源、冻结权重、检测、边关联、ILP 和其余后处理全部保持不变。12 个代码单元已通过编译检查，控制版代码 SHA256=`948a0f55d3bbf6d1ce90586b592a04e77007cc7cbb15213b1d2d5aabcf72071b`。

已推送 Kaggle 内核 [`laicxm/exp018-b-clean-focus3d-off-control`](https://www.kaggle.com/code/laicxm/exp018-b-clean-focus3d-off-control) version 1，使用与 EXP018-A 相同的 T4、断网和五个挂载数据源。

## 1. 实验目的

本实验以 `EXP017` 的公开 0.947 候选配置为唯一生产底座，只验证一个问题：

> 原始 FOCUS3D 的实例掩膜轮廓、体积和形状信息，能否帮助现有流程在疑似 division 的局部区域区分“真实第二个 daughter”和“错误 continuation”？

本实验不是把 FOCUS3D 当作新的主检测器，也不是把实例掩膜压缩成另一套质心检测结果。FOCUS3D 的贡献必须来自实例标签形成的三维掩膜几何；如果某个特征只依赖质心坐标，该特征不计为本实验的有效贡献。

底座固定为 `EXP017`：Pilkwang 双种子冻结权重、双向调和边融合、DeepCenter 测试时增强、`MOTION_RELINK_TIGHT_UM=5.5`，以及原有 ILP、gap、safe-division、短轨迹和线性平滑。底座任何参数不在本实验中改动。

## 2. 先前方案的审计结论

先前方案同时提出“补充 FOCUS3D 节点、重建局部边、生成 division 候选、使用掩膜特征”。这会把多个因果因素混在一起，无法判断收益来自节点召回、边重排还是轮廓证据，并且有破坏 `EXP017` 普通 continuation 的风险。

因此本实验删去以下内容：

- 不做全量 `EXP017 ∪ FOCUS3D` 节点合并；
- 不让 FOCUS3D 全局重建或批量修改普通 continuation edge；
- 不让 FOCUS3D 替换 learned edge、ILP、gap-close 或 DeepCenter；
- 不训练新的 verifier；
- 不对每一帧无条件运行 FOCUS3D 后再把所有结果写入图；
- 不使用 FOCUS3D 质心单独产生新边。

第一版只做“局部两假设仲裁”：对疑似 daughter，同时比较“保留当前 continuation”和“改接到 division parent”两个假设。这样既能处理审计中真实 daughter 已被错误 continuation 占用的问题，又把改图范围限制为单个冲突局部。

## 3. 具体流程

```text
EXP017 生成完整基础图
        ↓
只收集疑似 division 的 parent、已有 daughter、候选 daughter及其当前 parent
        ↓
对涉及的 t-1、t、t+1、必要时 t+2 帧运行原始 FOCUS3D
        ↓
保留 instance_map 的实例 ID，不只保存质心
        ↓
将 EXP017 节点与 FOCUS3D 实例一一匹配
        ↓
计算两个局部图假设的掩膜级几何证据
        ↓
仅在证据有明确优势时进行一删一加的局部边重接
        ↓
原有 DeepCenter、几何约束、容量上限和官方 scorer
```

FOCUS3D 只在局部触发，并按视频和帧缓存结果。没有疑似 division 的视频区域不运行 FOCUS3D，也不改变任何输出。

## 4. 必须真正使用的掩膜信息

当前运行时已确认稳定返回 `result["instance_map"]`。它是带实例 ID 的三维整数标签图，而不是中心热图。第一版直接从每个实例 ID 的体素集合计算以下特征：

### 4.1 运动补偿后的掩膜重叠

对 parent 掩膜使用 EXP017 已有 parent→child 的位移进行平移，再与 child 掩膜比较，计算：

- 三维交集体素数；
- 三维并集体素数；
- 掩膜 IoU；
- 交集占 parent、child 体积的比例。

不能直接比较相邻帧原坐标的重叠，因为细胞会移动；必须先用已有运动估计补偿。这个特征直接使用轮廓/体素占用，不是质心距离。

### 4.2 parent 与两个 daughter 的形状分裂证据

对于一个 `(parent, daughter1, daughter2)` 候选，计算：

- 两个 daughter 掩膜的体积；
- 两个 daughter 的掩膜是否互不重叠；
- 两个 daughter 掩膜并集与运动补偿后 parent 掩膜的软重叠；
- 两个 daughter 的体积比例是否极端不对称。

这些特征只用于判断“一个实例是否合理地分成两个实例”，不能把体积和或掩膜重叠设为硬门。比赛中的 division 步长可明显大于普通 continuation，真实 daughter 可能突然出现且离 parent 较远；因此 parent/daughter 重叠只能提供软证据，不能因为低重叠直接否决。

### 4.3 t+2 轮廓持续性

本版没有把 t+2 持续性接入生产决策。它仍是后续可注册的独立增强项，避免把跨两帧的额外假设和本次轮廓仲裁混在一起。

### 4.4 可选的实例分数

只有在当前 FOCUS3D 运行时明确返回实例级置信度时才使用。若返回对象只有 `instance_map`，则不伪造 score，也不把 `mask_thresh` 当作实例置信度。

## 5. 坐标匹配与校准

EXP017 节点和 FOCUS3D 实例必须先在同一帧做物理距离匹配。匹配半径沿用官方节点匹配半径 `7 um`，并执行一对一匹配。

第一版不对坐标做全局任意平移后再搜索最佳结果；这会把标签偏差和调参混在一起。只记录匹配误差，用于诊断。若后续证实存在稳定的系统偏移，再单独注册坐标校准实验。

这样可以避免“通过校准把错误实例强行吸到候选节点上”的隐性过拟合。

## 6. FOCUS3D 如何影响图

第一版只在一个极小的冲突局部比较两个假设：

```text
H0 continuation：Q → D，P → D1
H1 division：    删除 Q → D，保留 P → D1，并新增 P → D
```

其中 `P` 是疑似 division parent，`D1` 是它当前的 daughter，`D` 是候选第二 daughter，`Q` 是当前错误占用 `D` 的其他 parent。比较内容为：

- `Q→D` 的运动补偿掩膜相似度；
- `P→D1` 与 `P→D` 的掩膜相似度；
- `D1、D` 是否是两个独立且在 t+2 持续存在的实例；
- `P→{D1,D}` 的分裂形状证据是否明显优于 `Q→D` 的 continuation 证据。

只有 H1 相对 H0 超过预注册 margin，且 H1 通过原有 DeepCenter、几何和容量约束时，才允许在这个局部删除一条 `Q→D` 并新增一条 `P→D`。局部以外不改变任何已有普通边：

- 对已经存在的两个 daughter 候选，计算 `mask_division_score`；
- 对没有完整掩膜匹配的候选，保持 EXP017 原有决策，不因 FOCUS3D 缺失而删除；
- 不新增 FOCUS3D 节点；
- 不把 FOCUS3D 实例直接写成提交节点；
- 不允许掩膜分数单独越过 parent、sister、divergence、DeepCenter 和 frame/global cap。

最终图只能是：

```text
EXP017 原图
或
EXP017 原图 + 一次有明确掩膜优势的一删一加局部重接
```

这把潜在影响限制在单个冲突局部，避免批量破坏已验证的普通 tracking edge。

## 7. 第一版的固定门控

第一版不训练分类器，而使用可解释的固定门控：

1. parent、daughter1、daughter2 都必须能与 FOCUS3D 实例一一匹配；否则不使用掩膜加分；
2. 两个 daughter 掩膜不能明显重叠；
3. 两个 daughter 的并集必须对运动补偿后的 parent 掩膜有非零且合理的覆盖；
4. `division_mask_score` 达到固定最低分（默认 `0.20`），并且 H1 比 H0 高出固定 margin（默认 `0.10`）；
5. 掩膜证据不能绕过现有安全门；
6. 任何新增 fork 都必须通过原有官方图约束和 cap。

parent 掩膜覆盖不是非零硬门：真实 division 可能有明显位移，强制非零会重新引入已知瓶颈。覆盖率和 daughter 体积平衡只作为掩膜分数的一部分；所有阈值写入运行 manifest，不能按测试视频名称硬编码。

## 8. 评估与消融

使用完整 movie 级 OOF 和官方 patched scorer，不使用随机 edge CV。至少保存以下四个条件：

| 条件 | FOCUS3D 掩膜 | 作用 | 目的 |
|---|---|---|---|
| A | 不运行 | EXP017 原图 | 基线 |
| B | 只用 FOCUS3D 质心 | 后续对照实验 | 判断收益是否只来自另一套中心坐标 |
| C | 使用真实实例掩膜 | 当前实现 | 第一版正式实验 |
| D | 保持质心和体积不变，只在同帧实例之间随机置换轮廓/体素形状特征 | 后续对照实验 | 检查收益是否真的来自正确轮廓 |

B 是必要的质心对照，C 与 A 的差异才是提交候选。D 不能只“置换实例 ID”，因为 ID 本身没有语义；必须保持每个实例的质心和体积，只置换由体素集合计算出的形状/边界描述。只有 C 稳定优于 B，且 D 不能复制 C 的提升，才能宣称收益来自正确的轮廓掩膜，而不是 FOCUS3D 质心或额外计算本身。

每个条件必须报告：

- 官方 adjusted edge Jaccard；
- division Jaccard、TP/FP/FN；
- 完整 division triple recall；
- 普通 continuation edge 的 TP/FP/FN；
- 节点数和边数变化；
- FOCUS3D 实例匹配率、掩膜特征分布和运行时间；
- 按完整视频汇总，而不是按边随机打散。

## 9. 推广门槛

只有同时满足以下条件才允许生成测试提交：

- C 相对 A 在官方 scorer 上 division triple recall 增加；
- adjusted edge Jaccard 不下降超过预设容差；
- 普通 continuation edge 不出现系统性下降；
- C 稳定优于 B，且 D 不复制 C 的提升；
- 没有节点数量膨胀，因为第一版原则上不新增节点；
- 至少两个独立验证视频方向一致；
- 运行 manifest、掩膜缓存摘要和配置哈希完整保存。

否则只保留为诊断实验，不接入生产方案。

## 10. 再次审计：是否仍有画蛇添足

### 保留的必要部分

- 局部触发：避免把 4.5 GB FOCUS3D 模型变成全量第二检测器；
- `instance_map` 保留：确保真正使用轮廓/体素信息；
- 运动补偿后的 mask IoU：避免把细胞位移误判为掩膜不一致；
- parent→两个 daughter 的并集关系：这是区分 division 与 continuation 的核心；
- t+2 持续性：留作后续独立实验，当前版不启用；
- H0/H1 局部竞争：能够覆盖“真实 daughter 已被错误 continuation 占用”的已知瓶颈；
- 质心-only 和轮廓打乱对照：验证收益不是偷偷来自质心或候选数量。

### 明确删除的多余部分

- 第一版不新增 FOCUS3D 节点：EXP016 在一个视频的 8 个标注帧中两种方法均达到 GT 节点召回 1.0，FOCUS3D 反而多出约 9.9% 节点，暂时没有节点互补证据；
- 第一版不全局修改普通边：只允许在 H1 明显胜过 H0 的冲突局部执行一次可审计的一删一加；
- 第一版不做坐标全局校准：需要独立证据，否则会引入额外自由度；
- 第一版不训练 verifier：EXP005/006 已证明在候选召回未验证时，分类器容易过滤错或破坏普通边；
- 第一版不把面积、体积守恒设为硬规则：实例分割边界不保证物理守恒；
- 第一版不使用表面积、主轴、纹理等大批相关特征：它们会增加自由度，且暂时没有证据优于掩膜重叠、体积和持续性三类核心证据；
- 第一版不把 `mask_thresh` 当置信度：它是掩膜二值化阈值，不是可靠的实例级概率。

## 11. 风险与预期

这版实验的预期不是立即把 0.947 提升到更高分，而是回答一个可证伪问题：

> 在端点节点已经存在的真实 division 候选中，FOCUS3D 的三维实例边界是否包含 EXP017 没有利用的拓扑证据？

如果答案是否定的，应停止继续堆叠 FOCUS3D 后处理；如果答案是肯定的，第二版再单独研究“只在已确认端点缺失时增加 FOCUS3D 候选节点”，并重新做节点召回与官方图级评估。

## 12. 实现状态

已实现为 `CELL_infer_focus_mask_division.ipynb`。Notebook 内嵌 `focus_mask_arbitration.py`，运行时懒加载 `qiweiyin/focus3d-nuclei-runtime`，从原始 `instance_map` 保留完整三维体素集合，再执行局部 H0/H1 重接。FOCUS3D 不可用、匹配不足或推理失败时保留 EXP017 原图。

本地 CPU 单测覆盖：实例提取与一对一匹配、轮廓打乱后分裂分数下降、严格一删一加图重接。Notebook 已通过逐 cell Python 编译检查；本地无 CUDA，未执行 GPU 推理。已推送 Kaggle 私有内核 `laicxm/exp018-focus-mask-division` version 1，正式结果以 Kaggle 完整 movie 运行和官方 scorer 为准。

version 1 运行失败的原因是生成脚本误覆盖了 EXP017 的首个配置单元，配置漂移检查因此发现环境变量全部缺失；该问题与模型、FOCUS3D 数据集无关。已恢复原配置单元并重新推送 version 2。

version 2 完成了基础推理，但在评分阶段耗尽 12 小时。输出目录留下第一个视频 `t=0..18` 的 19 个 FOCUS3D 帧缓存，说明局部冲突扫描过宽，把普通占用 daughter 大量送入 FOCUS3D。已收紧为按几何分数排序的最多 4 个冲突、每个视频最多 2 个冲突帧，并关闭可选的 8 视频验证推理，重新推送 version 3。
