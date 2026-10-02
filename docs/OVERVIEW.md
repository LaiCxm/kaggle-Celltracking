# OVERVIEW — 比赛概览

> 最近抓取日期:2026-08-19(见 docs/Idea_Research/weekly-2026-08-19.md)

- 比赛名称:Biohub - Cell Tracking During Development
- 比赛 slug(数据/提交用):`biohub-cell-tracking-during-development`
- 比赛链接:https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- 任务类型:细胞追踪(检测 + 关联)。输入 3D+t 荧光显微镜视频,输出每个细胞(node)的时空位置 + 相邻帧关联(edge),允许二分裂(division)。
- 评估指标(官方 metrics.md,royerlab/kaggle-cell-tracking-competition,pinned commit `075fc5f5a52d11077f9dc2b074644618f26939e2`):
  ```
  1. 节点匹配:同帧最优二部指派(Hungarian),物理距离 ≤ 7 µm
     (物理尺度 SCALE = (1.625, 0.40625, 0.40625) µm/voxel,对应 z/y/x)
  2. Edge Jaccard = TP / (TP + FP + FN)
     - edge TP:预测边两端节点都匹配到 GT,且 GT 对应两点间存在 GT 边
     - edge FN:GT 边未被任何预测边命中
     - edge FP:预测边命中"target 匹配到 GT 且该 GT 有其他源"或
               "source 匹配到 GT 且该 GT 有其他 target" 的情形;
               其余未被覆盖的预测边被 metric 忽略
  3. Adjusted Edge Jaccard = max(0, jaccard * (1 - 0.1*(T_pred - T_true)/T_true))
     - T_pred 预测节点总数,T_true 来自 .geff 元数据 estimated_number_of_nodes(粗估计,含未标注节点)
  4. Division Jaccard = TP/(TP+FP+FN):围绕 GT 分裂的时间窗(±1 帧),
     局部分支拓扑匹配 + Kuhn 最大基数二部匹配;FP 判定有保守补丁规则
  5. Final score = adjusted_edge_jaccard + 0.1 * division_jaccard
     - 按样本宏观加权:adj_edge_jaccard 按 w_i = TP_i+FP_i+FN_i 加权平均,
       division_jaccard 用全部视频求和后算 Jaccard
  ```
- 提交格式(sample_submission.csv):
  ```
  id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
  node 行:row_type=node,node_id 从 1 开始,t/z/y/x 为四舍五入整数(≥0),source_id=target_id=-1
  edge 行:row_type=edge,node_id=t=z=y=x=-1,source_id/target_id 引用 node_id
  ```
- 时间线:截止 2026-09-29(约 40 天剩余,以 2026-08-19 计)
- 已知公开强方案要点(2026-08-19 抓取):
  - LB 榜首约 0.952;公开 clean 基线链 0.908 → 0.915
  - **直接复现首选**:Pilkwang Kim 的 support-pack 推理链
    `biohub-tracking-support-pack-50ep-v1`(公开,9324 次下载)+
    clean-approach notebook(no-hack,固定8胚胎官方 spec Lite CV)
  - 核心架构:3D 时间窗 UNet(TemporalUNet3D)检测 + edge transformer 打分 + ILP(SCIP)全局图优化 + 大量确定性后处理(gap 修补/分裂恢复/轨迹平滑)
  - 双种子 logit blend、8 视角 D4 TTA 等进阶手段详见周报
