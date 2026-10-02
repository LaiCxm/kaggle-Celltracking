# EXP048：确定性官方 CV 审计

EXP046 与 EXP047 使用同一验证流程，但四个预测图的节点数、边图和 official score 存在轻微差异；差异量已经大于许多单旋钮实验的预期收益。EXP048 在关闭第二出边召回的同一基线上增加确定性运行护栏：固定 Python/NumPy/PyTorch/CUDA 随机种子、设置 `PYTHONHASHSEED` 与 `CUBLAS_WORKSPACE_CONFIG`，并关闭 cuDNN benchmark、打开 cuDNN deterministic。

本轮不改模型、不改候选阈值、不改 FOCUS3D、不提交排行榜，只验证官方 patched scorer 下的固定基线和每视频图输出是否可重复。输出中保留节点/边摘要，作为后续实验的运行指纹。
