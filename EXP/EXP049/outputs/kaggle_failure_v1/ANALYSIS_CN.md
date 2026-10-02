# EXP049 version 1 失败分析

## 失败阶段

四个验证视频的冻结模型推理和 Top-2 候选导出均已完成，`exp049_candidates/*.npz` 共四份。失败发生在官方评分前的第一个 `off` 后处理分支；`focus_strategy_off.csv` 只有表头，因此没有任何 official CV 结果。

## 根因

EXP049 嵌入的辅助函数命名为 `_position_um(node, scale)`，覆盖了旧管线已有的 `_position_um(node)`。旧后处理随后按原单参数接口调用该名称，产生调用签名错误。

## 修复

- 辅助函数重命名为 `_exp049_position_um`；
- 生成 Notebook 后自动提取辅助函数名，与 EXP048 原后处理的全部函数名做集合冲突检查；
- 保留动态推理脚本补丁验证、Notebook 全单元编译和选择器单元测试。

该失败没有进入任一实验分支评分，不构成算法结果。
