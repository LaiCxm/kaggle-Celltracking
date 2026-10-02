# EXP027 V1 运行分析

## 运行结论

V1 不是超时，也不是模型推理失败。Kaggle 内核已经完成：

- 12 个视频；
- 18 个真实 division 事件；
- 36 个标注帧；
- Pilkwang 冻结模型推理；
- FOCUS3D 推理。

随后在结构选择结果汇总阶段失败，因此没有生成 `selection_summary.csv`、`selected_candidates.csv`、`REPORT.md` 或 `summary.json`。本次运行不能提供候选选择精度，也不能作为正式 CV。

## 失败原因

原代码使用：

```python
event_lookup = event_df[event_df.mode == mode]
```

`mode` 是 Pandas `DataFrame` 的方法名。属性访问得到的是 `event_df.mode()` 方法，而不是名为 `mode` 的数据列；比较后形成错误的布尔索引，最终触发：

```text
KeyError: False
```

已改为：

```python
event_lookup = event_df[event_df["mode"] == mode]
```

并加入生成器回归测试，确认不会再次使用方法属性访问。

## V2 状态

修正后的 Notebook 已重新生成并推送为 Kaggle version 2：

`laicxm/exp027-focus3d-structural-selection-diagnostic`

V2 仍是事件级结构诊断，不生成 `submission.csv`，不运行官方 scorer，也不改变生产回退方案。

## 证据

完整 Kaggle 日志：

`exp027-focus3d-structural-selection-diagnostic.log`

