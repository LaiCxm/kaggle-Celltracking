# EXP044 运行说明

## V2 失败原因

V2 没有进入推理或候选审计。Kaggle 在运行时补丁阶段报错：

```text
RuntimeError: Coordinate-manifest patch expected one pre-return block, found 0
```

原因是 EXP010 的物化支持仓库已经包含坐标清单钩子，EXP044 又对同一钩子执行了一次严格的文本替换。候选边审计逻辑尚未执行，因此 V2 没有任何候选覆盖结论，也不能视为实验阴性结果。

V3 已将该补丁改为幂等逻辑：钩子已存在时直接复用；只有代码漂移到既非原始块、也非已插入钩子时才失败。候选边清单仍在 `predict_video` 返回前保存，不改变图构建、ILP 或输出结果。

## V3 失败原因

V3 在第一个代码单元失败：复制的 EXP010 公开方案展示单元尝试读取 `our_probes.csv`、`provenance.csv`、`single_knob_steps.csv`，但本次内核没有挂载可选的 `biohub-knob-provenance` 数据集。该单元与候选审计无关，因此没有开始推理。

V4 已移除这个可选展示单元，只打印跳过提示；模型、数据集、候选边保存位置和审计逻辑均未改变。

## V4 失败原因

V4 已成功跳过公开溯源展示并完成依赖安装，但在 Cell 8 的运行时补丁阶段失败。当前支持包中的脚本没有 EXP010 Notebook 所假定的旧坐标清单文本块，因此即使该坐标清单对 EXP044 并无必要，严格计数仍报 `found 0`。没有开始预测或候选审计。

V5 删除该无关补丁，仅用空白兼容的正则表达式定位 `return coords, all_edges`，在该返回点保存候选边池；其余推理和审计逻辑不变。

本实验已推送到 Kaggle，等待用户运行完成后拉回结果。

它不是提交实验，也不计算官方 CV/LB。Notebook 只记录 EXP010 冻结推理链在建图/ILP 前生成的候选边池，并用四个完整训练视频的稀疏 GT 做逐事件审计。正式分析将根据 `preilp_candidate_audit/REPORT.md` 判断瓶颈属于：

1. 节点未检出；
2. parent→daughter 候选边未进入池；
3. 候选边已进入但在 parent 排名或 daughter 竞争中落后；
4. 候选边存在但被 ILP/后续全局约束舍弃。

在结果拉回前，不把候选层召回率写成 CV，也不改变 `EXP018-A` 的生产回退。
