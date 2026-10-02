# EXP028 候选级 OOF 排序与结构选择诊断

本实验使用 EXP027 的 union 候选池，按视频留一做 OOF；不生成 submission.csv，不运行官方 scorer。

```text
                 arm  events  rankable_events  missing_positive_events  multiple_positive_events  top1_recall  top3_recall  top5_recall  mean_reciprocal_rank  median_rank  selected_count  selected_positive_count  selected_negative_count  selection_precision  event_hit_rate
geometry_1_05_focus0      18               18                        0                         0     0.277778     0.555556     0.555556              0.416736          3.0              18                        5                       13             0.277778        0.277778
geometry_1_05_focus4      18               18                        0                         0     0.277778     0.666667     0.666667              0.444038          3.0              18                        5                       13             0.277778        0.277778
geometry_1_00_focus4      18               18                        0                         0     0.222222     0.555556     0.611111              0.386794          3.0              18                        4                       14             0.222222        0.222222
  oof_logistic_3feat      18               18                        0                         0     0.444444     0.722222     0.722222              0.575304          2.0              18                        8                       10             0.444444        0.444444
```

固定排序臂不读取标签；逻辑回归只在训练视频上读取标签，并在留出视频上预测。结构选择在每个留出视频内执行，避免跨视频节点冲突。

按视频看，逻辑回归相对几何基线新增了正确选择的主要来源是 `6bba_05db0fb1`（3 个事件中由 0 个提升到 1 个）、`6bba_87289e13`（1 个事件由 0 提升到 1 个）和 `6bba_fe670320`（2 个事件由 0 提升到 1 个）。它并非只依赖一个视频，但仍只有 12 个视频、18 个事件，不能视为已泛化到测试集。

最重要的后续验证是：用所有可用训练视频的标签重新训练固定模型，然后在完整测试电影上运行；在此之前不得把 OOF 结果直接写入生产提交。
