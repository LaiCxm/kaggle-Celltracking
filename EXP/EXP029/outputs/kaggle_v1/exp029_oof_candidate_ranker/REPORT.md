# EXP029 wider-video OOF division candidate ranker

Diagnostic only; no submission and no official CV.

Videos=24; events=31; union candidates=1967.

```text
                 arm  events  rankable_events  missing_positive_events  multiple_positive_events  top1_recall  top3_recall  top5_recall  mean_reciprocal_rank  median_rank  selected_count  selected_positive_count  selected_negative_count  selection_precision  event_hit_rate
geometry_1_05_focus0      31               31                        0                         0     0.419355     0.612903     0.709677              0.535870          3.0              31                       13                       18             0.419355        0.419355
geometry_1_05_focus4      31               31                        0                         0     0.419355     0.645161     0.741935              0.550097          3.0              31                       13                       18             0.419355        0.419355
geometry_1_00_focus4      31               31                        0                         0     0.354839     0.548387     0.645161              0.485997          3.0              31                       11                       20             0.354839        0.354839
  oof_logistic_3feat      31               31                        0                         0     0.451613     0.709677     0.838710              0.606457          2.0              31                       14                       17             0.451613        0.451613
```

The learned arm is trained only within leave-one-video-out folds. Pilkwang and FOCUS3D weights remain frozen.
