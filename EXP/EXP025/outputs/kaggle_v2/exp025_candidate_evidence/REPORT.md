# EXP025 FOCUS3D division candidate evidence calibration

Candidate-ranking diagnostic only; not official CV and not a submission.

Videos=12, events=18, rankable=14, candidates=506, candidates containing FOCUS-only nodes=250.

```text
            arm                                                                                                                                                                                                       features  candidate_roc_auc  candidate_average_precision  events  rankable_events  missing_positive_events  multiple_positive_events  top1_recall  top3_recall  top5_recall  mean_reciprocal_rank  median_rank
  distance_only                                                                                                                                    ["parent_distance_um", "daughter_distance_um", "daughter_separation_score"]           0.890044                     0.486541      14               14                        0                         0     0.428571     0.785714     0.785714              0.632426          2.0
centroid_volume                                                                                           ["parent_distance_um", "daughter_distance_um", "daughter_separation_score", "volume_balance", "volume_conservation"]           0.882884                     0.433608      14               14                        0                         0     0.500000     0.857143     0.857143              0.669728          1.5
      real_mask ["parent_distance_um", "daughter_distance_um", "daughter_separation_score", "volume_balance", "volume_conservation", "parent_union_overlap", "parent_union_parent_coverage", "parent_union_daughter_coverage"]           0.875895                     0.474551      14               14                        0                         0     0.500000     0.785714     0.857143              0.671693          1.5
  permuted_mask ["parent_distance_um", "daughter_distance_um", "daughter_separation_score", "volume_balance", "volume_conservation", "parent_union_overlap", "parent_union_parent_coverage", "parent_union_daughter_coverage"]           0.877429                     0.449691      14               14                        0                         0     0.642857     0.785714     0.857143              0.732143          1.0
```

Paired video bootstrap comparisons:

```json
[
  {
    "left": "real_mask",
    "right": "centroid_volume",
    "video_mean_top1_delta": 0.05555555555555555,
    "ci95_low": -0.16666666666666666,
    "ci95_high": 0.3333333333333333,
    "probability_gt_zero": 0.5477,
    "videos": 9,
    "iterations": 10000
  },
  {
    "left": "real_mask",
    "right": "permuted_mask",
    "video_mean_top1_delta": -0.09259259259259259,
    "ci95_low": -0.2222222222222222,
    "ci95_high": 0.0,
    "probability_gt_zero": 0.0,
    "videos": 9,
    "iterations": 10000
  }
]
```

Promotion requires real_mask to improve over centroid_volume and to lose that advantage after within-video mask-feature permutation.
