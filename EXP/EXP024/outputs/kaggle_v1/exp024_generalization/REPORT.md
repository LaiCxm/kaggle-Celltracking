# EXP024 cross-video sparse-GT instance-recall audit

This is a labeled-node recall audit, not a full confusion matrix or official CV.

Videos: 12; selected frames: 76; labeled GT nodes: 501.

| source | hits | misses | micro recall | mean match distance (um) |
|---|---:|---:|---:|---:|
| Pilkwang | 501 | 0 | 1.000000 | 1.6835 |
| FOCUS3D | 485 | 16 | 0.968064 | 1.7573 |

Paired GT outcomes: `{"both_hit": 485, "both_miss": 0, "focus3d_only_hit": 0, "pilkwang_only_hit": 16}`.

Primary video-level recall difference (Pilkwang - FOCUS3D): `{"bootstrap_probability_gt_zero": 0.9982, "ci95_high": 0.0509046002196687, "ci95_low": 0.004133597883597878, "iterations": 10000, "mean": 0.022349932538288698, "seed": 20260917, "units": 12}`.

Unmatched predictions cannot be interpreted as false positives because GT is sparse.
