# EXP034 parent evidence division discovery

Diagnostic only; no submission and no official CV.

Videos=24; transitions=31; truth events=31; candidates=686116.

```text
                          arm  events  rankable_events  top1_count  top1_recall_all  top1_recall_rankable  top3_count  top3_recall_all  top3_recall_rankable  top5_count  top5_recall_all  top5_recall_rankable  top10_count  top10_recall_all  top10_recall_rankable  top20_count  top20_recall_all  top20_recall_rankable  top50_count  top50_recall_all  top50_recall_rankable  top100_count  top100_recall_all  top100_recall_rankable  mean_reciprocal_rank_all  median_rank_rankable
                     geometry      31               31           0         0.000000              0.000000           0         0.000000              0.000000           0         0.000000              0.000000            1          0.032258               0.032258            2          0.064516               0.064516            7          0.225806               0.225806             8           0.258065                0.258065                  0.015023                 317.0
                logistic_base      31               31           2         0.064516              0.064516           3         0.096774              0.096774           3         0.096774              0.096774            4          0.129032               0.129032            6          0.193548               0.193548            9          0.290323               0.290323            10           0.322581                0.322581                  0.095186                 440.0
     logistic_parent_evidence      31               31           2         0.064516              0.064516           2         0.064516              0.064516           3         0.096774              0.096774            5          0.161290               0.161290            6          0.193548               0.193548            9          0.290323               0.290323            10           0.322581                0.322581                  0.086240                 415.0
logistic_parent_mask_geometry      31               31           3         0.096774              0.096774           3         0.096774              0.096774           3         0.096774              0.096774            6          0.193548               0.193548            7          0.225806               0.225806            8          0.258065               0.258065            11           0.354839                0.354839                  0.115996                 368.0
```

The correct parent is not supplied to candidate generation. Parent evidence is computed from observations in adjacent frames and the current frame. All Pilkwang/consensus parents in each selected transition compete. A score threshold represents the no-division option. Threshold scans are exploratory OOF diagnostics and are not deployment thresholds.
