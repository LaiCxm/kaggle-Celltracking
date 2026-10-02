# EXP035 hierarchical parent gate

Diagnostic only; no submission and no official CV.

Candidates=686116; truth events=31; transitions=31.

```text
                    arm  k  events  parent_recall  candidate_count  rankable_events  top1_count  top3_count  top5_count  top10_count  top1_recall_all  top3_recall_all  top5_recall_all  median_rank_rankable
         candidate_base  1      31       0.000000              309                0           0           0           0            0         0.000000         0.000000         0.000000                   NaN
         candidate_base  2      31       0.000000              680                0           0           0           0            0         0.000000         0.000000         0.000000                   NaN
         candidate_base  4      31       0.000000             1461                0           0           0           0            0         0.000000         0.000000         0.000000                   NaN
         candidate_base  8      31       0.032258             3598                1           1           1           1            1         0.032258         0.032258         0.032258                   1.0
         candidate_base 16      31       0.129032             8487                4           2           2           2            3         0.064516         0.064516         0.064516                   3.5
         candidate_base 32      31       0.322581            21483               10           2           3           5            6         0.064516         0.096774         0.161290                   5.5
         candidate_base 64      31       0.612903            55104               19           2           3           4            5         0.064516         0.096774         0.129032                  83.0
candidate_mask_geometry  1      31       0.000000              309                0           0           0           0            0         0.000000         0.000000         0.000000                   NaN
candidate_mask_geometry  2      31       0.000000              680                0           0           0           0            0         0.000000         0.000000         0.000000                   NaN
candidate_mask_geometry  4      31       0.000000             1461                0           0           0           0            0         0.000000         0.000000         0.000000                   NaN
candidate_mask_geometry  8      31       0.032258             3598                1           1           1           1            1         0.032258         0.032258         0.032258                   1.0
candidate_mask_geometry 16      31       0.129032             8487                4           2           2           2            2         0.064516         0.064516         0.064516                  11.0
candidate_mask_geometry 32      31       0.322581            21483               10           3           4           4            5         0.096774         0.129032         0.129032                  15.5
candidate_mask_geometry 64      31       0.612903            55104               19           3           3           5            6         0.096774         0.096774         0.161290                  79.0
```

Parent Top-K is selected from observation-only evidence. The positive parent is not supplied during candidate generation.
