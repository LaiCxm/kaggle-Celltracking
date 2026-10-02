# EXP036 hierarchical parent gate

Diagnostic only; no submission and no official CV.

Candidates=686116; truth events=31; transitions=31.

```text
                    arm   k  events  parent_recall  candidate_count  rankable_events  top1_count  top3_count  top5_count  top10_count  top1_recall_all  top3_recall_all  top5_recall_all  median_rank_rankable
         candidate_base   8      31       0.096774             7620                3           2           2           3            3         0.064516         0.064516         0.096774                   1.0
         candidate_base  16      31       0.258065            16331                8           3           5           6            6         0.096774         0.161290         0.193548                   2.0
         candidate_base  32      31       0.290323            36076                9           3           4           6            6         0.096774         0.129032         0.193548                   4.0
         candidate_base  64      31       0.516129            77347               16           2           3           4            6         0.064516         0.096774         0.129032                  60.0
         candidate_base 128      31       0.709677           162031               22           2           3           3            4         0.064516         0.096774         0.096774                  97.5
candidate_mask_geometry   8      31       0.096774             7620                3           2           2           3            3         0.064516         0.064516         0.096774                   1.0
candidate_mask_geometry  16      31       0.258065            16331                8           4           4           5            5         0.129032         0.129032         0.161290                   3.0
candidate_mask_geometry  32      31       0.290323            36076                9           4           4           4            5         0.129032         0.129032         0.129032                   6.0
candidate_mask_geometry  64      31       0.516129            77347               16           3           3           4            6         0.096774         0.096774         0.129032                  55.0
candidate_mask_geometry 128      31       0.709677           162031               22           3           3           4            6         0.096774         0.096774         0.129032                  89.0
```

Parent Top-K is selected from observation-only evidence. The positive parent is not supplied during candidate generation.
