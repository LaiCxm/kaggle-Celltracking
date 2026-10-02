# EXP037 soft parent evidence

Diagnostic only; no submission and no official CV.

Candidates=686116; truth events=31; transitions=31.

```text
                      arm  events  top1_count  top3_count  top5_count  top10_count  top1_recall  top3_recall  top5_recall  median_rank  mean_reciprocal_rank
           candidate_base      31           2           3           3            4     0.064516     0.096774     0.096774        427.0              0.094988
              soft_parent      31           2           3           3            3     0.064516     0.096774     0.096774        531.0              0.090386
soft_parent_mask_geometry      31           2           3           4            5     0.064516     0.096774     0.129032        222.0              0.098372
```

Parent aggregate evidence is used as a soft candidate feature. No parent Top-K gate is applied.
