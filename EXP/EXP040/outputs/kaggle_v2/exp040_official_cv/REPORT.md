# EXP040 official CV: compact occupied-daughter conflict rewiring

This is a four-complete-movie official patched-scorer validation; no test submission is generated.

```text
         mode  n  edge_jaccard  division_jaccard  division_tp  division_fp  division_fn  node_recall  adj_edge_jaccard  n_adj    score  rewire_candidates  rewire_accepted  rewire_rejected_divergence  rewire_rejected_conflict  rewire_rejected_cap
         base  4      0.915900          0.200000            1            0            4     0.974306          0.917988      4 0.937988                  0                0                           0                         0                    0
compact_10_14  4      0.916702          0.166667            1            1            4     0.974306          0.918818      4 0.935485              11572               24                      142871                         4                11544
compact_12_14  4      0.916702          0.166667            1            1            4     0.974306          0.918818      4 0.935485              21816               24                      290444                         4                21788
compact_14_14  4      0.916702          0.166667            1            1            4     0.974306          0.918818      4 0.935485              31090               24                      418992                         4                31062
```

The three non-base modes are pre-registered geometry gates. Each accepted operation deletes exactly q->daughter2 and inserts parent->daughter2; candidate and accepted counts are in rewire_stats.csv, with every accepted operation in rewire_log.csv.
