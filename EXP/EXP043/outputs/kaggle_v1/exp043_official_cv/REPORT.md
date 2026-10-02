# EXP043 official CV: strict learned-edge provenance gating

This is a four-complete-movie official patched-scorer validation; no test submission is generated.

```text
         mode  n  edge_jaccard  division_jaccard  division_tp  division_fp  division_fn  node_recall  adj_edge_jaccard  n_adj    score  rewire_candidates  rewire_accepted  rewire_rejected_divergence  rewire_rejected_advantage  rewire_rejected_edge_confidence  rewire_rejected_missing_edge_confidence  rewire_rejected_conflict  rewire_rejected_cap
strict_prob05  4        0.9159               0.2            1            0            4     0.974306          0.917994      4 0.937994                 55               18                         268                     449429                              277                                       53                         0                   37
strict_prob10  4        0.9159               0.2            1            0            4     0.974306          0.917994      4 0.937994                 50               18                         248                     449429                              302                                       53                         0                   32
strict_prob20  4        0.9159               0.2            1            0            4     0.974306          0.917994      4 0.937994                 43               18                         211                     449429                              346                                       53                         0                   25
         base  4        0.9159               0.2            1            0            4     0.974306          0.917988      4 0.937988                  0                0                           0                          0                                0                                        0                         0                    0
```

The three non-base modes are pre-registered geometry gates. Each accepted operation deletes exactly q->daughter2 and inserts parent->daughter2; candidate and accepted counts are in rewire_stats.csv, with every accepted operation in rewire_log.csv.
