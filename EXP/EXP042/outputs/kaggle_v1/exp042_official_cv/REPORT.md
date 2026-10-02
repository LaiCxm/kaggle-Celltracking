# EXP042 official CV: geometry plus edge-confidence dominance rewiring

This is a four-complete-movie official patched-scorer validation; no test submission is generated.

```text
        mode  n  edge_jaccard  division_jaccard  division_tp  division_fp  division_fn  node_recall  adj_edge_jaccard  n_adj    score  rewire_candidates  rewire_accepted  rewire_rejected_divergence  rewire_rejected_advantage  rewire_rejected_edge_confidence  rewire_rejected_conflict  rewire_rejected_cap
 geom1_prob0  4        0.9159               0.2            1            0            4     0.974306          0.917994      4 0.937994                 96               18                         428                     449429                              129                         0                   78
geom1_prob05  4        0.9159               0.2            1            0            4     0.974306          0.917991      4 0.937991                 62               18                         274                     449429                              317                         0                   44
geom1_prob10  4        0.9159               0.2            1            0            4     0.974306          0.917991      4 0.937991                 57               18                         254                     449429                              342                         0                   39
        base  4        0.9159               0.2            1            0            4     0.974306          0.917988      4 0.937988                  0                0                           0                          0                                0                         0                    0
```

The three non-base modes are pre-registered geometry gates. Each accepted operation deletes exactly q->daughter2 and inserts parent->daughter2; candidate and accepted counts are in rewire_stats.csv, with every accepted operation in rewire_log.csv.
