# EXP041 official CV: relative edge-dominance occupied-daughter rewiring

This is a four-complete-movie official patched-scorer validation; no test submission is generated.

```text
          mode  n  edge_jaccard  division_jaccard  division_tp  division_fp  division_fn  node_recall  adj_edge_jaccard  n_adj    score  rewire_candidates  rewire_accepted  rewire_rejected_divergence  rewire_rejected_advantage  rewire_rejected_conflict  rewire_rejected_cap
dominance_1_14  4        0.9159               0.2            1            0            4     0.974306          0.917994      4 0.937994                132               19                         521                     449429                         0                  113
dominance_2_14  4        0.9159               0.2            1            0            4     0.974306          0.917991      4 0.937991                 62               18                         242                     449778                         0                   44
dominance_4_14  4        0.9159               0.2            1            0            4     0.974306          0.917991      4 0.937991                 10               10                          26                     450046                         0                    0
          base  4        0.9159               0.2            1            0            4     0.974306          0.917988      4 0.937988                  0                0                           0                          0                         0                    0
```

The three non-base modes are pre-registered geometry gates. Each accepted operation deletes exactly q->daughter2 and inserts parent->daughter2; candidate and accepted counts are in rewire_stats.csv, with every accepted operation in rewire_log.csv.
