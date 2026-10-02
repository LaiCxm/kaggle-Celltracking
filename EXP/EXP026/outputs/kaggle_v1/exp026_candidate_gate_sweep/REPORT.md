# EXP026 division candidate geometry gate sweep

Candidate-recall diagnostic only; not official CV and not a submission.

Events=18, all endpoints matched=18, positive triple present at max gate=18.

## Diagonal gate sweep

```text
 parent_radius_um  daughter_radius_um  recalled_events  total_events  candidate_recall  candidate_count  focus_only_candidate_count  mean_candidates_per_event  median_candidates_per_event  p95_candidates_per_event  max_candidates_per_event  candidate_multiplier_vs_14
             14.0                14.0               14            18          0.777778              506                         250                  28.111111                         25.0                     60.30                        62                    1.000000
             16.0                16.0               17            18          0.944444             1171                         538                  65.055556                         60.0                    136.15                       137                    2.314229
             18.0                18.0               17            18          0.944444             2214                        1045                 123.000000                        118.0                    262.60                       300                    4.375494
             20.0                20.0               18            18          1.000000             3884                        1882                 215.777778                        212.0                    467.40                       481                    7.675889
             24.0                24.0               18            18          1.000000            10937                        5219                 607.611111                        500.0                   1344.05                      1401                   21.614625
             28.0                28.0               18            18          1.000000            24694                       12270                1371.888889                       1258.0                   2899.00                      2967                   48.802372
             32.0                32.0               18            18          1.000000            48002                       24174                2666.777778                       2444.0                   5613.25                      5915                   94.865613
```

## Recall-budget frontier

```text
 parent_radius_um  daughter_radius_um  recalled_events  total_events  candidate_recall  candidate_count  focus_only_candidate_count  mean_candidates_per_event  median_candidates_per_event  p95_candidates_per_event  max_candidates_per_event  candidate_multiplier_vs_14
             14.0                14.0               14            18          0.777778              506                         250                  28.111111                         25.0                     60.30                        62                    1.000000
             14.0                16.0               16            18          0.888889              650                         323                  36.111111                         34.0                     73.40                        87                    1.284585
             16.0                16.0               17            18          0.944444             1171                         538                  65.055556                         60.0                    136.15                       137                    2.314229
             16.0                20.0               18            18          1.000000             1708                         776                  94.888889                         82.5                    197.80                       208                    3.375494
```

## Pre-registered selected full-recall gate

```json
{
  "parent_radius_um": 16.0,
  "daughter_radius_um": 20.0,
  "recalled_events": 18,
  "total_events": 18,
  "candidate_recall": 1.0,
  "candidate_count": 1708,
  "focus_only_candidate_count": 776,
  "mean_candidates_per_event": 94.88888888888889,
  "median_candidates_per_event": 82.5,
  "p95_candidates_per_event": 197.79999999999998,
  "max_candidates_per_event": 208,
  "candidate_multiplier_vs_14": 3.375494071146245
}
```

The selected gate is a development input for the next global solver, not a validated CV optimum.
