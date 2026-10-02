# EXP027 FOCUS3D structural selection diagnostic

Diagnostic only; not official CV and not a submission.

Events=18; candidate gate=16.0/20.0 um.

```text
    mode  focus_activation_cost  events  candidate_count  positive_candidate_count  selected_count  selected_positive_count  selected_negative_count  missed_events  selection_precision  event_hit_rate
pil_only                    0.0      18              983                        15              15                        3                       12              3             0.200000        0.166667
   union                    0.0      18             1708                        18              18                        5                       13              0             0.277778        0.277778
   union                    2.0      18             1708                        18              18                        5                       13              0             0.277778        0.277778
   union                    4.0      18             1708                        18              18                        5                       13              0             0.277778        0.277778
   union                    8.0      18             1708                        18              18                        4                       14              0             0.222222        0.222222
   union                   12.0      18             1708                        18              18                        3                       15              0             0.166667        0.166667
```

Selection uses geometry plus FOCUS-only activation cost and never reads labels. This greedy selector is a structural diagnostic, not the production ILP.
