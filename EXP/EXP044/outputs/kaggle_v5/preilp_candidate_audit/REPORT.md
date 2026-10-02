# EXP044 pre-ILP candidate edge audit

This is a label-based diagnostic on complete training videos. It does not alter the graph or claim official CV/LB.
Candidate edges are the thresholded, per-source/per-target capped edges emitted immediately before graph construction and ILP.

Events audited: 5; node-match radius: 7.0 um.

| status | count |
|---|---:|
| both_edges_in_preilp_pool | 1 |
| one_edge_in_preilp_pool | 4 |

| video | event | status | daughter 1 rank/prob | daughter 2 rank/prob | incoming competitors |
|---|---:|---|---|---|---|
| 44b6_12dfb391 | 1 | one_edge_in_preilp_pool | 1/0.8025423884391785 | / | 1/1 |
| 44b6_267148e4 | 1 | one_edge_in_preilp_pool | / | 1/0.8018506765365601 | 0/1 |
| 6bba_062c8d37 | 1 | both_edges_in_preilp_pool | 1/0.9036263227462769 | 2/0.8140435218811035 | 1/1 |
| 6bba_07e24132 | 1 | one_edge_in_preilp_pool | 1/0.6342024803161621 | / | 1/1 |
| 6bba_07e24132 | 2 | one_edge_in_preilp_pool | / | 1/0.7463145852088928 | 0/1 |
