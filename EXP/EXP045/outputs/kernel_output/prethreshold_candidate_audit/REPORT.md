# EXP045 pre-threshold candidate edge audit

This is a label-based diagnostic. It does not alter the graph or claim official CV/LB.
Threshold=0.48; source budget=2; target budget=1; node-match radius=7.0 um.

GT division events: 5; GT parent→daughter edges: 10.

| status | count |
|---|---:|
| final_candidate | 6 |
| prethreshold_topk_only | 4 |

| video | event | daughter | raw probability | source rank | target rank | threshold | source budget | target budget | final | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 44b6_12dfb391 | 1 | 1 | 0.8025423884391785 | 1 | 1 | 1 | 0 | 0 | 1 | final_candidate |
| 44b6_12dfb391 | 1 | 2 | 0.18794992566108704 | 2 | 2 | 0 | 0 | 0 | 0 | prethreshold_topk_only |
| 44b6_267148e4 | 1 | 1 | 0.3561449348926544 | 2 | 1 | 0 | 0 | 0 | 0 | prethreshold_topk_only |
| 44b6_267148e4 | 1 | 2 | 0.8018506765365601 | 1 | 1 | 1 | 0 | 0 | 1 | final_candidate |
| 6bba_062c8d37 | 1 | 1 | 0.9036263227462769 | 1 | 1 | 1 | 0 | 0 | 1 | final_candidate |
| 6bba_062c8d37 | 1 | 2 | 0.8140435218811035 | 2 | 1 | 1 | 0 | 0 | 1 | final_candidate |
| 6bba_07e24132 | 1 | 1 | 0.6342024803161621 | 1 | 1 | 1 | 0 | 0 | 1 | final_candidate |
| 6bba_07e24132 | 1 | 2 | 0.012863502837717533 | 8 | 4 | 0 | 0 | 0 | 0 | prethreshold_topk_only |
| 6bba_07e24132 | 2 | 1 | 0.4062281847000122 | 3 | 1 | 0 | 0 | 0 | 0 | prethreshold_topk_only |
| 6bba_07e24132 | 2 | 2 | 0.7463145852088928 | 1 | 1 | 1 | 0 | 0 | 1 | final_candidate |
