# Provenance

Source: https://github.com/royerlab/kaggle-cell-tracking-competition
Commit: 075fc5f5a52d11077f9dc2b074644618f26939e2 ("Merge pull request #2 from royerlab/metrics-fix")
Retrieved: 2026-07-20
License: BSD 3-Clause (see LICENSE), (c) 2026 Thibaut Goldsborough

Unmodified copy of `src/`, `scripts/`, `metrics.md`, `README.md`, `LICENSE`.
Packaged as a Kaggle dataset only so that offline (no-internet) kernels can
score predictions with the OFFICIAL metric.

## Why this matters

The vendored copy of the metric inside `pilkwang/biohub-tracking-support-pack-50ep-v1`
(module `biohub_tracking`, built 2026-07-08) predates the division-exploit patch
and still contains `_weakly_connected_components`. This copy (module
`tracking_cellmot`, commit 075fc5f) contains the patched division scoring from
commit aa65e90 "updating metric to patch weakly connected component exploit".

Differences measured on 2026-07-20:
- `division_metrics.py`: 450 -> 574 lines, division scoring substantially rewritten
- `metrics.py`: 90 changed lines; edge side adds non-consecutive-edge filtering
  and duplicate-edge dedup (both no-ops for graphs that already satisfy
  consecutive-frame edges and in-degree <= 1)
