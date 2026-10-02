# Usage

## 1. Attach and import (offline kernels)

Kaggle mounts private datasets under `/kaggle/input/datasets/<owner>/<slug>/` and
public ones under `/kaggle/input/<slug>/`, so search for the package rather than
hard-coding a path:

```python
import sys, glob, os

cands = []
for pat in ("/kaggle/input/*/*/*/src/tracking_cellmot/metrics.py",
            "/kaggle/input/*/*/src/tracking_cellmot/metrics.py",
            "/kaggle/input/**/tracking_cellmot/metrics.py"):
    cands += glob.glob(pat, recursive=True)
assert cands, "dataset not attached"
SRC = os.path.dirname(os.path.dirname(cands[0]))   # .../src
sys.path.insert(0, SRC)

from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise
```

## 2. Verify you actually loaded the PATCHED metric

Worth doing explicitly. Several public support packs vendor a copy of this code
that predates the 2026-07-17 division patch, and a silent fallback to the old
metric is easy to miss:

```python
import inspect, tracking_cellmot.division_metrics as dm
assert "_weakly_connected_components" not in inspect.getsource(dm), "PRE-patch metric loaded!"
```

## 3. Score a submission CSV against train ground truth

The repo ships the full round trip the hosts document -
predict -> geffs -> CSV (upload) -> CSV -> geffs -> score. No images are loaded.

```bash
REPO=$(dirname $SRC)          # .../tracking_cellmot_075fc5f
python $REPO/scripts/csv_to_geffs.py --csv submission.csv --out-dir pred_geffs
PYTHONPATH=$SRC python $REPO/scripts/evaluate.py \
    --pred-dir pred_geffs \
    --gt-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train
```

`evaluate.py` scores every dataset present in **both** directories and reports the
run-level score: sample-size-weighted adjusted edge Jaccard plus
`0.1 x` division Jaccard.

## 4. Score a graph directly in Python

```python
from tracking_cellmot.io import open_dataset
from geff import GeffMetadata

ds  = open_dataset(f"{TRAIN}/{sample_id}.zarr", normalize=False,
                   load_image=False, require_tracks=True)
meta = GeffMetadata.read(f"{TRAIN}/{sample_id}.geff")

er  = evaluate(pred_graph, ds.tracks, scale=ds.scale, max_distance=7.0)
rec = node_recall(pred_graph, ds.tracks)
m   = per_sample_metrics(er=er,
                         n_total=float(meta.extra["estimated_number_of_nodes"]),
                         node_recall=rec)
print(m["edge_jaccard"], m["adj_edge_jaccard"])
```

## Gotchas

- `ILPSolver.solve()` returns a `GraphView`, and `evaluate()` internally calls
  `.copy()`, which `GraphView` does not support. Call `.detach()` first.
- Aggregation is not a plain mean: `adj_edge_jaccard` is weight-averaged across
  datasets by `TP + FP + FN`, while `division_jaccard` is pooled (micro) across
  datasets and turned into a single Jaccard afterwards.
- `adj_edge_jaccard = max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))` where
  `N_true` is `estimated_number_of_nodes` from the GEFF metadata, i.e. ALL cells
  including unannotated ones. When `N_pred < N_true` the multiplier exceeds 1,
  so the adjusted value can be greater than the raw Jaccard.
- Train labels are sparse. Edges whose endpoints match no annotated GT node are
  ignored entirely, so a local score computed on train movies is not a reliable
  proxy for the leaderboard - treat it as a diagnostic, not a predictor.
