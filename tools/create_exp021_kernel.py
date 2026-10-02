"""Generate a standalone EXP021 FOCUS3D-first diagnostic notebook.

The notebook uses only the frozen model APIs and datasets. It does not copy the
EXP016/017 business pipeline or any old post-processing function.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EXP" / "EXP021" / "CELL_infer_focus_first_v1.ipynb"


def cell(source: str, kind: str = "code") -> dict:
    if kind == "markdown":
        return {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    return {"cell_type": kind, "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def focus_modules_cell() -> str:
    lines = [
        "import base64, sys",
        "from pathlib import Path",
        "focus_pkg = Path('/kaggle/working/focus_first')",
        "focus_pkg.mkdir(parents=True, exist_ok=True)",
    ]
    for name in ("__init__.py", "observations.py", "candidates.py", "pipeline.py", "audit.py", "runtime.py"):
        payload = base64.b64encode((ROOT / "tools" / "focus_first" / name).read_bytes()).decode("ascii")
        lines.append(f"(focus_pkg / {name!r}).write_bytes(base64.b64decode({payload!r}))")
    lines += [
        "sys.path.insert(0, '/kaggle/working')",
        "from focus_first.observations import CenterObservation, extract_focus_instances",
        "from focus_first.pipeline import build_frame_candidate_layers",
        "from focus_first.audit import summarize_frame",
    ]
    return "\n".join(lines)


SETUP = r'''from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("missing path:\n" + "\n".join(map(str, paths)))


COMP_ROOT = first_existing([
    Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
    Path('/kaggle/input/biohub-cell-tracking-during-development'),
])
SUPPORT_ROOT = first_existing([
    Path('/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1'),
    Path('/kaggle/input/pilkwang/biohub-tracking-support-pack-50ep-v1'),
])
FOCUS_ROOT = first_existing([
    Path('/kaggle/input/datasets/qiweiyin/focus3d-nuclei-runtime'),
    Path('/kaggle/input/focus3d-nuclei-runtime'),
])

SAMPLE_ID = '6bba_337b1b3a'
MAX_EVAL_FRAMES = 8
MATCH_RADIUS_UM = 7.0
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
SAMPLE_PATH = COMP_ROOT / 'train' / f'{SAMPLE_ID}.zarr'
GT_PATH = COMP_ROOT / 'train' / f'{SAMPLE_ID}.geff'
assert SAMPLE_PATH.exists() and GT_PATH.exists()

try:
    import zarr
except Exception:
    wheels = SUPPORT_ROOT / 'wheels'
    packages = [
        'tracksdata', 'zarr==3.2.1', 'numcodecs==0.15.1', 'donfig==0.8.1.post1',
        'geff==1.2.0.1.1', 'geff-spec==1.1.1', 'polars==1.42.0',
        'polars-runtime-32==1.42.0', 'bidict==0.23.1', 'imagecodecs==2026.6.26',
        'rustworkx==0.18.0', 'ilpy==0.6.0', 'pyscipopt==6.2.1',
    ]
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
         '--find-links', str(wheels), *packages], capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError((result.stdout or '')[-2000:] + (result.stderr or '')[-2000:])
    import zarr

sys.path.insert(0, str(SUPPORT_ROOT / 'repo' / 'src'))
sys.path.insert(0, str(SUPPORT_ROOT / 'repo' / 'scripts'))
import torch
if not torch.cuda.is_available():
    raise RuntimeError('EXP021 requires CUDA')

gt_group = zarr.open_group(str(GT_PATH), mode='r')
gt_times = np.asarray(gt_group['nodes/props/t/values'][:], dtype=np.int64)
EVAL_FRAMES = np.unique(gt_times)[:MAX_EVAL_FRAMES]
PREDICT_MAX_FRAMES = int(EVAL_FRAMES[-1]) + 1
zarr_arr = zarr.open_group(str(SAMPLE_PATH), mode='r')['0']
print('CUDA:', torch.cuda.get_device_name(0))
print('sample:', SAMPLE_ID, 'frames:', EVAL_FRAMES.tolist())
'''


CENTERS = r'''# Frozen Pilkwang detector: center observations only.
from predict_unet_transformer import PredictConfig, load_model, predict_video
from focus_first.observations import CenterObservation

device = torch.device('cuda')
weight = SUPPORT_ROOT / 'weights' / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'
assert weight.exists(), weight
model, window, downsample = load_model(weight, device)
cfg = PredictConfig(det_threshold=0.96, det_tta=True, pool_kernel_um=3.0, use_ilp=False)
start = time.perf_counter()
coords, _ = predict_video(
    model, SAMPLE_PATH, device, cfg=cfg, window_size=window,
    max_frames=PREDICT_MAX_FRAMES, unet_batch_size=4, downsample=downsample,
)
pil_seconds = time.perf_counter() - start
coords = np.asarray(coords, dtype=np.float32)
coords = coords[np.isin(coords[:, 0].astype(np.int64), EVAL_FRAMES)]
centers_by_t = defaultdict(list)
for idx, row in enumerate(coords):
    centers_by_t[int(row[0])].append(CenterObservation(
        node_id=int(idx), t=int(row[0]), z=float(row[1]), y=float(row[2]), x=float(row[3]),
        score=0.0, source='pilkwang',
    ))
print('Pilkwang centers:', len(coords), 'seconds:', round(pil_seconds, 2))
'''


FOCUS = r'''# Original FOCUS3D instance model: retain masks and confidence maps.
import tifffile

FOCUS_CONFIG = FOCUS_ROOT / 'configs' / '3d_test.yaml'
FOCUS_WEIGHTS = first_existing([
    FOCUS_ROOT / 'models' / 'model_final_nuclei.pth',
    FOCUS_ROOT / 'models ' / 'model_final_nuclei.pth',
])
runtime = str(FOCUS_ROOT / 'focus3d_runtime')
if runtime not in sys.path:
    sys.path.insert(0, runtime)
from focus3d.segmentation.FOCUS3D.inference_win import build_predictor, infer_volume, setup_cfg

focus_model = build_predictor(setup_cfg(str(FOCUS_CONFIG), str(FOCUS_WEIGHTS), device='cuda'))
focus_model.eval()
focus_instances_by_t = {}
focus_maps_by_t = {}
start = time.perf_counter()
work = Path(tempfile.mkdtemp(prefix='exp021_focus_', dir='/kaggle/working'))
try:
    for t in EVAL_FRAMES:
        t = int(t)
        frame = np.ascontiguousarray(zarr_arr[t])
        image_path = work / f'{t:04d}.tif'
        output_dir = work / f'out_{t:04d}'
        tifffile.imwrite(str(image_path), frame, metadata={'axes': 'ZYX'})
        with open(os.devnull, 'w') as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                result = infer_volume(
                    image_path=image_path, config_file=FOCUS_CONFIG, weights_path=FOCUS_WEIGHTS,
                    model=focus_model, device='cuda', output_dir=output_dir,
                    z_ratio=float(SCALE_UM[0] / SCALE_UM[1]), lower_percentile=1.0,
                    upper_percentile=99.0, data_loader_num_workers=0, cell_radius=15.0,
                    background_threshold=float(np.median(frame)), stride=[32, 96, 96],
                    batch_size=12, score_thresh=0.6, mask_thresh=0.5,
                    min_edge_area=64, topk_postprocess=300, save_intermediate=False,
                )
        instance_map = np.asarray(result['instance_map'], dtype=np.int32)
        confidence_map = result.get('confidence_map')
        confidence_map = None if confidence_map is None else np.asarray(confidence_map, dtype=np.float32)
        focus_maps_by_t[t] = instance_map
        focus_instances_by_t[t] = extract_focus_instances(
            instance_map, t=t, confidence_map=confidence_map,
        )
        image_path.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
finally:
    shutil.rmtree(work, ignore_errors=True)
focus_seconds = time.perf_counter() - start
print('FOCUS3D instances:', sum(map(len, focus_instances_by_t.values())), 'seconds:', round(focus_seconds, 2))
'''


AUDIT = r'''# Unified nodes and source-independent continuation/division candidates.
layers = build_frame_candidate_layers(
    dict(centers_by_t), focus_instances_by_t,
    scale_um=tuple(float(v) for v in SCALE_UM), match_radius_um=MATCH_RADIUS_UM,
    max_distance_um=14.0, max_daughter_distance_um=14.0,
)
layer_rows = []
for layer in layers:
    row = summarize_frame(layer.nodes, layer.continuations_to_next, layer.divisions_to_next)
    row['t'] = int(layer.t)
    layer_rows.append(row)

def read_gt(path):
    group = zarr.open_group(str(path), mode='r')
    ids = np.asarray(group['nodes/ids'][:], dtype=np.int64)
    t = np.asarray(group['nodes/props/t/values'][:], dtype=np.int64)
    z = np.asarray(group['nodes/props/z/values'][:], dtype=np.float32)
    y = np.asarray(group['nodes/props/y/values'][:], dtype=np.float32)
    x = np.asarray(group['nodes/props/x/values'][:], dtype=np.float32)
    keep = np.isin(t, EVAL_FRAMES)
    nodes = {
        int(i): (int(tt), float(zz), float(yy), float(xx))
        for i, tt, zz, yy, xx, selected in zip(ids, t, z, y, x, keep)
        if selected
    }
    edges = [
        tuple(map(int, row)) for row in np.asarray(group['edges/ids'][:], dtype=np.int64)
        if int(row[0]) in nodes and int(row[1]) in nodes
    ]
    return nodes, edges

gt_nodes, gt_edges = read_gt(GT_PATH)
proposal_nodes = [node for layer in layers for node in layer.nodes]
by_t = defaultdict(list)
for node in proposal_nodes:
    by_t[int(node.t)].append(node)

def nearest(value):
    tt, zz, yy, xx = value
    choices = by_t.get(int(tt), [])
    if not choices:
        return None
    target = np.asarray((zz, yy, xx), dtype=np.float64) * SCALE_UM
    distances = [np.linalg.norm(np.asarray(node.point) * SCALE_UM - target) for node in choices]
    index = int(np.argmin(distances))
    return choices[index] if float(distances[index]) <= MATCH_RADIUS_UM else None

mapped = {node_id: nearest(value) for node_id, value in gt_nodes.items()}
outgoing = defaultdict(list)
for source_id, target_id in gt_edges:
    outgoing[source_id].append(target_id)
division_rows = []
for parent_id, children in sorted(outgoing.items()):
    if len(children) < 2:
        continue
    parent, d1, d2 = (mapped.get(parent_id), mapped.get(children[0]), mapped.get(children[1]))
    found = False
    if parent is not None and d1 is not None and d2 is not None:
        for layer in layers:
            triples = {(c.parent_id, c.daughter1_id, c.daughter2_id) for c in layer.divisions_to_next}
            found |= (parent.proposal_id, d1.proposal_id, d2.proposal_id) in triples
            found |= (parent.proposal_id, d2.proposal_id, d1.proposal_id) in triples
    division_rows.append({
        'gt_parent': int(parent_id), 'gt_daughters': [int(v) for v in children[:2]],
        'parent_detected': int(parent is not None), 'daughter1_detected': int(d1 is not None),
        'daughter2_detected': int(d2 is not None), 'complete_candidate': int(found),
    })

report = {
    'experiment': 'EXP021', 'stage': 'candidate_layer_only', 'sample_id': SAMPLE_ID,
    'eval_frames': [int(v) for v in EVAL_FRAMES],
    'runtime_seconds': {'pilkwang': pil_seconds, 'focus3d': focus_seconds},
    'node_counts': {
        'gt_nodes': len(gt_nodes), 'proposal_nodes': len(proposal_nodes),
        'consensus': sum(node.kind == 'consensus' for node in proposal_nodes),
        'pilkwang_only': sum(node.kind == 'pilkwang_only' for node in proposal_nodes),
        'focus_only': sum(node.kind == 'focus_only' for node in proposal_nodes),
        'matched_gt_nodes': sum(value is not None for value in mapped.values()),
    },
    'division_rows': division_rows, 'formal_cv': None, 'submission_generated': False,
}
out = Path('/kaggle/working/exp021_focus_first_v1')
out.mkdir(parents=True, exist_ok=True)
(out / 'REPORT.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
(out / 'LAYER_SUMMARY.json').write_text(json.dumps(layer_rows, indent=2, default=str), encoding='utf-8')
(out / 'DIVISION_CANDIDATES.json').write_text(json.dumps(division_rows, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, indent=2, default=str))
'''


def main() -> None:
    cells = [
        cell("""# EXP021 FOCUS3D-first V1 candidate audit\n\nStandalone diagnostic: frozen Pilkwang centers and original FOCUS3D instances are converted into one observation layer, then continuation and division candidates are enumerated before any graph solver. No submission is generated.\n""", "markdown"),
        cell(SETUP + "\n" + focus_modules_cell()),
        cell(CENTERS),
        cell(FOCUS),
        cell(AUDIT),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP021 FOCUS3D-first V1 candidate audit"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
