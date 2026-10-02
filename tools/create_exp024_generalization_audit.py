"""Create the EXP024 cross-video sparse-GT instance-recall audit notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EXP" / "EXP024" / "CELL_compare_instance_recall_generalization.ipynb"


def cell(source: str, kind: str = "code", cell_id: str | None = None) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    if cell_id:
        result["id"] = cell_id
    return result


def embedded_modules_cell() -> str:
    modules = {
        "observations.py": ROOT / "tools" / "focus_first" / "observations.py",
        "instance_recall_audit.py": ROOT / "tools" / "instance_recall_audit.py",
    }
    lines = ["import base64, sys", "from pathlib import Path"]
    for name, path in modules.items():
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        lines.append(
            f"Path('/kaggle/working/{name}').write_bytes(base64.b64decode({payload!r}))"
        )
    lines.extend([
        "sys.path.insert(0, '/kaggle/working')",
        "from observations import extract_focus_instances",
        "from instance_recall_audit import match_frame_instances, paired_hit_categories, bootstrap_mean_difference",
    ])
    return "\n".join(lines)


SETUP = r'''import contextlib
import csv
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
import pandas as pd


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

SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
MATCH_RADIUS_UM = 7.0
TARGET_VIDEO_COUNT = 12
TARGET_LABELED_NODES = 500
FRAME_POOL_PER_VIDEO = 16
SAMPLING_SEED = 20260917
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260917
EXCLUDED_STEMS = {
    '44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37',
    '6bba_07e24132', '6bba_337b1b3a',
}

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
    raise RuntimeError('EXP024 requires CUDA')


def read_gt(path):
    group = zarr.open_group(str(path), mode='r')
    ids = np.asarray(group['nodes/ids'][:], dtype=np.int64)
    t = np.asarray(group['nodes/props/t/values'][:], dtype=np.int64)
    z = np.asarray(group['nodes/props/z/values'][:], dtype=np.float64)
    y = np.asarray(group['nodes/props/y/values'][:], dtype=np.float64)
    x = np.asarray(group['nodes/props/x/values'][:], dtype=np.float64)
    return {
        int(node_id): (int(tt), float(zz), float(yy), float(xx))
        for node_id, tt, zz, yy, xx in zip(ids, t, z, y, x)
    }


def evenly_spaced_frames(frames, count):
    frames = np.asarray(sorted(set(int(v) for v in frames)), dtype=np.int64)
    if len(frames) <= count:
        return frames.tolist()
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(np.int64)
    return frames[np.unique(indices)].tolist()


def spread_select(values, count):
    values = sorted(values)
    if len(values) <= count:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    return [values[int(index)] for index in np.unique(indices)]


all_gt_nodes_by_video = {}
available_frames_by_video = {}
for gt_path in sorted((COMP_ROOT / 'train').glob('*.geff')):
    stem = gt_path.stem
    if stem in EXCLUDED_STEMS or not (COMP_ROOT / 'train' / f'{stem}.zarr').exists():
        continue
    nodes = read_gt(gt_path)
    frames = sorted({int(value[0]) for value in nodes.values()})
    if nodes and frames:
        all_gt_nodes_by_video[stem] = nodes
        available_frames_by_video[stem] = frames

if not all_gt_nodes_by_video:
    raise RuntimeError('No previously unseen labeled videos were found')

# Select a small cross-video panel, balanced across embryo prefixes. Within
# each video, build a temporally spread frame pool and shuffle it with a fixed
# seed. Round-robin frame selection prevents any single dense video from
# supplying the full target.
groups = defaultdict(list)
for stem in sorted(all_gt_nodes_by_video):
    groups[stem.split('_', 1)[0]].append(stem)
group_names = sorted(groups)
selected_stems = []
base = TARGET_VIDEO_COUNT // len(group_names)
remainder = TARGET_VIDEO_COUNT % len(group_names)
for index, group_name in enumerate(group_names):
    take = min(len(groups[group_name]), base + int(index < remainder))
    selected_stems.extend(spread_select(groups[group_name], take))
if len(selected_stems) < min(TARGET_VIDEO_COUNT, len(all_gt_nodes_by_video)):
    leftovers = sorted(set(all_gt_nodes_by_video) - set(selected_stems))
    selected_stems.extend(spread_select(
        leftovers,
        min(TARGET_VIDEO_COUNT, len(all_gt_nodes_by_video)) - len(selected_stems),
    ))
selected_stems = sorted(set(selected_stems))
if len(selected_stems) != min(TARGET_VIDEO_COUNT, len(all_gt_nodes_by_video)):
    raise AssertionError(('selected video count', len(selected_stems)))

frame_pools = {}
for index, stem in enumerate(selected_stems):
    pool = evenly_spaced_frames(
        available_frames_by_video[stem],
        min(FRAME_POOL_PER_VIDEO, len(available_frames_by_video[stem])),
    )
    rng = np.random.default_rng(SAMPLING_SEED + index)
    rng.shuffle(pool)
    frame_pools[stem] = pool

frames_by_video = defaultdict(list)
selected_labeled_nodes = 0
stop = False
for rank in range(FRAME_POOL_PER_VIDEO):
    for stem in selected_stems:
        if rank >= len(frame_pools[stem]):
            continue
        t = int(frame_pools[stem][rank])
        frames_by_video[stem].append(t)
        selected_labeled_nodes += sum(
            int(value[0]) == t for value in all_gt_nodes_by_video[stem].values()
        )
        if selected_labeled_nodes >= TARGET_LABELED_NODES:
            stop = True
            break
    if stop:
        break
if selected_labeled_nodes < TARGET_LABELED_NODES:
    raise RuntimeError(
        f'Frame pools contain only {selected_labeled_nodes} labeled nodes; '
        f'target is {TARGET_LABELED_NODES}'
    )
frames_by_video = {
    stem: sorted(frames) for stem, frames in frames_by_video.items() if frames
}
gt_nodes_by_video = {
    stem: all_gt_nodes_by_video[stem] for stem in frames_by_video
}
selected_frame_count = sum(map(len, frames_by_video.values()))

selection_manifest = []
for stem, frames in frames_by_video.items():
    nodes = gt_nodes_by_video[stem]
    for t in frames:
        selection_manifest.append({
            'video': stem,
            't': int(t),
            'gt_nodes': sum(int(value[0]) == int(t) for value in nodes.values()),
            'available_labeled_frames': len(available_frames_by_video[stem]),
        })

print('CUDA:', torch.cuda.get_device_name(0))
print('available unseen labeled videos:', len(all_gt_nodes_by_video))
print('selected videos:', len(frames_by_video), sorted(frames_by_video))
print('target labeled nodes:', TARGET_LABELED_NODES)
print('selected frames:', selected_frame_count)
print('selected labeled nodes:', sum(row['gt_nodes'] for row in selection_manifest))
display(pd.DataFrame(selection_manifest).groupby('video').agg(
    frames=('t', 'count'), gt_nodes=('gt_nodes', 'sum'),
    available_labeled_frames=('available_labeled_frames', 'first'),
).reset_index())
'''


PILKWANG = r'''from predict_unet_transformer import PredictConfig, load_model, predict_video

device = torch.device('cuda')
weight = SUPPORT_ROOT / 'weights' / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'
assert weight.exists(), weight
pil_model, window_size, downsample = load_model(weight, device)
pil_cfg = PredictConfig(det_threshold=0.96, det_tta=True, pool_kernel_um=3.0, use_ilp=False)
pilkwang_nodes_by_video = {}
pilkwang_seconds_by_video = {}

for video_index, stem in enumerate(sorted(frames_by_video), start=1):
    sample_path = COMP_ROOT / 'train' / f'{stem}.zarr'
    max_frames = max(frames_by_video[stem]) + 1
    start = time.perf_counter()
    coords, _ = predict_video(
        pil_model, sample_path, device, cfg=pil_cfg,
        window_size=window_size, max_frames=max_frames,
        unet_batch_size=4, downsample=downsample,
    )
    elapsed = time.perf_counter() - start
    wanted = set(frames_by_video[stem])
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4)
    coords = coords[np.isin(coords[:, 0].astype(np.int64), sorted(wanted))]
    by_t = defaultdict(list)
    for index, row in enumerate(coords):
        t = int(row[0])
        by_t[t].append((f'p:{t}:{index}', (t, float(row[1]), float(row[2]), float(row[3]))))
    pilkwang_nodes_by_video[stem] = {int(t): by_t[int(t)] for t in frames_by_video[stem]}
    pilkwang_seconds_by_video[stem] = float(elapsed)
    print(
        f'Pilkwang {video_index}/{len(frames_by_video)}', stem,
        'selected_frames', len(wanted), 'predicted_through', max_frames,
        'nodes', len(coords), 'seconds', round(elapsed, 2), flush=True,
    )

del pil_model
torch.cuda.empty_cache()
'''


FOCUS3D = r'''import tifffile

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
focus_nodes_by_video = {}
focus_seconds_by_video = {}

for video_index, stem in enumerate(sorted(frames_by_video), start=1):
    sample_path = COMP_ROOT / 'train' / f'{stem}.zarr'
    zarr_arr = zarr.open_group(str(sample_path), mode='r')['0']
    per_frame = {}
    start = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix=f'exp024_focus_{stem}_', dir='/kaggle/working'))
    try:
        for t in frames_by_video[stem]:
            frame = np.ascontiguousarray(zarr_arr[int(t)])
            image_path = work / f'{int(t):04d}.tif'
            output_dir = work / f'out_{int(t):04d}'
            tifffile.imwrite(str(image_path), frame, metadata={'axes': 'ZYX'})
            with open(os.devnull, 'w') as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    result = infer_volume(
                        image_path=image_path, config_file=FOCUS_CONFIG,
                        weights_path=FOCUS_WEIGHTS, model=focus_model, device='cuda',
                        output_dir=output_dir, z_ratio=float(SCALE_UM[0] / SCALE_UM[1]),
                        lower_percentile=1.0, upper_percentile=99.0,
                        data_loader_num_workers=0, cell_radius=15.0,
                        background_threshold=float(np.median(frame)), stride=[32, 96, 96],
                        batch_size=12, score_thresh=0.6, mask_thresh=0.5,
                        min_edge_area=64, topk_postprocess=300, save_intermediate=False,
                    )
            instance_map = np.asarray(result['instance_map'], dtype=np.int32)
            confidence_map = result.get('confidence_map')
            confidence_map = None if confidence_map is None else np.asarray(confidence_map, dtype=np.float32)
            instances = extract_focus_instances(
                instance_map, t=int(t), confidence_map=confidence_map,
            )
            per_frame[int(t)] = [
                (
                    f'f:{int(t)}:{int(instance.label)}',
                    (int(t), float(instance.centroid[0]), float(instance.centroid[1]), float(instance.centroid[2])),
                )
                for instance in instances
            ]
            del instances, instance_map, confidence_map, result, frame
            image_path.unlink(missing_ok=True)
            shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    elapsed = time.perf_counter() - start
    focus_nodes_by_video[stem] = per_frame
    focus_seconds_by_video[stem] = float(elapsed)
    print(
        f'FOCUS3D {video_index}/{len(frames_by_video)}', stem,
        'frames', len(per_frame), 'nodes', sum(map(len, per_frame.values())),
        'seconds', round(elapsed, 2), flush=True,
    )
'''


AUDIT = r'''def match_video(gt_nodes, pred_by_t, frames):
    matched = {}
    for t in frames:
        gt_items = [
            (node_id, value[1:]) for node_id, value in gt_nodes.items()
            if int(value[0]) == int(t)
        ]
        pred_items = [(pred_id, value[1:]) for pred_id, value in pred_by_t.get(int(t), [])]
        for item in match_frame_instances(
            gt_items, pred_items,
            scale_zyx_um=SCALE_UM, radius_um=MATCH_RADIUS_UM,
        ):
            matched[item.gt_id] = item
    return matched


per_gt_rows = []
per_frame_rows = []
per_video_rows = []
all_pil_distances = []
all_focus_distances = []

for stem in sorted(frames_by_video):
    frames = frames_by_video[stem]
    selected = set(frames)
    gt_nodes = {
        node_id: value for node_id, value in gt_nodes_by_video[stem].items()
        if int(value[0]) in selected
    }
    pil_matches = match_video(gt_nodes, pilkwang_nodes_by_video[stem], frames)
    focus_matches = match_video(gt_nodes, focus_nodes_by_video[stem], frames)
    categories = paired_hit_categories(gt_nodes, pil_matches, focus_matches)
    common_ids = set(pil_matches) & set(focus_matches)
    pil_distances = [item.distance_um for item in pil_matches.values()]
    focus_distances = [item.distance_um for item in focus_matches.values()]
    all_pil_distances.extend(pil_distances)
    all_focus_distances.extend(focus_distances)

    for node_id, value in sorted(gt_nodes.items(), key=lambda item: (item[1][0], item[0])):
        pil_item = pil_matches.get(node_id)
        focus_item = focus_matches.get(node_id)
        if pil_item is not None and focus_item is not None:
            category = 'both_hit'
        elif pil_item is not None:
            category = 'pilkwang_only_hit'
        elif focus_item is not None:
            category = 'focus3d_only_hit'
        else:
            category = 'both_miss'
        per_gt_rows.append({
            'video': stem, 't': int(value[0]), 'gt_node_id': int(node_id),
            'gt_z': value[1], 'gt_y': value[2], 'gt_x': value[3],
            'pilkwang_hit': int(pil_item is not None),
            'focus3d_hit': int(focus_item is not None),
            'pilkwang_distance_um': None if pil_item is None else pil_item.distance_um,
            'focus3d_distance_um': None if focus_item is None else focus_item.distance_um,
            'category': category,
        })

    for t in frames:
        frame_gt_ids = {node_id for node_id, value in gt_nodes.items() if int(value[0]) == int(t)}
        frame_pil_hits = frame_gt_ids & set(pil_matches)
        frame_focus_hits = frame_gt_ids & set(focus_matches)
        per_frame_rows.append({
            'video': stem, 't': int(t), 'gt_nodes': len(frame_gt_ids),
            'pilkwang_pred_nodes': len(pilkwang_nodes_by_video[stem].get(int(t), [])),
            'focus3d_pred_nodes': len(focus_nodes_by_video[stem].get(int(t), [])),
            'pilkwang_hits': len(frame_pil_hits), 'focus3d_hits': len(frame_focus_hits),
            'pilkwang_misses': len(frame_gt_ids - frame_pil_hits),
            'focus3d_misses': len(frame_gt_ids - frame_focus_hits),
        })

    common_deltas = [
        pil_matches[node_id].distance_um - focus_matches[node_id].distance_um
        for node_id in common_ids
    ]
    gt_count = len(gt_nodes)
    pil_recall = len(pil_matches) / gt_count
    focus_recall = len(focus_matches) / gt_count
    per_video_rows.append({
        'video': stem, 'frames': len(frames), 'gt_nodes': gt_count,
        'pilkwang_hits': len(pil_matches), 'focus3d_hits': len(focus_matches),
        'pilkwang_recall': pil_recall, 'focus3d_recall': focus_recall,
        'recall_diff_pil_minus_focus': pil_recall - focus_recall,
        **categories,
        'pilkwang_mean_distance_um': float(np.mean(pil_distances)) if pil_distances else None,
        'focus3d_mean_distance_um': float(np.mean(focus_distances)) if focus_distances else None,
        'common_hit_count': len(common_ids),
        'common_mean_distance_diff_pil_minus_focus_um': (
            float(np.mean(common_deltas)) if common_deltas else None
        ),
        'pilkwang_pred_nodes': sum(map(len, pilkwang_nodes_by_video[stem].values())),
        'focus3d_pred_nodes': sum(map(len, focus_nodes_by_video[stem].values())),
        'pilkwang_seconds': pilkwang_seconds_by_video[stem],
        'focus3d_seconds': focus_seconds_by_video[stem],
    })

all_gt_ids = [(row['video'], row['gt_node_id']) for row in per_gt_rows]
pil_hits_global = {
    (row['video'], row['gt_node_id']) for row in per_gt_rows if row['pilkwang_hit']
}
focus_hits_global = {
    (row['video'], row['gt_node_id']) for row in per_gt_rows if row['focus3d_hit']
}
global_categories = paired_hit_categories(all_gt_ids, pil_hits_global, focus_hits_global)
gt_total = len(per_gt_rows)
pil_total = len(pil_hits_global)
focus_total = len(focus_hits_global)

recall_bootstrap = bootstrap_mean_difference(
    [row['recall_diff_pil_minus_focus'] for row in per_video_rows],
    iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED,
)
distance_video_differences = [
    row['common_mean_distance_diff_pil_minus_focus_um'] for row in per_video_rows
    if row['common_mean_distance_diff_pil_minus_focus_um'] is not None
]
distance_bootstrap = bootstrap_mean_difference(
    distance_video_differences,
    iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED + 1,
) if distance_video_differences else None

summary = {
    'experiment': 'EXP024',
    'scope': {
        'excluded_previously_inspected_videos': sorted(EXCLUDED_STEMS),
        'evaluated_videos': len(per_video_rows),
        'selected_frames': len(per_frame_rows),
        'target_video_count': TARGET_VIDEO_COUNT,
        'target_labeled_nodes': TARGET_LABELED_NODES,
        'frame_pool_per_video': FRAME_POOL_PER_VIDEO,
        'sampling_seed': SAMPLING_SEED,
        'gt_labeled_nodes': gt_total,
        'sparse_gt_warning': 'Unmatched predictions are not false positives.',
    },
    'pilkwang': {
        'hits': pil_total, 'misses': gt_total - pil_total,
        'micro_recall': pil_total / gt_total,
        'mean_match_distance_um': float(np.mean(all_pil_distances)),
        'median_match_distance_um': float(np.median(all_pil_distances)),
        'p95_match_distance_um': float(np.quantile(all_pil_distances, 0.95)),
        'pred_nodes': sum(row['pilkwang_pred_nodes'] for row in per_video_rows),
    },
    'focus3d': {
        'hits': focus_total, 'misses': gt_total - focus_total,
        'micro_recall': focus_total / gt_total,
        'mean_match_distance_um': float(np.mean(all_focus_distances)),
        'median_match_distance_um': float(np.median(all_focus_distances)),
        'p95_match_distance_um': float(np.quantile(all_focus_distances, 0.95)),
        'pred_nodes': sum(row['focus3d_pred_nodes'] for row in per_video_rows),
    },
    'paired_hit_categories': global_categories,
    'video_level_recall_difference_pil_minus_focus': recall_bootstrap,
    'video_level_common_hit_distance_difference_pil_minus_focus_um': distance_bootstrap,
    'video_direction_counts': {
        'pilkwang_higher_recall': sum(row['recall_diff_pil_minus_focus'] > 0 for row in per_video_rows),
        'equal_recall': sum(row['recall_diff_pil_minus_focus'] == 0 for row in per_video_rows),
        'focus3d_higher_recall': sum(row['recall_diff_pil_minus_focus'] < 0 for row in per_video_rows),
    },
}

OUT_DIR = Path('/kaggle/working/exp024_generalization')
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(selection_manifest).to_csv(OUT_DIR / 'selection_manifest.csv', index=False)
pd.DataFrame(per_gt_rows).to_csv(OUT_DIR / 'per_gt.csv', index=False)
pd.DataFrame(per_frame_rows).to_csv(OUT_DIR / 'per_frame.csv', index=False)
pd.DataFrame(per_video_rows).to_csv(OUT_DIR / 'per_video.csv', index=False)
(OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

report = [
    '# EXP024 cross-video sparse-GT instance-recall audit', '',
    'This is a labeled-node recall audit, not a full confusion matrix or official CV.', '',
    f"Videos: {len(per_video_rows)}; selected frames: {len(per_frame_rows)}; labeled GT nodes: {gt_total}.", '',
    '| source | hits | misses | micro recall | mean match distance (um) |',
    '|---|---:|---:|---:|---:|',
    f"| Pilkwang | {pil_total} | {gt_total-pil_total} | {pil_total/gt_total:.6f} | {np.mean(all_pil_distances):.4f} |",
    f"| FOCUS3D | {focus_total} | {gt_total-focus_total} | {focus_total/gt_total:.6f} | {np.mean(all_focus_distances):.4f} |",
    '', 'Paired GT outcomes: `' + json.dumps(global_categories, sort_keys=True) + '`.',
    '', 'Primary video-level recall difference (Pilkwang - FOCUS3D): `' + json.dumps(recall_bootstrap, sort_keys=True) + '`.',
    '', 'Unmatched predictions cannot be interpreted as false positives because GT is sparse.',
]
(OUT_DIR / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')

display(pd.DataFrame(per_video_rows))
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in OUT_DIR.iterdir()))
'''


cells = [
    cell(
        "# EXP024 Pilkwang vs FOCUS3D cross-video generalization audit\n\n"
        "Evaluate sparse-GT labeled-node recall on previously unseen videos and evenly sampled labeled frames. "
        "No edge, division, ILP, or official-score stage is run. Unmatched predictions are not treated as false positives.",
        kind="markdown", cell_id="title",
    ),
    cell(embedded_modules_cell(), cell_id="embedded-modules"),
    cell(SETUP, cell_id="setup-and-sampling"),
    cell(PILKWANG, cell_id="pilkwang-inference"),
    cell(FOCUS3D, cell_id="focus3d-inference"),
    cell(AUDIT, cell_id="paired-audit"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"title": "EXP024 Pilkwang vs FOCUS3D generalization audit"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
