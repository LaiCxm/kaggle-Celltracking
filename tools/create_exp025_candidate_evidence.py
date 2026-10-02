"""Create EXP025 FOCUS3D division-candidate evidence calibration notebook."""

from __future__ import annotations

import base64
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EXP" / "EXP025" / "CELL_calibrate_focus3d_division_candidates.ipynb"
EMBEDDED_FOCUS_MODULES = (
    "__init__.py",
    "observations.py",
    "candidates.py",
    "runtime.py",
    "pipeline.py",
    "division_audit.py",
    "candidate_scoring.py",
    "gate_sweep.py",
    "global_selection.py",
)


def cell(source: str, kind: str = "code", cell_id: str | None = None) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    if cell_id:
        result["id"] = cell_id
    return result


def validate_embedded_modules() -> None:
    """Ensure package initialisation cannot import a module omitted from the notebook."""

    package_dir = ROOT / "tools" / "focus_first"
    embedded = set(EMBEDDED_FOCUS_MODULES)
    missing_files = sorted(name for name in embedded if not (package_dir / name).is_file())
    if missing_files:
        raise FileNotFoundError(f"Embedded FOCUS module files are missing: {missing_files}")

    tree = ast.parse((package_dir / "__init__.py").read_text(encoding="utf-8"))
    required = {
        f"{node.module.split('.', 1)[0]}.py"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
    }
    omitted = sorted(required - embedded)
    if omitted:
        raise RuntimeError(f"focus_first package imports modules omitted from notebook: {omitted}")


def embedded_modules() -> str:
    validate_embedded_modules()
    lines = [
        "import base64, sys",
        "from pathlib import Path",
        "pkg = Path('/kaggle/working/focus_first')",
        "pkg.mkdir(parents=True, exist_ok=True)",
    ]
    for name in EMBEDDED_FOCUS_MODULES:
        payload = base64.b64encode((ROOT / "tools" / "focus_first" / name).read_bytes()).decode("ascii")
        lines.append(f"(pkg / {name!r}).write_bytes(base64.b64decode({payload!r}))")
    helper_payload = base64.b64encode((ROOT / "tools" / "instance_recall_audit.py").read_bytes()).decode("ascii")
    lines.extend([
        f"Path('/kaggle/working/instance_recall_audit.py').write_bytes(base64.b64decode({helper_payload!r}))",
        "sys.path.insert(0, '/kaggle/working')",
        "from focus_first.observations import CenterObservation, extract_focus_instances, build_unified_nodes",
        "from focus_first.candidates import enumerate_divisions, candidate_to_dict",
        "from focus_first.division_audit import extract_division_events",
        "from focus_first.candidate_scoring import event_ranking_metrics, leave_one_group_out_splits, permute_columns_within_groups",
        "from instance_recall_audit import match_frame_instances",
    ])
    return "\n".join(lines)


SETUP = r'''import contextlib
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
PARENT_RADIUS_UM = 14.0
DAUGHTER_RADIUS_UM = 14.0
TARGET_VIDEO_COUNT = 12
MAX_EVENTS_PER_VIDEO = 3
MASK_PERMUTATION_SEED = 20260917
BOOTSTRAP_ITERATIONS = 10_000
EXCLUDED_STEMS = {
    '44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132',
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
    raise RuntimeError('EXP025 requires CUDA')


def read_gt(path):
    group = zarr.open_group(str(path), mode='r')
    ids = np.asarray(group['nodes/ids'][:], dtype=np.int64)
    t = np.asarray(group['nodes/props/t/values'][:], dtype=np.int64)
    z = np.asarray(group['nodes/props/z/values'][:], dtype=np.float64)
    y = np.asarray(group['nodes/props/y/values'][:], dtype=np.float64)
    x = np.asarray(group['nodes/props/x/values'][:], dtype=np.float64)
    nodes = {
        int(node_id): (int(tt), float(zz), float(yy), float(xx))
        for node_id, tt, zz, yy, xx in zip(ids, t, z, y, x)
    }
    edges = [tuple(map(int, row)) for row in np.asarray(group['edges/ids'][:], dtype=np.int64)]
    return nodes, edges


def spread_select(values, count):
    values = sorted(values)
    if len(values) <= count:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    return [values[int(index)] for index in np.unique(indices)]


all_gt = {}
all_events = {}
groups = defaultdict(list)
for gt_path in sorted((COMP_ROOT / 'train').glob('*.geff')):
    stem = gt_path.stem
    if stem in EXCLUDED_STEMS or not (COMP_ROOT / 'train' / f'{stem}.zarr').exists():
        continue
    nodes, edges = read_gt(gt_path)
    events = extract_division_events(nodes, edges, require_next_frame=True)
    if not events:
        continue
    all_gt[stem] = (nodes, edges)
    all_events[stem] = events
    groups[stem.split('_', 1)[0]].append(stem)

group_names = sorted(groups)
selected_stems = []
base = TARGET_VIDEO_COUNT // len(group_names)
remainder = TARGET_VIDEO_COUNT % len(group_names)
for index, group in enumerate(group_names):
    selected_stems.extend(spread_select(groups[group], base + int(index < remainder)))
selected_set = set(selected_stems)
while len(selected_set) < TARGET_VIDEO_COUNT:
    added = False
    for group in group_names:
        remaining = [stem for stem in sorted(groups[group]) if stem not in selected_set]
        if remaining:
            selected_set.add(spread_select(remaining, 1)[0])
            added = True
            if len(selected_set) == TARGET_VIDEO_COUNT:
                break
    if not added:
        break
selected_stems = sorted(selected_set)
if len(selected_stems) != TARGET_VIDEO_COUNT:
    raise RuntimeError(
        f'Expected {TARGET_VIDEO_COUNT} selected videos, got {len(selected_stems)}: {selected_stems}'
    )

events_by_video = {}
frames_by_video = {}
event_manifest = []
for stem in selected_stems:
    events = all_events[stem]
    chosen_indices = spread_select(list(range(len(events))), min(MAX_EVENTS_PER_VIDEO, len(events)))
    chosen = [events[index] for index in chosen_indices]
    events_by_video[stem] = chosen
    frames = sorted({frame for event in chosen for frame in (int(event.parent_t), int(event.parent_t) + 1)})
    frames_by_video[stem] = frames
    nodes, _ = all_gt[stem]
    for event in chosen:
        event_manifest.append({
            'video': stem, 'event_index': int(event.event_index),
            'parent_id': int(event.parent_id), 'parent_t': int(event.parent_t),
            'daughter1_id': int(event.daughter1_id), 'daughter2_id': int(event.daughter2_id),
            'daughter_t': int(nodes[event.daughter1_id][0]),
        })

print('CUDA:', torch.cuda.get_device_name(0))
print('event-bearing videos available:', len(all_events))
print('selected videos:', selected_stems)
print('selected events:', len(event_manifest), 'selected frames:', sum(map(len, frames_by_video.values())))
display(pd.DataFrame(event_manifest).groupby('video').size().rename('events').reset_index())
'''


PILKWANG = r'''from predict_unet_transformer import PredictConfig, load_model, predict_video

device = torch.device('cuda')
weight = SUPPORT_ROOT / 'weights' / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'
pil_model, window_size, downsample = load_model(weight, device)
pil_cfg = PredictConfig(det_threshold=0.96, det_tta=True, pool_kernel_um=3.0, use_ilp=False)
pilkwang_centers_by_video = {}
pilkwang_seconds = {}
for index, stem in enumerate(selected_stems, 1):
    start = time.perf_counter()
    coords, _ = predict_video(
        pil_model, COMP_ROOT / 'train' / f'{stem}.zarr', device, cfg=pil_cfg,
        window_size=window_size, max_frames=max(frames_by_video[stem]) + 1,
        unet_batch_size=4, downsample=downsample,
    )
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4)
    wanted = set(frames_by_video[stem])
    coords = coords[np.isin(coords[:, 0].astype(np.int64), sorted(wanted))]
    by_t = defaultdict(list)
    for node_id, row in enumerate(coords):
        by_t[int(row[0])].append(CenterObservation(
            node_id=node_id, t=int(row[0]), z=float(row[1]), y=float(row[2]), x=float(row[3]),
        ))
    pilkwang_centers_by_video[stem] = dict(by_t)
    pilkwang_seconds[stem] = time.perf_counter() - start
    print(f'Pilkwang {index}/{len(selected_stems)}', stem, 'nodes', len(coords),
          'seconds', round(pilkwang_seconds[stem], 2), flush=True)
del pil_model
torch.cuda.empty_cache()
'''


FOCUS = r'''import tifffile

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
focus_instances_by_video = {}
focus_seconds = {}
for index, stem in enumerate(selected_stems, 1):
    zarr_arr = zarr.open_group(str(COMP_ROOT / 'train' / f'{stem}.zarr'), mode='r')['0']
    per_frame = {}
    start = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix=f'exp025_{stem}_', dir='/kaggle/working'))
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
            confidence = result.get('confidence_map')
            confidence = None if confidence is None else np.asarray(confidence, dtype=np.float32)
            per_frame[int(t)] = extract_focus_instances(
                instance_map, t=int(t), confidence_map=confidence,
            )
            image_path.unlink(missing_ok=True)
            shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    focus_instances_by_video[stem] = per_frame
    focus_seconds[stem] = time.perf_counter() - start
    print(f'FOCUS3D {index}/{len(selected_stems)}', stem,
          'instances', sum(map(len, per_frame.values())),
          'seconds', round(focus_seconds[stem], 2), flush=True)
'''


CANDIDATES = r'''def build_union_nodes(stem):
    nodes_by_t = {}
    for t in frames_by_video[stem]:
        nodes_by_t[int(t)] = build_unified_nodes(
            pilkwang_centers_by_video[stem].get(int(t), []),
            focus_instances_by_video[stem].get(int(t), []),
            scale_um=tuple(SCALE_UM), match_radius_um=MATCH_RADIUS_UM,
            accept_focus_quality=0.0,
        )
    return nodes_by_t


def match_gt_to_nodes(gt_nodes, nodes_by_t):
    mapping = {}
    for t, predicted in nodes_by_t.items():
        gt_items = [
            (node_id, value[1:]) for node_id, value in gt_nodes.items()
            if int(value[0]) == int(t)
        ]
        pred_items = [(node.proposal_id, node.point) for node in predicted]
        for match in match_frame_instances(
            gt_items, pred_items, scale_zyx_um=SCALE_UM, radius_um=MATCH_RADIUS_UM,
        ):
            mapping[int(match.gt_id)] = str(match.pred_id)
    return mapping


candidate_rows = []
event_rows = []
nodes_by_video = {}
for stem in selected_stems:
    gt_nodes, _ = all_gt[stem]
    nodes_by_t = build_union_nodes(stem)
    nodes_by_video[stem] = nodes_by_t
    node_lookup = {node.proposal_id: node for nodes in nodes_by_t.values() for node in nodes}
    instance_lookup = {
        (int(instance.t), int(instance.label)): instance
        for instances in focus_instances_by_video[stem].values() for instance in instances
    }
    matched = match_gt_to_nodes(gt_nodes, nodes_by_t)
    events = {event.event_index: event for event in events_by_video[stem]}
    for manifest in [row for row in event_manifest if row['video'] == stem]:
        event = events[manifest['event_index']]
        event_id = f"{stem}:{event.event_index}"
        mapped_parent = matched.get(event.parent_id)
        mapped_daughters = frozenset(filter(None, (
            matched.get(event.daughter1_id), matched.get(event.daughter2_id),
        )))
        objects = []
        if mapped_parent is not None and len(mapped_daughters) == 2:
            parent_node = node_lookup[mapped_parent]
            objects = enumerate_divisions(
                [parent_node], nodes_by_t.get(int(event.parent_t) + 1, []),
                instances_by_label=instance_lookup,
                max_parent_distance_um=PARENT_RADIUS_UM,
                max_daughter_distance_um=DAUGHTER_RADIUS_UM,
                scale_um=tuple(SCALE_UM),
            )
        positive_count = 0
        for candidate_index, candidate in enumerate(objects):
            is_positive = int(frozenset((candidate.daughter1_id, candidate.daughter2_id)) == mapped_daughters)
            positive_count += is_positive
            row = {
                'video': stem, 'event_id': event_id, 'event_index': int(event.event_index),
                'candidate_index': candidate_index, 'label': is_positive,
                **candidate_to_dict(candidate),
                'parent_kind': node_lookup[candidate.parent_id].kind,
                'daughter1_kind': node_lookup[candidate.daughter1_id].kind,
                'daughter2_kind': node_lookup[candidate.daughter2_id].kind,
            }
            row['contains_focus_only'] = int(any(
                row[key] == 'focus_only' for key in ('parent_kind', 'daughter1_kind', 'daughter2_kind')
            ))
            candidate_rows.append(row)
        event_rows.append({
            **manifest, 'event_id': event_id,
            'parent_matched': int(mapped_parent is not None),
            'daughter_count_matched': len(mapped_daughters),
            'candidate_count': len(objects), 'positive_candidate_count': positive_count,
        })

candidates_df = pd.DataFrame(candidate_rows)
events_df = pd.DataFrame(event_rows)
if candidates_df.empty:
    raise RuntimeError('No division candidates generated')
rankable_ids = set(events_df.loc[events_df.positive_candidate_count == 1, 'event_id'])
rankable_df = candidates_df[candidates_df.event_id.isin(rankable_ids)].reset_index(drop=True)
print('events total/rankable:', len(events_df), len(rankable_ids))
print('candidates total/rankable:', len(candidates_df), len(rankable_df))
print('FOCUS-only candidates:', int(candidates_df.contains_focus_only.sum()))
display(events_df)
'''


CALIBRATION = r'''from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DISTANCE_FEATURES = [
    'parent_distance_um', 'daughter_distance_um', 'daughter_separation_score',
]
VOLUME_FEATURES = ['volume_balance', 'volume_conservation']
MASK_FEATURES = [
    'parent_union_overlap', 'parent_union_parent_coverage',
    'parent_union_daughter_coverage',
]
ARMS = {
    'distance_only': DISTANCE_FEATURES,
    'centroid_volume': DISTANCE_FEATURES + VOLUME_FEATURES,
    'real_mask': DISTANCE_FEATURES + VOLUME_FEATURES + MASK_FEATURES,
    'permuted_mask': DISTANCE_FEATURES + VOLUME_FEATURES + MASK_FEATURES,
}

permuted_rows = permute_columns_within_groups(
    rankable_df.to_dict('records'), columns=MASK_FEATURES,
    group_key='video', seed=MASK_PERMUTATION_SEED,
)
arm_frames = {
    'distance_only': rankable_df,
    'centroid_volume': rankable_df,
    'real_mask': rankable_df,
    'permuted_mask': pd.DataFrame(permuted_rows),
}


def oof_scores(frame, features):
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    groups = frame.video.astype(str).tolist()
    for fold, (train_idx, valid_idx) in enumerate(leave_one_group_out_splits(groups)):
        train_y = frame.iloc[train_idx].label.to_numpy(dtype=np.int64)
        if len(np.unique(train_y)) != 2:
            raise RuntimeError(f'Fold {fold} training data lacks both labels')
        train_x = frame.iloc[train_idx][features].astype(np.float64).replace([np.inf, -np.inf], np.nan).copy()
        valid_x = frame.iloc[valid_idx][features].astype(np.float64).replace([np.inf, -np.inf], np.nan).copy()
        for feature in features:
            train_missing = train_x[feature].isna()
            valid_missing = valid_x[feature].isna()
            median = train_x.loc[~train_missing, feature].median()
            if not np.isfinite(median):
                median = 0.0
            train_x[f'{feature}__missing'] = train_missing.astype(np.float64)
            valid_x[f'{feature}__missing'] = valid_missing.astype(np.float64)
            train_x[feature] = train_x[feature].fillna(float(median))
            valid_x[feature] = valid_x[feature].fillna(float(median))
        model = Pipeline([
            ('scale', StandardScaler()),
            ('logistic', LogisticRegression(
                C=1.0, class_weight='balanced', max_iter=2000,
                random_state=20260917, solver='liblinear',
            )),
        ])
        model.fit(train_x, train_y)
        scores[valid_idx] = model.predict_proba(valid_x)[:, 1]
    if not np.all(np.isfinite(scores)):
        raise RuntimeError('OOF scoring left missing rows')
    return scores


arm_summaries = []
event_score_rows = []
for arm, features in ARMS.items():
    frame = arm_frames[arm]
    scores = oof_scores(frame, features)
    metrics = event_ranking_metrics(frame.event_id, frame.label, scores)
    arm_summaries.append({
        'arm': arm, 'features': json.dumps(features),
        'candidate_roc_auc': float(roc_auc_score(frame.label, scores)),
        'candidate_average_precision': float(average_precision_score(frame.label, scores)),
        **metrics,
    })
    for index, score in enumerate(scores):
        event_score_rows.append({
            'arm': arm, 'video': frame.iloc[index].video,
            'event_id': frame.iloc[index].event_id,
            'candidate_index': int(frame.iloc[index].candidate_index),
            'label': int(frame.iloc[index].label), 'score': float(score),
        })

scores_df = pd.DataFrame(event_score_rows)


def top1_by_event(arm):
    result = {}
    subset = scores_df[scores_df.arm == arm]
    for event_id, group in subset.groupby('event_id'):
        positive_score = float(group.loc[group.label == 1, 'score'].iloc[0])
        strictly_higher = int(((group.label == 0) & (group.score > positive_score)).sum())
        tied = int(((group.label == 0) & np.isclose(group.score, positive_score, atol=1e-12, rtol=0)).sum())
        rank = 1.0 + strictly_higher + 0.5 * tied
        result[event_id] = float(rank <= 1.0)
    return result


def paired_video_bootstrap(left_arm, right_arm, seed):
    left = top1_by_event(left_arm); right = top1_by_event(right_arm)
    event_video = events_df.set_index('event_id').video.to_dict()
    per_video = defaultdict(list)
    for event_id in sorted(set(left) & set(right)):
        per_video[event_video[event_id]].append(left[event_id] - right[event_id])
    deltas = np.asarray([np.mean(values) for values in per_video.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(BOOTSTRAP_ITERATIONS, len(deltas)), replace=True).mean(axis=1)
    return {
        'left': left_arm, 'right': right_arm,
        'video_mean_top1_delta': float(deltas.mean()),
        'ci95_low': float(np.quantile(sampled, 0.025)),
        'ci95_high': float(np.quantile(sampled, 0.975)),
        'probability_gt_zero': float(np.mean(sampled > 0)),
        'videos': int(len(deltas)), 'iterations': BOOTSTRAP_ITERATIONS,
    }

comparisons = [
    paired_video_bootstrap('real_mask', 'centroid_volume', 20260918),
    paired_video_bootstrap('real_mask', 'permuted_mask', 20260919),
]

out = Path('/kaggle/working/exp025_candidate_evidence')
out.mkdir(parents=True, exist_ok=True)
candidates_df.to_csv(out / 'candidate_rows.csv', index=False)
events_df.to_csv(out / 'event_rows.csv', index=False)
scores_df.to_csv(out / 'oof_candidate_scores.csv', index=False)
pd.DataFrame(arm_summaries).to_csv(out / 'arm_summary.csv', index=False)
summary = {
    'experiment': 'EXP025', 'official_cv': None, 'submission_generated': False,
    'selected_videos': selected_stems, 'events': len(events_df),
    'rankable_events': len(rankable_ids), 'candidate_rows': len(candidates_df),
    'focus_only_candidate_rows': int(candidates_df.contains_focus_only.sum()),
    'mask_feature_definition': 'single motion-compensated parent mask versus daughter-mask union',
    'arms': arm_summaries, 'paired_video_bootstrap': comparisons,
    'runtime_seconds': {'pilkwang': pilkwang_seconds, 'focus3d': focus_seconds},
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

table = pd.DataFrame(arm_summaries)
report = [
    '# EXP025 FOCUS3D division candidate evidence calibration', '',
    'Candidate-ranking diagnostic only; not official CV and not a submission.', '',
    f"Videos={len(selected_stems)}, events={len(events_df)}, rankable={len(rankable_ids)}, "
    f"candidates={len(candidates_df)}, candidates containing FOCUS-only nodes={int(candidates_df.contains_focus_only.sum())}.",
    '', '```text', table.to_string(index=False), '```', '',
    'Paired video bootstrap comparisons:', '', '```json',
    json.dumps(comparisons, indent=2), '```', '',
    'Promotion requires real_mask to improve over centroid_volume and to lose that advantage after within-video mask-feature permutation.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
display(table)
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    cells = [
        cell(
            "# EXP025 FOCUS3D division-candidate evidence calibration\n\n"
            "FOCUS-only instances remain provisional candidates by explicit user override. "
            "This notebook tests whether real 3-D masks improve division-candidate ranking beyond centroids and volumes.",
            "markdown", "title",
        ),
        cell(embedded_modules(), "code", "embedded-modules"),
        cell(SETUP, "code", "setup-and-event-sampling"),
        cell(PILKWANG, "code", "pilkwang-centers"),
        cell(FOCUS, "code", "focus3d-instances"),
        cell(CANDIDATES, "code", "candidate-table"),
        cell(CALIBRATION, "code", "grouped-calibration"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP025 FOCUS3D division candidate evidence"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
