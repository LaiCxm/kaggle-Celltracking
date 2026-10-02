"""Create the EXP021 division-event candidate recall audit notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EXP" / "EXP021" / "CELL_infer_focus_first_division_audit_v1.ipynb"


def cell(source: str, kind: str = "code", cell_id: str | None = None) -> dict:
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    if cell_id:
        result["id"] = cell_id
    return result


def modules_cell() -> str:
    lines = [
        "import base64, sys",
        "from pathlib import Path",
        "focus_pkg = Path('/kaggle/working/focus_first')",
        "focus_pkg.mkdir(parents=True, exist_ok=True)",
    ]
    for name in (
        "__init__.py",
        "observations.py",
        "candidates.py",
        "pipeline.py",
        "audit.py",
        "runtime.py",
        "division_audit.py",
    ):
        payload = base64.b64encode(
            (ROOT / "tools" / "focus_first" / name).read_bytes()
        ).decode("ascii")
        lines.append(
            f"(focus_pkg / {name!r}).write_bytes(base64.b64decode({payload!r}))"
        )
    lines += [
        "sys.path.insert(0, '/kaggle/working')",
        "from focus_first.observations import CenterObservation, UnifiedNode, extract_focus_instances, build_unified_nodes",
        "from focus_first.candidates import enumerate_divisions",
        "from focus_first.division_audit import extract_division_events, match_nodes_by_frame, has_complete_division_candidate",
    ]
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
from dataclasses import replace
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

SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
MATCH_RADIUS_UM = 7.0
MAX_PARENT_DISTANCE_UM = 14.0
MAX_DAUGHTER_DISTANCE_UM = 14.0
LEGACY_PARENT_DISTANCE_UM = 9.9
WINDOW_BEFORE = 2
WINDOW_AFTER = 3
GT_STEMS = (
    '44b6_12dfb391',
    '44b6_267148e4',
    '6bba_062c8d37',
    '6bba_07e24132',
)

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
    raise RuntimeError('EXP021 division audit requires CUDA')


def read_gt(path):
    group = zarr.open_group(str(path), mode='r')
    ids = np.asarray(group['nodes/ids'][:], dtype=np.int64)
    t = np.asarray(group['nodes/props/t/values'][:], dtype=np.int64)
    z = np.asarray(group['nodes/props/z/values'][:], dtype=np.float32)
    y = np.asarray(group['nodes/props/y/values'][:], dtype=np.float32)
    x = np.asarray(group['nodes/props/x/values'][:], dtype=np.float32)
    nodes = {
        int(node_id): (int(tt), float(zz), float(yy), float(xx))
        for node_id, tt, zz, yy, xx in zip(ids, t, z, y, x)
    }
    edges = [tuple(map(int, row)) for row in np.asarray(group['edges/ids'][:], dtype=np.int64)]
    return nodes, edges


# The four stems below are a fixed, previously inspected panel.  We still
# scan all available labels to report the full training-set event count, but
# keep this first GPU audit bounded and reproducible.
all_train_event_count = 0
all_train_event_stems = []
for gt_path in sorted((COMP_ROOT / 'train').glob('*.geff')):
    scan_nodes, scan_edges = read_gt(gt_path)
    scan_events = extract_division_events(scan_nodes, scan_edges, require_next_frame=True)
    all_train_event_count += len(scan_events)
    if scan_events:
        all_train_event_stems.append(gt_path.stem)

gt_by_video = {}
events_by_video = {}
frames_by_video = {}
event_manifest = []
for stem in GT_STEMS:
    gt_path = COMP_ROOT / 'train' / f'{stem}.geff'
    sample_path = COMP_ROOT / 'train' / f'{stem}.zarr'
    assert gt_path.exists(), gt_path
    assert sample_path.exists(), sample_path
    nodes, edges = read_gt(gt_path)
    events = extract_division_events(nodes, edges, require_next_frame=True)
    if not events:
        raise RuntimeError(f'no official two-daughter event found in {stem}')
    gt_by_video[stem] = (nodes, edges)
    events_by_video[stem] = events
    zarr_arr = zarr.open_group(str(sample_path), mode='r')['0']
    max_t = int(zarr_arr.shape[0]) - 1
    frames = set()
    for event in events:
        window = list(range(
            max(0, int(event.parent_t) - WINDOW_BEFORE),
            min(max_t, int(event.parent_t) + WINDOW_AFTER) + 1,
        ))
        frames.update(window)
        daughters = (event.daughter1_id, event.daughter2_id)
        daughter_frames = [int(nodes[node_id][0]) for node_id in daughters]
        event_manifest.append({
            'video': stem,
            'event_index': int(event.event_index),
            'parent_id': int(event.parent_id),
            'parent_t': int(event.parent_t),
            'daughter_ids': [int(v) for v in daughters],
            'daughter_frames': daughter_frames,
            'window_frames': window,
        })
    frames_by_video[stem] = sorted(frames)

print('CUDA:', torch.cuda.get_device_name(0))
print('all train exact-two-daughter events:', all_train_event_count,
      'event-bearing videos:', len(all_train_event_stems))
print('official GT division events:', json.dumps(event_manifest, indent=2))
'''


CENTERS = r'''from predict_unet_transformer import PredictConfig, load_model, predict_video

device = torch.device('cuda')
weight = SUPPORT_ROOT / 'weights' / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'
assert weight.exists(), weight
pil_model, window_size, downsample = load_model(weight, device)
pil_cfg = PredictConfig(det_threshold=0.96, det_tta=True, pool_kernel_um=3.0, use_ilp=False)
pilkwang_nodes_by_video = {}
pil_seconds_by_video = {}

for stem in GT_STEMS:
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
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords[np.isin(coords[:, 0].astype(np.int64), sorted(wanted))]
    by_t = defaultdict(list)
    for row in coords:
        by_t[int(row[0])].append(tuple(float(v) for v in row[1:]))
    pilkwang_nodes_by_video[stem] = {
        int(t): [
            (f'p:{int(t)}:{index}', (int(t), *point))
            for index, point in enumerate(by_t.get(int(t), []))
        ]
        for t in frames_by_video[stem]
    }
    pil_seconds_by_video[stem] = float(elapsed)
    print('Pilkwang', stem, 'frames', len(frames_by_video[stem]),
          'nodes', sum(map(len, pilkwang_nodes_by_video[stem].values())),
          'seconds', round(elapsed, 2))
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
focus_seconds_by_video = {}

for stem in GT_STEMS:
    sample_path = COMP_ROOT / 'train' / f'{stem}.zarr'
    zarr_arr = zarr.open_group(str(sample_path), mode='r')['0']
    per_frame = {}
    start = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix=f'exp021_focus_{stem}_', dir='/kaggle/working'))
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
            per_frame[int(t)] = extract_focus_instances(
                instance_map, t=int(t), confidence_map=confidence_map,
            )
            image_path.unlink(missing_ok=True)
            shutil.rmtree(output_dir, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    focus_instances_by_video[stem] = per_frame
    elapsed = time.perf_counter() - start
    focus_seconds_by_video[stem] = float(elapsed)
    print('FOCUS3D', stem, 'frames', len(frames_by_video[stem]),
          'instances', sum(map(len, per_frame.values())), 'seconds', round(elapsed, 2))
'''


AUDIT = r'''def make_pil_nodes(nodes_by_t):
    return {
        int(t): [
            UnifiedNode(
                proposal_id=str(node_id), t=int(value[0]), z=float(value[1]),
                y=float(value[2]), x=float(value[3]), kind='pilkwang_only',
                center_ids=(index,), provisional=False,
            )
            for index, (node_id, value) in enumerate(rows)
        ]
        for t, rows in nodes_by_t.items()
    }


def make_focus_nodes(instances_by_t):
    return {
        int(t): [
            UnifiedNode(
                proposal_id=f'f:{int(t)}:{int(instance.label)}', t=int(t),
                z=float(instance.centroid[0]), y=float(instance.centroid[1]),
                x=float(instance.centroid[2]), kind='focus_only',
                focus_label=int(instance.label), focus_key=(int(t), int(instance.label)),
                focus_quality=float(instance.quality), provisional=True,
            )
            for instance in rows
        ]
        for t, rows in instances_by_t.items()
    }


def make_union_nodes(pil_nodes_by_t, focus_instances_by_t):
    centers_by_t = {}
    for t, rows in pil_nodes_by_t.items():
        centers_by_t[int(t)] = [
            CenterObservation(
                node_id=index, t=int(value[0]), z=float(value[1]),
                y=float(value[2]), x=float(value[3]), score=0.0,
            )
            for index, (_, value) in enumerate(rows)
        ]
    return {
        int(t): build_unified_nodes(
            centers_by_t.get(int(t), []), focus_instances_by_t.get(int(t), []),
            scale_um=tuple(float(v) for v in SCALE_UM), match_radius_um=MATCH_RADIUS_UM,
        )
        for t in sorted(set(pil_nodes_by_t) | set(focus_instances_by_t))
    }


def node_records(nodes_by_t):
    return [
        (node.proposal_id, (int(node.t), float(node.z), float(node.y), float(node.x)))
        for rows in nodes_by_t.values() for node in rows
    ]


def node_kind(nodes_by_t, node_id):
    for rows in nodes_by_t.values():
        for node in rows:
            if node.proposal_id == node_id:
                return node.kind
    return None


all_event_rows = []
frame_rows = []
source_nodes_by_video = {}
for stem in GT_STEMS:
    pil_nodes = make_pil_nodes(pilkwang_nodes_by_video[stem])
    focus_nodes = make_focus_nodes(focus_instances_by_video[stem])
    union_nodes = make_union_nodes(pilkwang_nodes_by_video[stem], focus_instances_by_video[stem])
    source_nodes_by_video[stem] = {
        'pilkwang': pil_nodes, 'focus3d': focus_nodes, 'union': union_nodes,
    }
    for t in frames_by_video[stem]:
        frame_rows.append({
            'video': stem, 't': int(t),
            'pilkwang_nodes': len(pil_nodes.get(int(t), [])),
            'focus3d_nodes': len(focus_nodes.get(int(t), [])),
            'union_nodes': len(union_nodes.get(int(t), [])),
            'union_consensus': sum(n.kind == 'consensus' for n in union_nodes.get(int(t), [])),
            'union_pilkwang_only': sum(n.kind == 'pilkwang_only' for n in union_nodes.get(int(t), [])),
            'union_focus_only': sum(n.kind == 'focus_only' for n in union_nodes.get(int(t), [])),
        })

for manifest in event_manifest:
    stem = manifest['video']
    gt_nodes, gt_edges = gt_by_video[stem]
    event = next(item for item in events_by_video[stem] if item.event_index == manifest['event_index'])
    endpoints = (event.parent_id, event.daughter1_id, event.daughter2_id)
    per_source = {}
    for source, nodes_by_t in source_nodes_by_video[stem].items():
        matched, distances = match_nodes_by_frame(
            gt_nodes, node_records(nodes_by_t), scale_um=SCALE_UM,
            radius_um=MATCH_RADIUS_UM,
        )
        mapped = tuple(matched.get(node_id) for node_id in endpoints)
        parent_nodes = [node for node in nodes_by_t.get(int(event.parent_t), [])]
        daughter_nodes = [node for node in nodes_by_t.get(int(event.parent_t) + 1, [])]
        candidates = []
        legacy_candidates = []
        if mapped[0] is not None:
            parent = next((node for node in parent_nodes if node.proposal_id == mapped[0]), None)
            if parent is not None:
                candidate_objects = enumerate_divisions(
                    [parent], daughter_nodes,
                    max_parent_distance_um=MAX_PARENT_DISTANCE_UM,
                    max_daughter_distance_um=MAX_DAUGHTER_DISTANCE_UM,
                    scale_um=tuple(float(v) for v in SCALE_UM),
                )
                candidates = [
                    (item.parent_id, item.daughter1_id, item.daughter2_id)
                    for item in candidate_objects
                ]
                legacy_objects = enumerate_divisions(
                    [parent], daughter_nodes,
                    max_parent_distance_um=LEGACY_PARENT_DISTANCE_UM,
                    max_daughter_distance_um=MAX_DAUGHTER_DISTANCE_UM,
                    scale_um=tuple(float(v) for v in SCALE_UM),
                )
                legacy_candidates = [
                    (item.parent_id, item.daughter1_id, item.daughter2_id)
                    for item in legacy_objects
                ]
        complete = has_complete_division_candidate(
            candidates, mapped[0], mapped[1], mapped[2],
        )
        per_source[source] = {
            'matched': mapped,
            'distances': [distances.get(node_id) for node_id in endpoints],
            'kinds': [node_kind(nodes_by_t, item) if item is not None else None for item in mapped],
            'detected_count': sum(item is not None for item in mapped),
            'candidate_count': len(candidates),
            'complete_candidate': int(complete),
            'legacy_candidate_count': len(legacy_candidates),
            'legacy_complete_candidate': int(has_complete_division_candidate(
                legacy_candidates, mapped[0], mapped[1], mapped[2],
            )),
            'candidates': candidates,
        }
    row = {
        'video': stem,
        'event_index': int(event.event_index),
        'parent_t': int(event.parent_t),
        'parent_id': int(event.parent_id),
        'daughter1_id': int(event.daughter1_id),
        'daughter2_id': int(event.daughter2_id),
        'window_frames': json.dumps(manifest['window_frames']),
    }
    for source, result in per_source.items():
        prefix = source.replace('3d', '3d').replace('pilkwang', 'pil')
        row[f'{prefix}_detected_count'] = int(result['detected_count'])
        row[f'{prefix}_parent_detected'] = int(result['matched'][0] is not None)
        row[f'{prefix}_daughter1_detected'] = int(result['matched'][1] is not None)
        row[f'{prefix}_daughter2_detected'] = int(result['matched'][2] is not None)
        row[f'{prefix}_candidate_count'] = int(result['candidate_count'])
        row[f'{prefix}_complete_candidate'] = int(result['complete_candidate'])
        row[f'{prefix}_legacy_candidate_count'] = int(result['legacy_candidate_count'])
        row[f'{prefix}_legacy_complete_candidate'] = int(result['legacy_complete_candidate'])
        row[f'{prefix}_kinds'] = json.dumps(result['kinds'])
        row[f'{prefix}_distances_um'] = json.dumps(result['distances'])
    row['union_added_endpoint_count_vs_pil'] = int(sum(
        union_id is not None and pil_id is None
        for union_id, pil_id in zip(per_source['union']['matched'], per_source['pilkwang']['matched'])
    ))
    row['union_lost_endpoint_count_vs_pil'] = int(sum(
        pil_id is not None and union_id is None
        for union_id, pil_id in zip(per_source['union']['matched'], per_source['pilkwang']['matched'])
    ))
    row['union_added_complete_candidate_vs_pil'] = int(
        per_source['union']['complete_candidate'] > per_source['pilkwang']['complete_candidate']
    )
    row['focus_only_in_union_complete_candidate'] = int(
        per_source['union']['complete_candidate']
        and any(kind == 'focus_only' for kind in per_source['union']['kinds'])
    )
    all_event_rows.append(row)

summary = {
    'experiment': 'EXP021',
    'stage': 'division_event_candidate_recall_audit',
    'official_cv': None,
    'submission_generated': False,
    'division_definition': 'GT out-degree exactly 2, both daughters at parent_t+1',
    'all_train_event_count': int(all_train_event_count),
    'audited_stems': list(GT_STEMS),
    'audited_event_count': len(all_event_rows),
    'node_match_radius_um': MATCH_RADIUS_UM,
    'max_parent_distance_um': MAX_PARENT_DISTANCE_UM,
    'max_daughter_distance_um': MAX_DAUGHTER_DISTANCE_UM,
    'legacy_parent_distance_um': LEGACY_PARENT_DISTANCE_UM,
    'mask_aware_candidate_inclusion': False,
    'events': len(all_event_rows),
    'runtime_seconds': {
        'pilkwang_by_video': pil_seconds_by_video,
        'focus3d_by_video': focus_seconds_by_video,
    },
    'source_summary': {},
}
for source, prefix in (('pilkwang', 'pil'), ('focus3d', 'focus3d'), ('union', 'union')):
    summary['source_summary'][source] = {
        'endpoint_hits': sum(row[f'{prefix}_detected_count'] for row in all_event_rows),
        'endpoint_total': 3 * len(all_event_rows),
        'complete_candidate_hits': sum(row[f'{prefix}_complete_candidate'] for row in all_event_rows),
        'legacy_complete_candidate_hits': sum(row[f'{prefix}_legacy_complete_candidate'] for row in all_event_rows),
    }
summary['union_added_endpoint_hits_vs_pil'] = sum(
    row['union_added_endpoint_count_vs_pil'] for row in all_event_rows
)
summary['union_added_complete_candidates_vs_pil'] = sum(
    row['union_added_complete_candidate_vs_pil'] for row in all_event_rows
)
summary['union_added_legacy_complete_candidates_vs_pil'] = sum(
    int(row['union_legacy_complete_candidate'] > row['pil_legacy_complete_candidate'])
    for row in all_event_rows
)
summary['union_lost_endpoint_hits_vs_pil'] = sum(
    row['union_lost_endpoint_count_vs_pil'] for row in all_event_rows
)

out = Path('/kaggle/working/exp021_division_audit')
out.mkdir(parents=True, exist_ok=True)
(out / 'EVENT_MANIFEST.json').write_text(json.dumps(event_manifest, indent=2), encoding='utf-8')
(out / 'EVENT_AUDIT.json').write_text(json.dumps(all_event_rows, indent=2, default=str), encoding='utf-8')
(out / 'FRAME_SUMMARY.json').write_text(json.dumps(frame_rows, indent=2, default=str), encoding='utf-8')
(out / 'REPORT.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
for filename, rows in (('event_audit.csv', all_event_rows), ('frame_summary.csv', frame_rows)):
    if rows:
        with (out / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

report_lines = [
    '# EXP021 FOCUS3D-first division candidate recall audit',
    '',
    'This is a candidate-layer diagnostic, not official CV and not a competition submission.',
    'GT events use the official-compatible definition: out-degree exactly 2 and both daughters at t+1.',
    'The 9.9um column is a radius-only historical comparison; it does not reproduce the old Top-10 ranking budget.',
    '',
    f"Events audited: {len(all_event_rows)}; node matching radius: {MATCH_RADIUS_UM} um.",
    '',
    '| source | endpoint hits | endpoint total | direct triples (14/9.9 parent gate) |',
    '|---|---:|---:|---:|',
]
for source, values in summary['source_summary'].items():
    report_lines.append(
        f"| {source} | {values['endpoint_hits']} | {values['endpoint_total']} | "
        f"{values['complete_candidate_hits']} / {values['legacy_complete_candidate_hits']} |"
    )
report_lines += [
    '',
    f"Union endpoint hits added over Pilkwang: {summary['union_added_endpoint_hits_vs_pil']}; "
    f"lost: {summary['union_lost_endpoint_hits_vs_pil']}; "
    f"complete triples added at 14um: {summary['union_added_complete_candidates_vs_pil']}; "
    f"at legacy 9.9um: {summary['union_added_legacy_complete_candidates_vs_pil']}.",
    '',
    'Candidate inclusion in this audit is geometry-only; FOCUS masks are retained but not used as a hard gate. A complete direct triple is only a necessary candidate-layer condition, not the official patched division score.',
]
(out / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2, default=str))
print('Artifacts:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    cells = [
        cell(
            '# EXP021 FOCUS3D-first division candidate audit V1\n\n'
            '定位四个完整带标签训练视频中的 5 个官方兼容 division 事件，'
            '在事件窗口内分别比较 Pilkwang、FOCUS3D 和并集节点的端点召回与完整三元组候选覆盖。'
            '本 Notebook 不运行 ILP、不运行旧后处理、不生成 submission.csv，也不把候选覆盖称为正式 CV。',
            'markdown', 'exp021-title',
        ),
        cell(modules_cell() + '\n' + SETUP, 'code', 'exp021-setup'),
        cell(CENTERS, 'code', 'exp021-pilkwang'),
        cell(FOCUS, 'code', 'exp021-focus3d'),
        cell(AUDIT, 'code', 'exp021-audit'),
    ]
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
            'kaggle': {'title': 'EXP021 FOCUS3D-first division candidate audit V1'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
