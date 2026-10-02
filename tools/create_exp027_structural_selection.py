"""Create the EXP027 FOCUS3D structural selection diagnostic notebook."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .create_exp025_candidate_evidence import FOCUS, PILKWANG, SETUP, cell, embedded_modules, ROOT
except ImportError:  # Direct execution via ``python tools/...py``.
    from create_exp025_candidate_evidence import FOCUS, PILKWANG, SETUP, cell, embedded_modules, ROOT  # type: ignore


OUT = ROOT / "EXP" / "EXP027" / "CELL_diagnose_focus3d_structural_selection.ipynb"


CANDIDATE_AND_SELECTION = r'''from collections import defaultdict

from focus_first.gate_sweep import enumerate_geometry_divisions
from focus_first.global_selection import (
    DivisionSelectionCandidate,
    select_nonconflicting_divisions,
    selection_metrics,
)

PARENT_RADIUS_UM = 16.0
DAUGHTER_RADIUS_UM = 20.0
ACTIVATION_COSTS = [0.0, 2.0, 4.0, 8.0, 12.0]
MODES = ('pil_only', 'union')


def build_nodes_for_mode(stem, mode):
    nodes_by_t = {}
    for t in frames_by_video[stem]:
        focus_instances = () if mode == 'pil_only' else focus_instances_by_video[stem].get(int(t), [])
        nodes_by_t[int(t)] = build_unified_nodes(
            pilkwang_centers_by_video[stem].get(int(t), []),
            focus_instances,
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
endpoint_rows = []
event_rows = []
selection_inputs = defaultdict(list)
events_by_video_id = defaultdict(list)

for stem in selected_stems:
    gt_nodes, _ = all_gt[stem]
    events = {event.event_index: event for event in events_by_video[stem]}
    for mode in MODES:
        nodes_by_t = build_nodes_for_mode(stem, mode)
        node_lookup = {node.proposal_id: node for nodes in nodes_by_t.values() for node in nodes}
        matched = match_gt_to_nodes(gt_nodes, nodes_by_t)
        for manifest in [row for row in event_manifest if row['video'] == stem]:
            event = events[manifest['event_index']]
            event_id = f"{stem}:{event.event_index}"
            events_by_video_id[stem].append(event_id)
            mapped_parent = matched.get(event.parent_id)
            mapped_daughters = frozenset(filter(None, (
                matched.get(event.daughter1_id), matched.get(event.daughter2_id),
            )))
            endpoint_rows.append({
                'video': stem, 'event_id': event_id, 'mode': mode,
                'parent_matched': int(mapped_parent is not None),
                'daughter_count_matched': len(mapped_daughters),
            })
            objects = []
            if mapped_parent is not None and len(mapped_daughters) == 2:
                objects = enumerate_geometry_divisions(
                    [node_lookup[mapped_parent]],
                    nodes_by_t.get(int(event.parent_t) + 1, []),
                    max_parent_distance_um=PARENT_RADIUS_UM,
                    max_daughter_distance_um=DAUGHTER_RADIUS_UM,
                    scale_um=tuple(SCALE_UM),
                )
            event_rows.append({
                'video': stem, 'event_id': event_id, 'mode': mode,
                'candidate_count': len(objects),
                'positive_candidate_count': 0,
            })
            for candidate_index, candidate in enumerate(objects):
                label = int(
                    frozenset((candidate.daughter1_id, candidate.daughter2_id)) == mapped_daughters
                    and str(candidate.parent_id) == str(mapped_parent)
                )
                parent_node = node_lookup[candidate.parent_id]
                daughter1_node = node_lookup[candidate.daughter1_id]
                daughter2_node = node_lookup[candidate.daughter2_id]
                focus_only_count = sum(
                    int(node.kind == 'focus_only')
                    for node in (parent_node, daughter1_node, daughter2_node)
                )
                candidate_id = f"{mode}|{event_id}|{candidate_index}"
                geometry_cost = float(
                    candidate.parent_distance_um + 0.5 * candidate.daughter_distance_um
                )
                row = {
                    'video': stem, 'event_id': event_id, 'mode': mode,
                    'candidate_index': int(candidate_index), 'candidate_id': candidate_id,
                    'label': label, **candidate.__dict__,
                    'focus_only_count': focus_only_count,
                    'geometry_cost': geometry_cost,
                }
                candidate_rows.append(row)
                selection_inputs[(stem, mode)].append(DivisionSelectionCandidate(
                    candidate_id=candidate_id,
                    parent_id=candidate.parent_id,
                    daughter1_id=candidate.daughter1_id,
                    daughter2_id=candidate.daughter2_id,
                    event_id=event_id,
                    geometry_cost=geometry_cost,
                    focus_only_count=focus_only_count,
                    label=label,
                ))
                event_rows[-1]['positive_candidate_count'] += label

candidate_df = pd.DataFrame(candidate_rows)
endpoint_df = pd.DataFrame(endpoint_rows).drop_duplicates()
event_df = pd.DataFrame(event_rows).drop_duplicates()
if candidate_df.empty:
    raise RuntimeError('No structural selection candidates generated')

selection_rows = []
selected_rows = []
for mode in MODES:
    costs = [0.0] if mode == 'pil_only' else ACTIVATION_COSTS
    for activation_cost in costs:
        all_selected = []
        for stem in selected_stems:
            source_rows = selection_inputs[(stem, mode)]
            selected = select_nonconflicting_divisions(
                source_rows, focus_activation_cost=activation_cost,
            )
            all_selected.extend(selected)
            for item in selected:
                selected_rows.append({
                    'video': stem, 'mode': mode,
                    'focus_activation_cost': activation_cost,
                    'event_id': item.event_id, 'candidate_id': item.candidate_id,
                    'label': item.label, 'focus_only_count': item.focus_only_count,
                    'geometry_cost': item.geometry_cost,
                    'total_cost': item.total_cost(activation_cost),
                })
        all_candidates = [item for (stem, item_mode), rows in selection_inputs.items()
                          if item_mode == mode for item in rows]
        metrics = selection_metrics(all_candidates, all_selected)
        # ``DataFrame.mode`` is a method; use explicit column access here.
        event_lookup = event_df[event_df["mode"] == mode]
        expected_events = int(event_lookup.event_id.nunique())
        selected_positive = int(metrics['selected_positive_count'])
        selected_negative = int(metrics['selected_negative_count'])
        selection_rows.append({
            'mode': mode,
            'focus_activation_cost': activation_cost,
            'events': expected_events,
            'candidate_count': int(metrics['candidate_count']),
            'positive_candidate_count': int(metrics['positive_candidate_count']),
            'selected_count': int(metrics['selected_count']),
            'selected_positive_count': selected_positive,
            'selected_negative_count': selected_negative,
            'missed_events': expected_events - int(metrics['selected_count']),
            'selection_precision': float(metrics['selection_precision']),
            'event_hit_rate': float(selected_positive / expected_events) if expected_events else 0.0,
        })

selection_df = pd.DataFrame(selection_rows)
selected_df = pd.DataFrame(selected_rows)
print('endpoint summary')
display(endpoint_df.groupby('mode')[['parent_matched', 'daughter_count_matched']].mean())
print('structural selection summary')
display(selection_df)
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp027_structural_selection')
out.mkdir(parents=True, exist_ok=True)
candidate_df.to_csv(out / 'candidate_rows.csv', index=False)
endpoint_df.to_csv(out / 'endpoint_rows.csv', index=False)
event_df.to_csv(out / 'event_rows.csv', index=False)
selection_df.to_csv(out / 'selection_summary.csv', index=False)
selected_df.to_csv(out / 'selected_candidates.csv', index=False)

summary = {
    'experiment': 'EXP027',
    'official_cv': None,
    'submission_generated': False,
    'selected_videos': selected_stems,
    'events': int(event_df.event_id.nunique()),
    'candidate_gate_um': {'parent': PARENT_RADIUS_UM, 'daughter': DAUGHTER_RADIUS_UM},
    'modes': list(MODES),
    'activation_costs_um': ACTIVATION_COSTS,
    'selection_constraints': ['unique_parent', 'unique_daughter', 'one_candidate_per_event'],
    'selection_algorithm': 'deterministic_greedy_set_packing',
    'runtime_seconds': {'pilkwang': pilkwang_seconds, 'focus3d': focus_seconds},
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

report = [
    '# EXP027 FOCUS3D structural selection diagnostic', '',
    'Diagnostic only; not official CV and not a submission.', '',
    f"Events={summary['events']}; candidate gate={PARENT_RADIUS_UM}/{DAUGHTER_RADIUS_UM} um.", '',
    '```text', selection_df.to_string(index=False), '```', '',
    'Selection uses geometry plus FOCUS-only activation cost and never reads labels. '
    'This greedy selector is a structural diagnostic, not the production ILP.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    notebook = {
        "cells": [
            cell(
                "# EXP027 FOCUS3D structural selection diagnostic\n\n"
                "Compare Pilkwang-only and union candidates under deterministic structural constraints "
                "and preregistered FOCUS-only activation costs.",
                "markdown",
                "title",
            ),
            cell(embedded_modules(), "code", "embedded-modules"),
            cell(SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(CANDIDATE_AND_SELECTION, "code", "candidate-and-selection"),
            cell(OUTPUTS, "code", "write-diagnostics"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP027 FOCUS3D structural selection diagnostic"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
