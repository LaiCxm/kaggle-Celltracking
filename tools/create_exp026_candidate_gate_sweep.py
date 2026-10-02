"""Create the EXP026 recall-first division candidate gate sweep notebook."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .create_exp025_candidate_evidence import (
        ROOT,
        FOCUS,
        PILKWANG,
        SETUP,
        cell,
        embedded_modules,
    )
except ImportError:  # Direct execution via ``python tools/...py``.
    from create_exp025_candidate_evidence import (  # type: ignore
        ROOT,
        FOCUS,
        PILKWANG,
        SETUP,
        cell,
        embedded_modules,
    )


OUT = ROOT / "EXP" / "EXP026" / "CELL_audit_division_candidate_gate_sweep.ipynb"


CANDIDATE_SWEEP = r'''from focus_first.gate_sweep import (
    candidate_to_dict as geometry_candidate_to_dict,
    enumerate_geometry_divisions,
    select_minimum_budget_full_recall,
    summarize_gate_grid,
)

GATE_LEVELS_UM = [14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 32.0]
MAX_GATE_UM = max(GATE_LEVELS_UM)


def build_union_nodes(stem):
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
for stem in selected_stems:
    gt_nodes, _ = all_gt[stem]
    nodes_by_t = build_union_nodes(stem)
    node_lookup = {node.proposal_id: node for nodes in nodes_by_t.values() for node in nodes}
    matched = match_gt_to_nodes(gt_nodes, nodes_by_t)
    events = {event.event_index: event for event in events_by_video[stem]}
    for manifest in [row for row in event_manifest if row['video'] == stem]:
        event = events[manifest['event_index']]
        event_id = f"{stem}:{event.event_index}"
        mapped_parent = matched.get(event.parent_id)
        mapped_daughter1 = matched.get(event.daughter1_id)
        mapped_daughter2 = matched.get(event.daughter2_id)
        mapped_daughters = frozenset(filter(None, (mapped_daughter1, mapped_daughter2)))
        objects = []
        if mapped_parent is not None and len(mapped_daughters) == 2:
            objects = enumerate_geometry_divisions(
                [node_lookup[mapped_parent]],
                nodes_by_t.get(int(event.parent_t) + 1, []),
                max_parent_distance_um=MAX_GATE_UM,
                max_daughter_distance_um=MAX_GATE_UM,
                scale_um=tuple(SCALE_UM),
            )
        positive_rows = []
        for candidate_index, candidate in enumerate(objects):
            is_positive = int(
                str(candidate.parent_id) == str(mapped_parent)
                and frozenset((candidate.daughter1_id, candidate.daughter2_id)) == mapped_daughters
            )
            row = {
                'video': stem,
                'event_id': event_id,
                'event_index': int(event.event_index),
                'candidate_index': int(candidate_index),
                'label': is_positive,
                **geometry_candidate_to_dict(candidate),
            }
            candidate_rows.append(row)
            if is_positive:
                positive_rows.append(row)
        if len(positive_rows) > 1:
            raise RuntimeError(f'Multiple positive triples generated for {event_id}')
        positive = positive_rows[0] if positive_rows else None
        event_rows.append({
            **manifest,
            'event_id': event_id,
            'mapped_parent_id': mapped_parent,
            'mapped_daughter1_id': mapped_daughter1,
            'mapped_daughter2_id': mapped_daughter2,
            'parent_matched': int(mapped_parent is not None),
            'daughter_count_matched': len(mapped_daughters),
            'max_gate_candidate_count': len(objects),
            'positive_in_max_gate': int(positive is not None),
            'required_parent_radius_um': None if positive is None else positive['parent_distance_um'],
            'required_daughter_radius_um': None if positive is None else positive['daughter_distance_um'],
            'positive_contains_focus_only': None if positive is None else positive['contains_focus_only'],
        })

candidates_df = pd.DataFrame(candidate_rows)
events_df = pd.DataFrame(event_rows)
if candidates_df.empty:
    raise RuntimeError('No candidate superset generated')
if not ((events_df.parent_matched == 1) & (events_df.daughter_count_matched == 2)).all():
    print('WARNING: Some event endpoints were not matched; see event_rows.csv')

grid_rows = summarize_gate_grid(
    candidates_df.to_dict('records'),
    event_ids=events_df.event_id.astype(str).tolist(),
    parent_radii_um=GATE_LEVELS_UM,
    daughter_radii_um=GATE_LEVELS_UM,
)
grid_df = pd.DataFrame(grid_rows)
baseline = grid_df[
    (grid_df.parent_radius_um == 14.0) & (grid_df.daughter_radius_um == 14.0)
].iloc[0].to_dict()
grid_df['candidate_multiplier_vs_14'] = grid_df.candidate_count / max(int(baseline['candidate_count']), 1)
selected_gate = select_minimum_budget_full_recall(grid_df.to_dict('records'))

events_df['baseline_14_recalled'] = (
    (events_df.required_parent_radius_um <= 14.0)
    & (events_df.required_daughter_radius_um <= 14.0)
).astype(np.int64)

print('events:', len(events_df))
print('max-gate positives:', int(events_df.positive_in_max_gate.sum()))
print('14/14 baseline:', json.dumps(baseline, indent=2))
print('selected full-recall gate:', json.dumps(selected_gate, indent=2))
display(events_df[[
    'video', 'event_id', 'required_parent_radius_um', 'required_daughter_radius_um',
    'baseline_14_recalled', 'positive_contains_focus_only', 'max_gate_candidate_count',
]])
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp026_candidate_gate_sweep')
out.mkdir(parents=True, exist_ok=True)
candidates_df.to_csv(out / 'candidate_superset.csv', index=False)
events_df.to_csv(out / 'event_gate_requirements.csv', index=False)
grid_df.to_csv(out / 'gate_grid.csv', index=False)

frontier_rows = []
best_recall = -1
for row in grid_df.sort_values(
    ['candidate_count', 'parent_radius_um', 'daughter_radius_um'], kind='stable'
).to_dict('records'):
    if int(row['recalled_events']) > best_recall:
        frontier_rows.append(row)
        best_recall = int(row['recalled_events'])
frontier_df = pd.DataFrame(frontier_rows)
frontier_df.to_csv(out / 'recall_budget_frontier.csv', index=False)

summary = {
    'experiment': 'EXP026',
    'official_cv': None,
    'submission_generated': False,
    'selected_videos': selected_stems,
    'events': int(len(events_df)),
    'all_endpoints_matched_events': int(
        ((events_df.parent_matched == 1) & (events_df.daughter_count_matched == 2)).sum()
    ),
    'max_gate_positive_events': int(events_df.positive_in_max_gate.sum()),
    'gate_levels_um': GATE_LEVELS_UM,
    'baseline_14': baseline,
    'selected_full_recall_gate': selected_gate,
    'maximum_gate_candidate_count': int(len(candidates_df)),
    'runtime_seconds': {'pilkwang': pilkwang_seconds, 'focus3d': focus_seconds},
    'selection_rule': [
        'full event recall', 'minimum candidate count', 'minimum maximum radius',
        'minimum radius sum', 'minimum parent radius', 'minimum daughter radius',
    ],
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

diagonal = grid_df[grid_df.parent_radius_um == grid_df.daughter_radius_um]
report = [
    '# EXP026 division candidate geometry gate sweep', '',
    'Candidate-recall diagnostic only; not official CV and not a submission.', '',
    f"Events={len(events_df)}, all endpoints matched={summary['all_endpoints_matched_events']}, "
    f"positive triple present at max gate={summary['max_gate_positive_events']}.", '',
    '## Diagonal gate sweep', '', '```text', diagonal.to_string(index=False), '```', '',
    '## Recall-budget frontier', '', '```text', frontier_df.to_string(index=False), '```', '',
    '## Pre-registered selected full-recall gate', '', '```json',
    json.dumps(selected_gate, indent=2), '```', '',
    'The selected gate is a development input for the next global solver, not a validated CV optimum.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
display(diagonal)
display(frontier_df)
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    notebook = {
        "cells": [
            cell(
                "# EXP026 division candidate geometry gate sweep\n\n"
                "The same preselected event panel as EXP025 is used. A 32 um candidate superset is "
                "generated once, then replayed through a preregistered 7x7 gate grid.",
                "markdown",
                "title",
            ),
            cell(embedded_modules(), "code", "embedded-modules"),
            cell(SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(CANDIDATE_SWEEP, "code", "candidate-gate-sweep"),
            cell(OUTPUTS, "code", "write-diagnostics"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP026 division candidate gate sweep"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
