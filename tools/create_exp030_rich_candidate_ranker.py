"""Create EXP030: rich FOCUS3D-evidence division candidate ranking."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .create_exp027_structural_selection import (
        FOCUS,
        PILKWANG,
        ROOT,
        SETUP,
        cell,
        embedded_modules,
    )
except ImportError:  # pragma: no cover
    from create_exp027_structural_selection import (  # type: ignore
        FOCUS,
        PILKWANG,
        ROOT,
        SETUP,
        cell,
        embedded_modules,
    )


OUT = ROOT / "EXP" / "EXP030" / "CELL_train_rich_focus3d_candidate_ranker.ipynb"
WIDE_SETUP = SETUP.replace("TARGET_VIDEO_COUNT = 12", "TARGET_VIDEO_COUNT = 24").replace(
    "MAX_EVENTS_PER_VIDEO = 3", "MAX_EVENTS_PER_VIDEO = 2"
)


CANDIDATE_AND_MASK_FEATURES = r'''from collections import defaultdict

from focus_first.candidates import enumerate_divisions, candidate_to_dict
from focus_first.global_selection import (
    DivisionSelectionCandidate,
    select_nonconflicting_divisions,
    selection_metrics,
)

PARENT_RADIUS_UM = 16.0
DAUGHTER_RADIUS_UM = 20.0
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

for stem in selected_stems:
    gt_nodes, _ = all_gt[stem]
    events = {event.event_index: event for event in events_by_video[stem]}
    for mode in MODES:
        nodes_by_t = build_nodes_for_mode(stem, mode)
        node_lookup = {node.proposal_id: node for nodes in nodes_by_t.values() for node in nodes}
        instance_lookup = {
            (int(instance.t), int(instance.label)): instance
            for instances in focus_instances_by_video[stem].values() for instance in instances
        } if mode == 'union' else {}
        matched = match_gt_to_nodes(gt_nodes, nodes_by_t)
        for manifest in [row for row in event_manifest if row['video'] == stem]:
            event = events[manifest['event_index']]
            event_id = f"{stem}:{event.event_index}"
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
                objects = enumerate_divisions(
                    [node_lookup[mapped_parent]],
                    nodes_by_t.get(int(event.parent_t) + 1, []),
                    instances_by_label=instance_lookup,
                    max_parent_distance_um=PARENT_RADIUS_UM,
                    max_daughter_distance_um=DAUGHTER_RADIUS_UM,
                    scale_um=tuple(SCALE_UM),
                )
            event_rows.append({
                'video': stem, 'event_id': event_id, 'mode': mode,
                'candidate_count': len(objects), 'positive_candidate_count': 0,
            })
            for candidate_index, candidate in enumerate(objects):
                label = int(
                    frozenset((candidate.daughter1_id, candidate.daughter2_id)) == mapped_daughters
                    and str(candidate.parent_id) == str(mapped_parent)
                )
                parent_node = node_lookup[candidate.parent_id]
                daughter1_node = node_lookup[candidate.daughter1_id]
                daughter2_node = node_lookup[candidate.daughter2_id]
                nodes = (parent_node, daughter1_node, daughter2_node)
                focus_only_count = sum(int(node.kind == 'focus_only') for node in nodes)
                focus_instance_count = sum(int(node.focus_key is not None) for node in nodes)
                consensus_count = sum(int(node.kind == 'consensus') for node in nodes)
                focus_quality_mean = float(np.mean([
                    node.focus_quality for node in nodes if node.focus_key is not None
                ])) if focus_instance_count else np.nan
                candidate_id = f"{mode}|{event_id}|{candidate_index}"
                row = {
                    'video': stem, 'event_id': event_id, 'mode': mode,
                    'candidate_index': int(candidate_index), 'candidate_id': candidate_id,
                    'label': label, **candidate_to_dict(candidate),
                    'focus_only_count': focus_only_count,
                    'focus_instance_count': focus_instance_count,
                    'consensus_count': consensus_count,
                    'focus_quality_mean': focus_quality_mean,
                }
                candidate_rows.append(row)
                selection_inputs[(stem, mode)].append(DivisionSelectionCandidate(
                    candidate_id=candidate_id,
                    parent_id=candidate.parent_id,
                    daughter1_id=candidate.daughter1_id,
                    daughter2_id=candidate.daughter2_id,
                    event_id=event_id,
                    geometry_cost=float(candidate.parent_distance_um + 0.5 * candidate.daughter_distance_um),
                    focus_only_count=focus_only_count,
                    label=label,
                ))
                event_rows[-1]['positive_candidate_count'] += label

candidate_df = pd.DataFrame(candidate_rows)
endpoint_df = pd.DataFrame(endpoint_rows).drop_duplicates()
event_df = pd.DataFrame(event_rows).drop_duplicates()
if candidate_df.empty:
    raise RuntimeError('No rich division candidates generated')

selection_rows = []
selected_rows = []
for mode in MODES:
    all_selected = []
    for stem in selected_stems:
        selected = select_nonconflicting_divisions(
            selection_inputs[(stem, mode)], focus_activation_cost=0.0,
        )
        all_selected.extend(selected)
        for item in selected:
            selected_rows.append({
                'video': stem, 'mode': mode, 'event_id': item.event_id,
                'candidate_id': item.candidate_id, 'label': item.label,
                'focus_only_count': item.focus_only_count,
                'geometry_cost': item.geometry_cost,
                'total_cost': item.total_cost(0.0),
            })
    all_candidates = [
        item for (stem, item_mode), rows in selection_inputs.items()
        if item_mode == mode for item in rows
    ]
    metrics = selection_metrics(all_candidates, all_selected)
    lookup = event_df[event_df["mode"] == mode]
    expected_events = int(lookup["event_id"].nunique())
    selection_rows.append({
        'mode': mode, 'events': expected_events,
        'candidate_count': int(metrics['candidate_count']),
        'positive_candidate_count': int(metrics['positive_candidate_count']),
        'selected_count': int(metrics['selected_count']),
        'selected_positive_count': int(metrics['selected_positive_count']),
        'selected_negative_count': int(metrics['selected_negative_count']),
        'selection_precision': float(metrics['selection_precision']),
        'event_hit_rate': float(metrics['selected_positive_count'] / expected_events)
        if expected_events else 0.0,
    })

selection_df = pd.DataFrame(selection_rows)
selected_df = pd.DataFrame(selected_rows)
print('rich endpoint summary')
display(endpoint_df.groupby('mode')[['parent_matched', 'daughter_count_matched']].mean())
print('rich geometry selection summary')
display(selection_df)
'''


RANKER = r'''from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from focus_first.candidate_scoring import (
    event_ranking_metrics,
    leave_one_group_out_splits,
    permute_columns_within_groups,
)

rank_frame = candidate_df[candidate_df["mode"] == "union"].copy().reset_index(drop=True)
BASE_FEATURES = [
    'parent_distance_um', 'daughter_distance_um', 'focus_only_count',
]
MASK_FEATURES = [
    'daughter_separation_score', 'volume_balance', 'volume_conservation',
    'parent_union_overlap', 'parent_union_parent_coverage',
    'parent_union_daughter_coverage', 'focus_instance_count',
    'consensus_count', 'focus_quality_mean',
]
RICH_FEATURES = BASE_FEATURES + MASK_FEATURES
if rank_frame.empty:
    raise RuntimeError('EXP030 union candidate pool is empty')
if int(rank_frame['label'].sum()) != int(rank_frame['event_id'].nunique()):
    raise RuntimeError('EXP030 expects exactly one positive candidate per union event')


def fixed_scores(frame, daughter_weight, focus_cost):
    return -(
        frame['parent_distance_um'].astype(float).to_numpy()
        + float(daughter_weight) * frame['daughter_distance_um'].astype(float).to_numpy()
        + float(focus_cost) * frame['focus_only_count'].astype(float).to_numpy()
    )


def prepare(train, valid, features):
    train_x = train[features].astype(float).replace([np.inf, -np.inf], np.nan).copy()
    valid_x = valid[features].astype(float).replace([np.inf, -np.inf], np.nan).copy()
    for feature in features:
        train_missing = train_x[feature].isna()
        valid_missing = valid_x[feature].isna()
        median = train_x.loc[~train_missing, feature].median()
        if not np.isfinite(median):
            median = 0.0
        train_x[f'{feature}__missing'] = train_missing.astype(float)
        valid_x[f'{feature}__missing'] = valid_missing.astype(float)
        train_x[feature] = train_x[feature].fillna(float(median))
        valid_x[feature] = valid_x[feature].fillna(float(median))
    return train_x, valid_x


def oof_scores(frame, features, model_kind='logistic'):
    scores = np.full(len(frame), np.nan, dtype=float)
    groups = frame['video'].astype(str).tolist()
    for fold, (train_idx, valid_idx) in enumerate(leave_one_group_out_splits(groups)):
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        if train['label'].nunique() != 2:
            raise RuntimeError(f'OOF fold {fold} has one class')
        train_x, valid_x = prepare(train, valid, features)
        if model_kind == 'forest':
            model = RandomForestClassifier(
                n_estimators=300, max_depth=7, min_samples_leaf=4,
                class_weight='balanced_subsample', random_state=20260917,
                n_jobs=-1,
            )
        else:
            model = Pipeline([
                ('scale', StandardScaler()),
                ('logistic', LogisticRegression(
                    C=0.5, class_weight='balanced', max_iter=3000,
                    random_state=20260917, solver='liblinear',
                )),
            ])
        model.fit(train_x, train['label'].astype(int).to_numpy())
        scores[valid_idx] = model.predict_proba(valid_x)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError('OOF scores contain NaN')
    return scores


def selection_items(frame, scores):
    items = []
    for row, score in zip(frame.itertuples(index=False), scores):
        items.append(DivisionSelectionCandidate(
            candidate_id=str(row.candidate_id), parent_id=str(row.parent_id),
            daughter1_id=str(row.daughter1_id), daughter2_id=str(row.daughter2_id),
            event_id=str(row.event_id), geometry_cost=float(-score),
            focus_only_count=0, label=int(row.label),
        ))
    return items


def structural_metrics(frame, scores):
    chosen_rows = []
    for video, group in frame.assign(_score=scores).groupby('video', sort=True):
        chosen = select_nonconflicting_divisions(
            selection_items(group, group['_score'].to_numpy(float)),
            focus_activation_cost=0.0,
        )
        for item in chosen:
            chosen_rows.append({
                'video': video, 'event_id': item.event_id,
                'candidate_id': item.candidate_id, 'label': int(item.label),
            })
    selected_count = len(chosen_rows)
    selected_positive = sum(int(row['label'] == 1) for row in chosen_rows)
    events = int(frame['event_id'].nunique())
    return {
        'selected_count': selected_count,
        'selected_positive_count': selected_positive,
        'selected_negative_count': selected_count - selected_positive,
        'selection_precision': float(selected_positive / selected_count) if selected_count else 0.0,
        'event_hit_rate': float(selected_positive / events) if events else 0.0,
    }, chosen_rows


permuted = pd.DataFrame(permute_columns_within_groups(
    rank_frame.to_dict('records'), columns=MASK_FEATURES,
    group_key='video', seed=20260917,
))
arms = {
    'geometry_1_05_focus0': ('fixed', fixed_scores(rank_frame, 0.5, 0.0), BASE_FEATURES),
    'logistic_3feat': ('fixed', oof_scores(rank_frame, BASE_FEATURES), BASE_FEATURES),
    'logistic_rich': ('fixed', oof_scores(rank_frame, RICH_FEATURES), RICH_FEATURES),
    'logistic_rich_mask_permuted': ('fixed', oof_scores(permuted, RICH_FEATURES), RICH_FEATURES),
    'forest_rich': ('fixed', oof_scores(rank_frame, RICH_FEATURES, model_kind='forest'), RICH_FEATURES),
}
arm_rows = []
score_rows = []
selected_rows = []
for arm, (_, scores, features) in arms.items():
    rank_metrics = event_ranking_metrics(rank_frame.event_id, rank_frame.label, scores)
    structure, chosen = structural_metrics(rank_frame, scores)
    arm_rows.append({'arm': arm, 'features': json.dumps(features), **rank_metrics, **structure})
    for index, score in enumerate(scores):
        row = rank_frame.iloc[index]
        score_rows.append({
            'arm': arm, 'video': row.video, 'event_id': row.event_id,
            'candidate_id': row.candidate_id, 'label': int(row.label),
            'score': float(score),
        })
    for row in chosen:
        row['arm'] = arm
        selected_rows.append(row)
arm_summary_df = pd.DataFrame(arm_rows)
rank_scores_df = pd.DataFrame(score_rows)
rank_selected_df = pd.DataFrame(selected_rows)
print('EXP030 rich ranking summary')
display(arm_summary_df)
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp030_rich_focus3d_candidate_ranker')
out.mkdir(parents=True, exist_ok=True)
candidate_df.to_csv(out / 'candidate_rows.csv', index=False)
endpoint_df.to_csv(out / 'endpoint_rows.csv', index=False)
event_df.to_csv(out / 'event_rows.csv', index=False)
selection_df.to_csv(out / 'geometry_selection_summary.csv', index=False)
selected_df.to_csv(out / 'geometry_selected_candidates.csv', index=False)
arm_summary_df.to_csv(out / 'arm_summary.csv', index=False)
rank_scores_df.to_csv(out / 'candidate_oof_scores.csv', index=False)
rank_selected_df.to_csv(out / 'ranker_selected_candidates.csv', index=False)
summary = {
    'experiment': 'EXP030', 'official_cv': None, 'submission_generated': False,
    'selected_videos': selected_stems, 'events': int(event_df.event_id.nunique()),
    'union_candidates': int(len(rank_frame)), 'rich_features': RICH_FEATURES,
    'mask_features': MASK_FEATURES, 'oof_group': 'video', 'arms': list(arms),
    'candidate_gate_um': {'parent': PARENT_RADIUS_UM, 'daughter': DAUGHTER_RADIUS_UM},
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = [
    '# EXP030 rich FOCUS3D candidate ranking', '',
    'Diagnostic only; no submission and no official CV.', '',
    f"Videos={len(selected_stems)}; events={summary['events']}; union candidates={summary['union_candidates']}.", '',
    '```text', arm_summary_df.to_string(index=False), '```', '',
    'The original Pilkwang and FOCUS3D weights are frozen. The rich arms use '
    'FOCUS3D mask-derived features; the permuted arm destroys their within-video association.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    notebook = {
        "cells": [
            cell(
                "# EXP030 rich FOCUS3D division candidate ranking\n\n"
                "Use frozen Pilkwang and FOCUS3D predictions, then compare rich mask-aware "
                "candidate rankers under leave-one-video-out evaluation.",
                "markdown", "title",
            ),
            cell(embedded_modules(), "code", "embedded-modules"),
            cell(WIDE_SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(CANDIDATE_AND_MASK_FEATURES, "code", "rich-candidate-pool"),
            cell(RANKER, "code", "rich-oof-rankers"),
            cell(OUTPUTS, "code", "write-diagnostics"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP030 rich FOCUS3D division candidate ranking"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
