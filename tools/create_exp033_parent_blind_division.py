"""Create EXP033: parent-blind division discovery with a no-division option."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp031_focus_mask_group_ablation as source
except ImportError:  # pragma: no cover
    import create_exp031_focus_mask_group_ablation as source  # type: ignore


ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP033" / "CELL_train_parent_blind_division.ipynb"
SETUP = source.WIDE_SETUP
EMBEDDED_MODULES = source.EMBEDDED_MODULES
PILKWANG = source.PILKWANG
FOCUS = source.FOCUS
cell = source.cell


PARENT_BLIND_CANDIDATES = r'''from collections import defaultdict

from focus_first.candidates import enumerate_divisions, candidate_to_dict

PARENT_RADIUS_UM = 16.0
DAUGHTER_RADIUS_UM = 20.0


def build_union_nodes(stem):
    result = {}
    for t in frames_by_video[stem]:
        result[int(t)] = build_unified_nodes(
            pilkwang_centers_by_video[stem].get(int(t), []),
            focus_instances_by_video[stem].get(int(t), []),
            scale_um=tuple(SCALE_UM), match_radius_um=MATCH_RADIUS_UM,
            accept_focus_quality=0.0,
        )
    return result


def match_gt_to_pool(gt_nodes, nodes_by_t, *, allow_focus_only):
    mapping = {}
    for t, predicted in nodes_by_t.items():
        if not allow_focus_only:
            predicted = [node for node in predicted if node.kind != 'focus_only']
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
truth_rows = []
transition_rows = []

for stem in selected_stems:
    gt_nodes, _ = all_gt[stem]
    nodes_by_t = build_union_nodes(stem)
    node_lookup = {node.proposal_id: node for nodes in nodes_by_t.values() for node in nodes}
    instance_lookup = {
        (int(instance.t), int(instance.label)): instance
        for instances in focus_instances_by_video[stem].values() for instance in instances
    }
    parent_mapping = match_gt_to_pool(gt_nodes, nodes_by_t, allow_focus_only=False)
    union_mapping = match_gt_to_pool(gt_nodes, nodes_by_t, allow_focus_only=True)
    transitions = sorted({int(event.parent_t) for event in events_by_video[stem]})

    for t in transitions:
        transition_id = f'{stem}:{t}'
        parent_nodes = [node for node in nodes_by_t.get(t, []) if node.kind != 'focus_only']
        daughter_nodes = nodes_by_t.get(t + 1, [])
        transition_events = [event for event in all_events[stem] if int(event.parent_t) == t]
        truth_by_key = defaultdict(list)
        local_truth = []
        for event in transition_events:
            event_id = f'{stem}:{event.event_index}'
            parent_id = parent_mapping.get(event.parent_id)
            daughter_ids = tuple(filter(None, (
                union_mapping.get(event.daughter1_id), union_mapping.get(event.daughter2_id),
            )))
            endpoint_complete = int(parent_id is not None and len(daughter_ids) == 2)
            key = None
            if endpoint_complete:
                key = (str(parent_id), frozenset(map(str, daughter_ids)))
                truth_by_key[key].append(event_id)
            row = {
                'video': stem, 'transition_id': transition_id, 'event_id': event_id,
                'parent_t': t, 'parent_endpoint_matched': int(parent_id is not None),
                'daughter_endpoint_count': len(daughter_ids),
                'endpoint_complete': endpoint_complete, 'candidate_found': 0,
                'positive_candidate_id': '',
            }
            truth_rows.append(row)
            local_truth.append(row)

        objects = enumerate_divisions(
            parent_nodes, daughter_nodes, instances_by_label=instance_lookup,
            max_parent_distance_um=PARENT_RADIUS_UM,
            max_daughter_distance_um=DAUGHTER_RADIUS_UM,
            scale_um=tuple(SCALE_UM),
        )
        positive_count = 0
        for candidate_index, candidate in enumerate(objects):
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
            key = (
                str(candidate.parent_id),
                frozenset((str(candidate.daughter1_id), str(candidate.daughter2_id))),
            )
            positive_events = truth_by_key.get(key, [])
            candidate_id = f'{transition_id}|{candidate_index}'
            row = {
                'video': stem, 'transition_id': transition_id,
                'candidate_id': candidate_id, 'parent_t': t,
                'label': int(bool(positive_events)),
                'positive_event_ids': ';'.join(positive_events),
                **candidate_to_dict(candidate),
                'focus_only_count': focus_only_count,
                'focus_instance_count': focus_instance_count,
                'consensus_count': consensus_count,
                'focus_quality_mean': focus_quality_mean,
            }
            candidate_rows.append(row)
            if positive_events:
                positive_count += len(positive_events)
                for truth in local_truth:
                    if truth['event_id'] in positive_events:
                        truth['candidate_found'] = 1
                        truth['positive_candidate_id'] = candidate_id

        transition_rows.append({
            'video': stem, 'transition_id': transition_id, 'parent_t': t,
            'parent_count': len(parent_nodes), 'daughter_count': len(daughter_nodes),
            'candidate_count': len(objects), 'gt_event_count': len(transition_events),
            'positive_candidate_count': positive_count,
        })
        print('parent-blind', transition_id, 'parents', len(parent_nodes),
              'daughters', len(daughter_nodes), 'candidates', len(objects),
              'positives', positive_count, flush=True)

candidate_df = pd.DataFrame(candidate_rows)
truth_df = pd.DataFrame(truth_rows).drop_duplicates('event_id').reset_index(drop=True)
transition_df = pd.DataFrame(transition_rows)
if candidate_df.empty:
    raise RuntimeError('EXP033 generated no parent-blind candidates')
if int(candidate_df['label'].sum()) != int(truth_df['candidate_found'].sum()):
    raise RuntimeError('EXP033 positive candidate/truth manifest mismatch')
print('EXP033 candidate pool', len(candidate_df), 'truth events', len(truth_df),
      'rankable', int(truth_df.candidate_found.sum()))
'''


PARENT_BLIND_RANKER = r'''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ['parent_distance_um', 'daughter_distance_um', 'focus_only_count']
MASK_GEOMETRY_FEATURES = [
    'volume_balance', 'volume_conservation',
    'parent_union_overlap', 'parent_union_parent_coverage',
    'parent_union_daughter_coverage',
]
RICH_FEATURES = BASE_FEATURES + MASK_GEOMETRY_FEATURES
HARD_NEGATIVES_PER_TRANSITION = 256
RANDOM_NEGATIVES_PER_TRANSITION = 256


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


def sampled_training_indices(frame, eligible_indices, seed):
    eligible = frame.loc[eligible_indices]
    positives = eligible.index[eligible['label'] == 1].tolist()
    negatives = eligible[eligible['label'] == 0].copy()
    negatives['_geometry_cost'] = (
        negatives['parent_distance_um'].astype(float)
        + 0.5 * negatives['daughter_distance_um'].astype(float)
    )
    rng = np.random.default_rng(seed)
    selected = set(positives)
    for _, group in negatives.groupby('transition_id', sort=True):
        hard = group.nsmallest(HARD_NEGATIVES_PER_TRANSITION, '_geometry_cost').index.to_numpy()
        selected.update(map(int, hard))
        remaining = group.index.difference(hard).to_numpy(dtype=int)
        if len(remaining):
            size = min(RANDOM_NEGATIVES_PER_TRANSITION, len(remaining))
            selected.update(map(int, rng.choice(remaining, size=size, replace=False)))
    return np.asarray(sorted(selected), dtype=int)


def oof_scores(frame, features):
    scores = np.full(len(frame), np.nan, dtype=float)
    videos = frame['video'].astype(str).to_numpy()
    for fold, video in enumerate(sorted(set(videos))):
        valid_idx = np.flatnonzero(videos == video)
        train_pool = np.flatnonzero(videos != video)
        train_idx = sampled_training_indices(frame, train_pool, seed=20260919 + fold)
        train = frame.loc[train_idx]
        valid = frame.loc[valid_idx]
        if train['label'].nunique() != 2:
            raise RuntimeError(f'EXP033 fold {video} has one training class')
        train_x, valid_x = prepare(train, valid, features)
        model = Pipeline([
            ('scale', StandardScaler()),
            ('logistic', LogisticRegression(
                C=0.5, class_weight='balanced', max_iter=3000,
                random_state=20260919, solver='liblinear',
            )),
        ])
        model.fit(train_x, train['label'].astype(int).to_numpy())
        scores[valid_idx] = model.predict_proba(valid_x)[:, 1]
        print('OOF', fold + 1, '/', len(set(videos)), video,
              'train', len(train_idx), 'valid', len(valid_idx), flush=True)
    if not np.isfinite(scores).all():
        raise RuntimeError('EXP033 OOF scores contain NaN')
    return scores


def event_ranks(frame, truth, scores, arm):
    scored = frame[['transition_id', 'candidate_id']].copy()
    scored['_score'] = scores
    score_lookup = dict(zip(scored['candidate_id'], scored['_score']))
    rows = []
    grouped_scores = {
        transition_id: group['_score'].to_numpy(float)
        for transition_id, group in scored.groupby('transition_id', sort=False)
    }
    for row in truth.itertuples(index=False):
        rank = np.nan
        if int(row.candidate_found):
            positive_score = float(score_lookup[str(row.positive_candidate_id)])
            values = grouped_scores[str(row.transition_id)]
            rank = 1.0 + float(np.sum(values > positive_score))
            rank += 0.5 * float(np.sum(np.isclose(values, positive_score, rtol=0.0, atol=1e-12)) - 1)
        rows.append({
            'arm': arm, 'video': row.video, 'transition_id': row.transition_id,
            'event_id': row.event_id, 'candidate_found': int(row.candidate_found),
            'positive_rank': rank,
        })
    return pd.DataFrame(rows)


def summarize_ranks(ranks):
    total = len(ranks)
    rankable = ranks['positive_rank'].notna()
    values = ranks.loc[rankable, 'positive_rank'].to_numpy(float)
    result = {'events': total, 'rankable_events': int(rankable.sum())}
    for k in [1, 3, 5, 10, 20, 50, 100]:
        count = int(np.sum(values <= k))
        result[f'top{k}_count'] = count
        result[f'top{k}_recall_all'] = float(count / total) if total else 0.0
        result[f'top{k}_recall_rankable'] = float(count / len(values)) if len(values) else 0.0
    result['mean_reciprocal_rank_all'] = float(np.sum(1.0 / values) / total) if total else 0.0
    result['median_rank_rankable'] = float(np.median(values)) if len(values) else np.nan
    return result


def threshold_scan(frame, truth, scores, arm):
    scored = frame.copy()
    scored['_score'] = scores
    parent_best = (
        scored.sort_values('_score', ascending=False, kind='mergesort')
        .drop_duplicates(['transition_id', 'parent_id'])
    )
    quantiles = np.linspace(0.0, 1.0, 201)
    thresholds = np.unique(np.quantile(parent_best['_score'].to_numpy(float), quantiles))
    thresholds = np.r_[thresholds, np.nextafter(parent_best['_score'].max(), np.inf)]
    total_truth = len(truth)
    rows = []
    for threshold in thresholds:
        proposed = parent_best[parent_best['_score'] >= float(threshold)].sort_values(
            '_score', ascending=False, kind='mergesort'
        )
        selected = []
        used_daughters = defaultdict(set)
        for candidate in proposed.itertuples(index=False):
            daughters = {str(candidate.daughter1_id), str(candidate.daughter2_id)}
            if daughters & used_daughters[str(candidate.transition_id)]:
                continue
            used_daughters[str(candidate.transition_id)].update(daughters)
            selected.append(candidate)
        tp = sum(int(candidate.label == 1) for candidate in selected)
        fp = len(selected) - tp
        fn = total_truth - tp
        denom = tp + fp + fn
        rows.append({
            'arm': arm, 'threshold': float(threshold), 'selected_count': len(selected),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': float(tp / len(selected)) if selected else 0.0,
            'recall': float(tp / total_truth) if total_truth else 0.0,
            'division_jaccard_surrogate': float(tp / denom) if denom else 0.0,
        })
    return pd.DataFrame(rows)


score_arms = {
    'geometry': -(
        candidate_df['parent_distance_um'].to_numpy(float)
        + 0.5 * candidate_df['daughter_distance_um'].to_numpy(float)
    ),
    'logistic_base': oof_scores(candidate_df, BASE_FEATURES),
    'logistic_mask_geometry': oof_scores(candidate_df, RICH_FEATURES),
}
rank_frames = []
scan_frames = []
arm_rows = []
for arm, scores in score_arms.items():
    ranks = event_ranks(candidate_df, truth_df, scores, arm)
    rank_frames.append(ranks)
    arm_rows.append({'arm': arm, **summarize_ranks(ranks)})
    scan_frames.append(threshold_scan(candidate_df, truth_df, scores, arm))

event_ranks_df = pd.concat(rank_frames, ignore_index=True)
arm_summary_df = pd.DataFrame(arm_rows)
threshold_scan_df = pd.concat(scan_frames, ignore_index=True)
best_threshold_df = (
    threshold_scan_df.sort_values(
        ['arm', 'division_jaccard_surrogate', 'precision', 'threshold'],
        ascending=[True, False, False, False], kind='mergesort',
    ).groupby('arm', as_index=False).head(1).reset_index(drop=True)
)
parent_best_rows = []
for arm, scores in score_arms.items():
    scored = candidate_df.assign(score=scores)
    top = scored.sort_values('score', ascending=False, kind='mergesort').drop_duplicates(
        ['transition_id', 'parent_id']
    )
    for row in top.itertuples(index=False):
        parent_best_rows.append({
            'arm': arm, 'video': row.video, 'transition_id': row.transition_id,
            'parent_id': row.parent_id, 'candidate_id': row.candidate_id,
            'label': int(row.label), 'score': float(row.score),
        })
parent_best_df = pd.DataFrame(parent_best_rows)
print('EXP033 parent-blind rank summary')
display(arm_summary_df)
print('EXP033 exploratory no-division thresholds')
display(best_threshold_df)
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp033_parent_blind_division')
out.mkdir(parents=True, exist_ok=True)
truth_df.to_csv(out / 'truth_events.csv', index=False)
transition_df.to_csv(out / 'transition_summary.csv', index=False)
arm_summary_df.to_csv(out / 'arm_summary.csv', index=False)
event_ranks_df.to_csv(out / 'event_ranks.csv', index=False)
threshold_scan_df.to_csv(out / 'no_division_threshold_scan.csv', index=False)
best_threshold_df.to_csv(out / 'best_thresholds.csv', index=False)
parent_best_df.to_csv(out / 'parent_best_candidates.csv', index=False)
summary = {
    'experiment': 'EXP033', 'official_cv': None, 'submission_generated': False,
    'parent_oracle_used': False, 'no_division_option': 'score_threshold',
    'selected_videos': selected_stems, 'transitions': int(transition_df.transition_id.nunique()),
    'truth_events': int(len(truth_df)), 'endpoint_complete_events': int(truth_df.endpoint_complete.sum()),
    'candidate_recalled_events': int(truth_df.candidate_found.sum()),
    'candidate_count': int(len(candidate_df)),
    'parent_pool': 'pilkwang_or_consensus', 'daughter_pool': 'union',
    'candidate_gate_um': {'parent': PARENT_RADIUS_UM, 'daughter': DAUGHTER_RADIUS_UM},
    'features': {'base': BASE_FEATURES, 'mask_geometry': MASK_GEOMETRY_FEATURES},
    'threshold_scope': 'exploratory_global_oof_not_nested',
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = [
    '# EXP033 parent-blind division discovery', '',
    'Diagnostic only; no submission and no official CV.', '',
    f"Videos={len(selected_stems)}; transitions={summary['transitions']}; truth events={summary['truth_events']}; candidates={summary['candidate_count']}.", '',
    '```text', arm_summary_df.to_string(index=False), '```', '',
    'The correct parent is not supplied to candidate generation. All Pilkwang/consensus '
    'parents in each selected transition compete. A score threshold represents the no-division '
    'option. Threshold scans are exploratory OOF diagnostics and are not deployment thresholds.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    notebook = {
        "cells": [
            cell(
                "# EXP033 parent-blind division discovery\n\n"
                "Enumerate candidate triples from every Pilkwang/consensus parent in selected "
                "transitions, rank them with and without FOCUS3D mask geometry, and allow a "
                "score-threshold no-division decision.",
                "markdown", "title",
            ),
            cell(EMBEDDED_MODULES(), "code", "embedded-modules"),
            cell(SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(PARENT_BLIND_CANDIDATES, "code", "parent-blind-candidates"),
            cell(PARENT_BLIND_RANKER, "code", "parent-blind-ranker"),
            cell(OUTPUTS, "code", "write-diagnostics"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP033 parent-blind division discovery"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
