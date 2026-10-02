"""Create EXP037: soft parent aggregate evidence in the full candidate ranker."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp036_parent_aggregate_evidence as source
except ImportError:  # pragma: no cover
    import create_exp036_parent_aggregate_evidence as source  # type: ignore


ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP037" / "CELL_train_soft_parent_evidence.ipynb"
SETUP = source.SETUP
EMBEDDED_MODULES = source.EMBEDDED_MODULES
PILKWANG = source.PILKWANG
FOCUS = source.FOCUS
PARENT_CANDIDATES = source.PARENT_CANDIDATES
cell = source.cell


SOFT_RANKER = r'''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ['parent_distance_um', 'daughter_distance_um', 'focus_only_count']
PARENT_FEATURES = [
    'parent_center_score', 'parent_is_consensus', 'parent_focus_quality',
    'parent_backward_distance_um', 'parent_forward_count',
]
MASK_GEOMETRY_FEATURES = [
    'volume_balance', 'volume_conservation',
    'parent_union_overlap', 'parent_union_parent_coverage',
    'parent_union_daughter_coverage',
]
PARENT_AGGREGATE_FEATURES = [
    'candidate_count', 'best_geometry_cost', 'second_geometry_cost',
    'geometry_gap', 'best_volume_balance', 'best_volume_conservation',
    'best_parent_union_overlap', 'best_parent_union_parent_coverage',
    'best_parent_union_daughter_coverage', 'best_focus_only_count',
]
SOFT_PARENT_FEATURES = BASE_FEATURES + PARENT_FEATURES + PARENT_AGGREGATE_FEATURES
SOFT_PARENT_MASK_FEATURES = SOFT_PARENT_FEATURES + MASK_GEOMETRY_FEATURES
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


def model_for(train, valid, features):
    train_x, valid_x = prepare(train, valid, features)
    model = Pipeline([
        ('scale', StandardScaler()),
        ('logistic', LogisticRegression(
            C=0.5, class_weight='balanced', max_iter=3000,
            random_state=20260920, solver='liblinear',
        )),
    ])
    model.fit(train_x, train['label'].astype(int).to_numpy())
    return model.predict_proba(valid_x)[:, 1]


def sampled_candidate_indices(frame, eligible_indices, seed):
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


def parent_frame(frame):
    rows = []
    for (video, transition_id, parent_id), group in frame.groupby(
        ['video', 'transition_id', 'parent_id'], sort=False
    ):
        costs = (
            group['parent_distance_um'].astype(float)
            + 0.5 * group['daughter_distance_um'].astype(float)
        ).to_numpy()
        costs.sort()
        second = float(costs[1]) if len(costs) > 1 else float(costs[0])
        rows.append({
            'video': video, 'transition_id': transition_id, 'parent_id': parent_id,
            'candidate_count': int(len(group)),
            'best_geometry_cost': float(costs[0]),
            'second_geometry_cost': second,
            'geometry_gap': float(second - costs[0]),
            'best_volume_balance': float(group['volume_balance'].max()),
            'best_volume_conservation': float(group['volume_conservation'].max()),
            'best_parent_union_overlap': float(group['parent_union_overlap'].max()),
            'best_parent_union_parent_coverage': float(group['parent_union_parent_coverage'].max()),
            'best_parent_union_daughter_coverage': float(group['parent_union_daughter_coverage'].max()),
            'best_focus_only_count': float(group['focus_only_count'].min()),
        })
    return pd.DataFrame(rows)


def add_parent_aggregates(frame, aggregates):
    keys = ['video', 'transition_id', 'parent_id']
    return frame.merge(aggregates[keys + PARENT_AGGREGATE_FEATURES], on=keys, how='left')


def oof_scores(frame):
    scores = {
        'candidate_base': np.full(len(frame), np.nan, dtype=float),
        'soft_parent': np.full(len(frame), np.nan, dtype=float),
        'soft_parent_mask_geometry': np.full(len(frame), np.nan, dtype=float),
    }
    videos = frame['video'].astype(str).to_numpy()
    for fold, video in enumerate(sorted(set(videos))):
        valid_idx = np.flatnonzero(videos == video)
        train_pool = np.flatnonzero(videos != video)
        train_candidates = frame.loc[sampled_candidate_indices(frame, train_pool, 20260920 + fold)]
        valid_candidates = frame.loc[valid_idx]
        if train_candidates['label'].nunique() != 2:
            raise RuntimeError(f'EXP037 fold {video} has one class')
        train_aggregates = parent_frame(frame.loc[train_pool])
        valid_aggregates = parent_frame(valid_candidates)
        train_aug = add_parent_aggregates(train_candidates, train_aggregates)
        valid_aug = add_parent_aggregates(valid_candidates, valid_aggregates)
        scores['candidate_base'][valid_idx] = model_for(
            train_candidates, valid_candidates, BASE_FEATURES
        )
        scores['soft_parent'][valid_idx] = model_for(
            train_aug, valid_aug, SOFT_PARENT_FEATURES
        )
        scores['soft_parent_mask_geometry'][valid_idx] = model_for(
            train_aug, valid_aug, SOFT_PARENT_MASK_FEATURES
        )
        print('OOF', fold + 1, '/', len(set(videos)), video,
              'train_candidates', len(train_candidates),
              'valid_candidates', len(valid_candidates), flush=True)
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError('EXP037 OOF scores contain NaN')
    return scores


def event_ranks(frame, truth, scores, arm):
    scored = frame[['transition_id', 'candidate_id']].copy()
    scored['score'] = scores
    lookup = dict(zip(scored['candidate_id'], scored['score']))
    grouped = {
        str(transition_id): group['score'].to_numpy(float)
        for transition_id, group in scored.groupby('transition_id', sort=False)
    }
    rows = []
    for row in truth.itertuples(index=False):
        positive_score = float(lookup[str(row.positive_candidate_id)])
        values = grouped[str(row.transition_id)]
        rank = 1.0 + float(np.sum(values > positive_score))
        rank += 0.5 * float(np.sum(np.isclose(values, positive_score, rtol=0.0, atol=1e-12)) - 1)
        rows.append({
            'arm': arm, 'video': row.video, 'transition_id': row.transition_id,
            'event_id': row.event_id, 'positive_rank': rank,
        })
    return pd.DataFrame(rows)


def summarize(ranks):
    values = ranks['positive_rank'].to_numpy(float)
    return {
        'events': len(values),
        'top1_count': int(np.sum(values <= 1)),
        'top3_count': int(np.sum(values <= 3)),
        'top5_count': int(np.sum(values <= 5)),
        'top10_count': int(np.sum(values <= 10)),
        'top1_recall': float(np.mean(values <= 1)),
        'top3_recall': float(np.mean(values <= 3)),
        'top5_recall': float(np.mean(values <= 5)),
        'median_rank': float(np.median(values)),
        'mean_reciprocal_rank': float(np.mean(1.0 / values)),
    }


rank_frame = candidate_df.copy().reset_index(drop=True)
if rank_frame.empty or int(rank_frame['label'].sum()) != int(truth_df['candidate_found'].sum()):
    raise RuntimeError('EXP037 candidate/truth manifest mismatch')
score_arms = oof_scores(rank_frame)
rank_frames = []
summary_rows = []
for arm, scores in score_arms.items():
    ranks = event_ranks(rank_frame, truth_df, scores, arm)
    rank_frames.append(ranks)
    summary_rows.append({'arm': arm, **summarize(ranks)})
event_ranks_df = pd.concat(rank_frames, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)
display(summary_df)
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp037_soft_parent_evidence')
out.mkdir(parents=True, exist_ok=True)
truth_df.to_csv(out / 'truth_events.csv', index=False)
transition_df.to_csv(out / 'transition_summary.csv', index=False)
summary_df.to_csv(out / 'summary.csv', index=False)
event_ranks_df.to_csv(out / 'event_ranks.csv', index=False)
summary = {
    'experiment': 'EXP037', 'official_cv': None, 'submission_generated': False,
    'parent_oracle_used': False, 'candidate_count': int(len(rank_frame)),
    'truth_events': int(len(truth_df)), 'transitions': int(transition_df.transition_id.nunique()),
    'arms': list(score_arms), 'selection': 'soft_parent_features_full_candidate_rank',
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = [
    '# EXP037 soft parent evidence', '',
    'Diagnostic only; no submission and no official CV.', '',
    f"Candidates={summary['candidate_count']}; truth events={summary['truth_events']}; transitions={summary['transitions']}.", '',
    '```text', summary_df.to_string(index=False), '```', '',
    'Parent aggregate evidence is used as a soft candidate feature. No parent Top-K gate is applied.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            cell(EMBEDDED_MODULES(), "code", "embedded-modules"),
            cell(SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(PARENT_CANDIDATES, "code", "parent-evidence-candidates"),
            cell(SOFT_RANKER, "code", "soft-parent-ranker"),
            cell(OUTPUTS, "code", "outputs"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP037 soft parent evidence"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
