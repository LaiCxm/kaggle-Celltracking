"""Create EXP035: hierarchical parent gate followed by daughter ranking."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp034_parent_evidence as source
except ImportError:  # pragma: no cover
    import create_exp034_parent_evidence as source  # type: ignore


ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP035" / "CELL_train_hierarchical_parent_gate.ipynb"
SETUP = source.SETUP
EMBEDDED_MODULES = source.EMBEDDED_MODULES
PILKWANG = source.PILKWANG
FOCUS = source.FOCUS
cell = source.cell
PARENT_CANDIDATES = source.PARENT_EVIDENCE_CANDIDATES


HIERARCHICAL_RANKER = r'''from sklearn.linear_model import LogisticRegression
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
CANDIDATE_BASE_FEATURES = BASE_FEATURES
CANDIDATE_MASK_FEATURES = BASE_FEATURES + MASK_GEOMETRY_FEATURES
GATE_FEATURES = PARENT_FEATURES
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
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


def model_for(train, valid, features, label='label'):
    train_x, valid_x = prepare(train, valid, features)
    model = Pipeline([
        ('scale', StandardScaler()),
        ('logistic', LogisticRegression(
            C=0.5, class_weight='balanced', max_iter=3000,
            random_state=20260920, solver='liblinear',
        )),
    ])
    model.fit(train_x, train[label].astype(int).to_numpy())
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
    aggregations = {
        'parent_center_score': 'first',
        'parent_is_consensus': 'first',
        'parent_focus_quality': 'first',
        'parent_backward_distance_um': 'first',
        'parent_forward_count': 'first',
        'label': 'max',
    }
    result = (
        frame.groupby(['video', 'transition_id', 'parent_id'], as_index=False)
        .agg(aggregations)
    )
    return result


def rank_parent_scores(frame, scores):
    parent = parent_frame(frame)
    parent_train_scores = scores.groupby(['video', 'transition_id', 'parent_id'], as_index=False)['score'].max()
    parent = parent.merge(
        parent_train_scores,
        on=['video', 'transition_id', 'parent_id'],
        how='left',
        suffixes=('', '_score'),
    )
    parent['parent_rank'] = (
        parent.groupby('transition_id')['score']
        .rank(method='first', ascending=False)
        .astype(int)
    )
    return parent


def oof_predictions(frame):
    candidate_base = np.full(len(frame), np.nan, dtype=float)
    candidate_mask = np.full(len(frame), np.nan, dtype=float)
    parent_score = np.full(len(frame), np.nan, dtype=float)
    videos = frame['video'].astype(str).to_numpy()
    for fold, video in enumerate(sorted(set(videos))):
        valid_idx = np.flatnonzero(videos == video)
        train_pool = np.flatnonzero(videos != video)
        train_candidates = frame.loc[sampled_candidate_indices(frame, train_pool, 20260920 + fold)]
        valid_candidates = frame.loc[valid_idx]
        if train_candidates['label'].nunique() != 2:
            raise RuntimeError(f'EXP035 candidate fold {video} has one class')
        candidate_base[valid_idx] = model_for(
            train_candidates, valid_candidates, CANDIDATE_BASE_FEATURES
        )
        candidate_mask[valid_idx] = model_for(
            train_candidates, valid_candidates, CANDIDATE_MASK_FEATURES
        )

        train_parents = parent_frame(frame.loc[train_pool])
        valid_parents = parent_frame(valid_candidates)
        if train_parents['label'].nunique() != 2:
            raise RuntimeError(f'EXP035 parent fold {video} has one class')
        valid_parent_scores = model_for(train_parents, valid_parents, GATE_FEATURES)
        scored_parents = valid_parents[['video', 'transition_id', 'parent_id']].copy()
        scored_parents['score'] = valid_parent_scores
        scored_parents['parent_rank'] = (
            scored_parents.groupby('transition_id')['score']
            .rank(method='first', ascending=False)
            .astype(int)
        )
        parent_lookup = {
            (str(row.transition_id), str(row.parent_id)): int(row.parent_rank)
            for row in scored_parents.itertuples(index=False)
        }
        parent_score[valid_idx] = [
            float(parent_lookup[(str(row.transition_id), str(row.parent_id))])
            for row in valid_candidates.itertuples(index=False)
        ]
        print('OOF', fold + 1, '/', len(set(videos)), video,
              'train_candidates', len(train_candidates),
              'valid_candidates', len(valid_candidates),
              'valid_parents', len(valid_parents), flush=True)
    if not np.isfinite(candidate_base).all() or not np.isfinite(candidate_mask).all():
        raise RuntimeError('EXP035 candidate OOF scores contain NaN')
    return candidate_base, candidate_mask, parent_score


def event_ranks(frame, truth, scores, parent_ranks, k, arm):
    scored = frame[['transition_id', 'candidate_id', 'parent_id']].copy()
    scored['score'] = scores
    scored['parent_rank'] = parent_ranks
    score_lookup = dict(zip(scored['candidate_id'], scored['score']))
    parent_lookup = dict(zip(scored['candidate_id'], scored['parent_rank']))
    rows = []
    for row in truth.itertuples(index=False):
        positive_parent_rank = int(parent_lookup[str(row.positive_candidate_id)])
        kept = scored[
            (scored['transition_id'].astype(str) == str(row.transition_id))
            & (scored['parent_rank'] <= int(k))
        ]
        if positive_parent_rank > int(k):
            positive_rank = np.nan
        else:
            positive_score = float(score_lookup[str(row.positive_candidate_id)])
            values = kept['score'].to_numpy(float)
            positive_rank = 1.0 + float(np.sum(values > positive_score))
            positive_rank += 0.5 * float(
                np.sum(np.isclose(values, positive_score, rtol=0.0, atol=1e-12)) - 1
            )
        rows.append({
            'arm': arm, 'k': int(k), 'video': row.video,
            'transition_id': row.transition_id, 'event_id': row.event_id,
            'positive_parent_rank': positive_parent_rank,
            'parent_kept': int(positive_parent_rank <= int(k)),
            'positive_rank': positive_rank,
        })
    return pd.DataFrame(rows)


def summarize(ranks, frame, parent_ranks):
    total = len(ranks)
    rankable = ranks['positive_rank'].notna()
    values = ranks.loc[rankable, 'positive_rank'].to_numpy(float)
    return {
        'events': total,
        'parent_recall': float(ranks['parent_kept'].mean()),
        'candidate_count': int(np.sum(parent_ranks <= int(ranks['k'].iloc[0]))),
        'rankable_events': int(rankable.sum()),
        'top1_count': int(np.sum(values <= 1)),
        'top3_count': int(np.sum(values <= 3)),
        'top5_count': int(np.sum(values <= 5)),
        'top10_count': int(np.sum(values <= 10)),
        'top1_recall_all': float(np.sum(values <= 1) / total) if total else 0.0,
        'top3_recall_all': float(np.sum(values <= 3) / total) if total else 0.0,
        'top5_recall_all': float(np.sum(values <= 5) / total) if total else 0.0,
        'median_rank_rankable': float(np.median(values)) if len(values) else np.nan,
    }


rank_frame = candidate_df.copy().reset_index(drop=True)
if rank_frame.empty or int(rank_frame['label'].sum()) != int(truth_df['candidate_found'].sum()):
    raise RuntimeError('EXP035 candidate/truth manifest mismatch')
candidate_base_scores, candidate_mask_scores, parent_ranks = oof_predictions(rank_frame)
rank_frames = []
summary_rows = []
for arm, scores in {
    'candidate_base': candidate_base_scores,
    'candidate_mask_geometry': candidate_mask_scores,
}.items():
    for k in K_VALUES:
        ranks = event_ranks(rank_frame, truth_df, scores, parent_ranks, k, arm)
        rank_frames.append(ranks)
        summary_rows.append({
            'arm': arm, 'k': int(k),
            **summarize(ranks, rank_frame, parent_ranks),
        })
event_ranks_df = pd.concat(rank_frames, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)
display(summary_df)
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp035_hierarchical_parent_gate')
out.mkdir(parents=True, exist_ok=True)
event_ranks_df.to_csv(out / 'event_ranks.csv', index=False)
summary_df.to_csv(out / 'summary.csv', index=False)
truth_df.to_csv(out / 'truth_events.csv', index=False)
transition_df.to_csv(out / 'transition_summary.csv', index=False)
summary = {
    'experiment': 'EXP035', 'official_cv': None, 'submission_generated': False,
    'parent_oracle_used': False, 'candidate_count': int(len(rank_frame)),
    'truth_events': int(len(truth_df)), 'transitions': int(transition_df.transition_id.nunique()),
    'k_values': K_VALUES, 'arms': ['candidate_base', 'candidate_mask_geometry'],
    'selection': 'parent_top_k_then_candidate_rank',
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = [
    '# EXP035 hierarchical parent gate', '',
    'Diagnostic only; no submission and no official CV.', '',
    f"Candidates={summary['candidate_count']}; truth events={summary['truth_events']}; transitions={summary['transitions']}.", '',
    '```text', summary_df.to_string(index=False), '```', '',
    'Parent Top-K is selected from observation-only evidence. The positive parent is not supplied during candidate generation.',
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
            cell(HIERARCHICAL_RANKER, "code", "hierarchical-ranker"),
            cell(OUTPUTS, "code", "outputs"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP035 hierarchical parent gate"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == '__main__':
    main()
