"""Create EXP039: precision-first parent/daughter distance gate sweep."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp038_continuation_conflict as source
except ImportError:  # pragma: no cover
    import create_exp038_continuation_conflict as source  # type: ignore


ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP039" / "CELL_precision_gate_sweep.ipynb"


GATE_RANKER = r'''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ['parent_distance_um', 'daughter_distance_um', 'focus_only_count']
STRUCTURAL_FEATURES = [
    'daughter1_incoming_rank', 'daughter2_incoming_rank',
    'daughter1_incoming_margin_um', 'daughter2_incoming_margin_um',
    'incoming_margin_min_um', 'incoming_margin_mean_um',
    'candidate_parent_closest_both',
    'daughter1_forward_distance_um', 'daughter2_forward_distance_um',
    'daughter_forward_distance_mean_um', 'daughter_forward_distance_max_um',
    'daughter_forward_count',
]
HARD_NEGATIVES_PER_TRANSITION = 256
RANDOM_NEGATIVES_PER_TRANSITION = 256
PARENT_GATES_UM = [8.0, 10.0, 12.0, 14.0, 16.0]
DAUGHTER_GATES_UM = [10.0, 12.0, 14.0, 16.0, 20.0]


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
            random_state=20260921, solver='liblinear',
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


def oof_scores(frame):
    scores = {
        'candidate_base': np.full(len(frame), np.nan, dtype=float),
        'continuation_conflict': np.full(len(frame), np.nan, dtype=float),
    }
    videos = frame['video'].astype(str).to_numpy()
    unique_videos = sorted(set(videos))
    for fold, video in enumerate(unique_videos):
        valid_idx = np.flatnonzero(videos == video)
        train_pool = np.flatnonzero(videos != video)
        train_candidates = frame.loc[sampled_candidate_indices(frame, train_pool, 20260921 + fold)]
        valid_candidates = frame.loc[valid_idx]
        if train_candidates['label'].nunique() != 2:
            raise RuntimeError(f'EXP039 fold {video} has one class')
        scores['candidate_base'][valid_idx] = model_for(
            train_candidates, valid_candidates, BASE_FEATURES
        )
        scores['continuation_conflict'][valid_idx] = model_for(
            train_candidates, valid_candidates, BASE_FEATURES + STRUCTURAL_FEATURES
        )
        print('OOF', fold + 1, '/', len(unique_videos), video,
              'train_candidates', len(train_candidates),
              'valid_candidates', len(valid_candidates), flush=True)
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise RuntimeError('EXP039 OOF scores contain NaN')
    return scores


def rank_one_gate(frame, truth, scores, arm, parent_gate, daughter_gate):
    keep = (
        frame['parent_distance_um'].astype(float).to_numpy() <= float(parent_gate)
    ) & (
        frame['daughter_distance_um'].astype(float).to_numpy() <= float(daughter_gate)
    )
    filtered = frame.loc[keep, ['transition_id', 'candidate_id']].copy()
    filtered['score'] = scores[keep]
    lookup = dict(zip(filtered['candidate_id'].astype(str), filtered['score'].astype(float)))
    grouped = {
        str(transition_id): group['score'].to_numpy(float)
        for transition_id, group in filtered.groupby('transition_id', sort=False)
    }
    rows = []
    for row in truth.itertuples(index=False):
        candidate_id = str(row.positive_candidate_id)
        found = candidate_id in lookup
        rank = np.nan
        if found:
            positive_score = float(lookup[candidate_id])
            values = grouped[str(row.transition_id)]
            rank = 1.0 + float(np.sum(values > positive_score))
            rank += 0.5 * float(np.sum(np.isclose(values, positive_score, rtol=0.0, atol=1e-12)) - 1)
        rows.append({
            'arm': arm,
            'parent_gate_um': float(parent_gate),
            'daughter_gate_um': float(daughter_gate),
            'gate': f'p{int(parent_gate):02d}_d{int(daughter_gate):02d}',
            'video': row.video,
            'transition_id': row.transition_id,
            'event_id': row.event_id,
            'candidate_found': int(found),
            'positive_rank': rank,
        })
    ranks = pd.DataFrame(rows)
    rankable = ranks[ranks['candidate_found'] == 1]
    values = rankable['positive_rank'].to_numpy(float)
    total_events = int(len(ranks))
    found_count = int(len(rankable))
    return {
        'arm': arm,
        'parent_gate_um': float(parent_gate),
        'daughter_gate_um': float(daughter_gate),
        'gate': f'p{int(parent_gate):02d}_d{int(daughter_gate):02d}',
        'candidate_count': int(len(filtered)),
        'candidate_recall_count': found_count,
        'candidate_recall': float(found_count / total_events) if total_events else 0.0,
        'events': total_events,
        'rankable_events': found_count,
        'top1_all_count': int(np.sum((ranks['candidate_found'] == 1) & (ranks['positive_rank'] <= 1))),
        'top3_all_count': int(np.sum((ranks['candidate_found'] == 1) & (ranks['positive_rank'] <= 3))),
        'top5_all_count': int(np.sum((ranks['candidate_found'] == 1) & (ranks['positive_rank'] <= 5))),
        'top1_all_recall': float(np.sum((ranks['candidate_found'] == 1) & (ranks['positive_rank'] <= 1)) / total_events) if total_events else 0.0,
        'top3_all_recall': float(np.sum((ranks['candidate_found'] == 1) & (ranks['positive_rank'] <= 3)) / total_events) if total_events else 0.0,
        'top5_all_recall': float(np.sum((ranks['candidate_found'] == 1) & (ranks['positive_rank'] <= 5)) / total_events) if total_events else 0.0,
        'top1_given_found': float(np.sum(values <= 1) / found_count) if found_count else 0.0,
        'top3_given_found': float(np.sum(values <= 3) / found_count) if found_count else 0.0,
        'median_rank_given_found': float(np.median(values)) if found_count else np.nan,
    }, ranks


rank_frame = candidate_df.copy().reset_index(drop=True)
if rank_frame.empty or int(rank_frame['label'].sum()) != int(truth_df['candidate_found'].sum()):
    raise RuntimeError('EXP039 candidate/truth manifest mismatch')
score_arms = oof_scores(rank_frame)
summary_rows = []
rank_rows = []
for parent_gate in PARENT_GATES_UM:
    for daughter_gate in DAUGHTER_GATES_UM:
        for arm, scores in score_arms.items():
            summary, ranks = rank_one_gate(
                rank_frame, truth_df, scores, arm, parent_gate, daughter_gate
            )
            summary_rows.append(summary)
            rank_rows.append(ranks)
summary_df = pd.DataFrame(summary_rows)
event_ranks_df = pd.concat(rank_rows, ignore_index=True)
display(summary_df.sort_values(['arm', 'top1_all_recall', 'candidate_count'], ascending=[True, False, True]))
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp039_precision_gate_sweep')
out.mkdir(parents=True, exist_ok=True)
truth_df.to_csv(out / 'truth_events.csv', index=False)
transition_df.to_csv(out / 'transition_summary.csv', index=False)
summary_df.to_csv(out / 'gate_summary.csv', index=False)
event_ranks_df.to_csv(out / 'event_ranks.csv', index=False)
summary = {
    'experiment': 'EXP039', 'official_cv': None, 'submission_generated': False,
    'parent_oracle_used': False, 'max_candidate_count': int(len(rank_frame)),
    'truth_events': int(len(truth_df)), 'transitions': int(transition_df.transition_id.nunique()),
    'parent_gates_um': PARENT_GATES_UM, 'daughter_gates_um': DAUGHTER_GATES_UM,
    'arms': list(score_arms), 'selection': 'fixed_oof_ranker_precision_first_gate_sweep',
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = [
    '# EXP039 precision-first distance gate sweep', '',
    'Diagnostic only; no submission and no official CV.', '',
    f"Maximum candidates={summary['max_candidate_count']}; truth events={summary['truth_events']}; transitions={summary['transitions']}.", '',
    'The OOF ranker is fixed at the maximum 16/20 um pool; only candidate gates are changed.', '',
    '```text', summary_df.sort_values(['arm', 'top1_all_recall', 'candidate_count'], ascending=[True, False, True]).to_string(index=False), '```', '',
    'Top-1 all-recall counts missing candidates as failures. No non-division transitions are included in this positive-event panel; official scorer validation remains required.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            source.source.cell(source.source.EMBEDDED_MODULES(), "code", "embedded-modules"),
            source.source.cell(source.source.SETUP, "code", "setup-and-event-sampling"),
            source.source.cell(source.source.PILKWANG, "code", "pilkwang-centers"),
            source.source.cell(source.source.FOCUS, "code", "focus3d-instances"),
            source.source.cell(source.continuation_candidates(), "code", "maximum-candidate-pool"),
            source.source.cell(GATE_RANKER, "code", "precision-first-gate-sweep"),
            source.source.cell(OUTPUTS, "code", "outputs"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP039 precision-first distance gate sweep"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
