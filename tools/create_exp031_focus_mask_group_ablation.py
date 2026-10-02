"""Create EXP031: grouped ablation of FOCUS3D candidate evidence."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp030_rich_candidate_ranker as base
except ImportError:  # pragma: no cover
    import create_exp030_rich_candidate_ranker as base  # type: ignore


ROOT = base.ROOT
OUT = ROOT / "EXP" / "EXP031" / "CELL_train_focus_mask_group_ablation.ipynb"
WIDE_SETUP = base.WIDE_SETUP
CANDIDATE_AND_MASK_FEATURES = base.CANDIDATE_AND_MASK_FEATURES
PILKWANG = base.PILKWANG
FOCUS = base.FOCUS
EMBEDDED_MODULES = base.embedded_modules
cell = base.cell


RANKER = r'''from sklearn.linear_model import LogisticRegression
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
PURE_MASK_FEATURES = [
    'volume_balance', 'volume_conservation',
    'parent_union_overlap', 'parent_union_parent_coverage',
    'parent_union_daughter_coverage',
]
QUALITY_FEATURES = ['focus_quality_mean']
SOURCE_FEATURES = ['focus_instance_count', 'consensus_count']
SEPARATION_FEATURES = ['daughter_separation_score']
GROUPS = {
    'base': BASE_FEATURES,
    'base_plus_mask_geometry': BASE_FEATURES + PURE_MASK_FEATURES,
    'base_plus_mask_quality': BASE_FEATURES + PURE_MASK_FEATURES + QUALITY_FEATURES,
    'base_plus_mask_source': BASE_FEATURES + PURE_MASK_FEATURES + SOURCE_FEATURES,
    'rich_all': BASE_FEATURES + PURE_MASK_FEATURES + QUALITY_FEATURES + SOURCE_FEATURES + SEPARATION_FEATURES,
}
if rank_frame.empty:
    raise RuntimeError('EXP031 union candidate pool is empty')
if int(rank_frame['label'].sum()) != int(rank_frame['event_id'].nunique()):
    raise RuntimeError('EXP031 expects exactly one positive candidate per union event')


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


def oof_scores(frame, features):
    scores = np.full(len(frame), np.nan, dtype=float)
    groups = frame['video'].astype(str).tolist()
    for fold, (train_idx, valid_idx) in enumerate(leave_one_group_out_splits(groups)):
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        if train['label'].nunique() != 2:
            raise RuntimeError(f'OOF fold {fold} has one class')
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


def rank_table(frame, scores):
    rows = []
    for (video, event_id), group in frame.assign(_score=scores).groupby(
        ['video', 'event_id'], sort=True
    ):
        ordered = group.sort_values('_score', ascending=False, kind='mergesort')
        positive = np.flatnonzero(ordered['label'].to_numpy() == 1)
        if len(positive) != 1:
            raise RuntimeError(f'event {video}:{event_id} has {len(positive)} positives')
        rows.append({'video': video, 'event_id': event_id, 'positive_rank': int(positive[0] + 1)})
    return pd.DataFrame(rows)


def bootstrap_video_delta(ranks, arm, baseline, iterations=10_000, seed=20260919):
    pair = ranks[ranks['arm'].isin([arm, baseline])].pivot_table(
        index=['video', 'event_id'], columns='arm', values='positive_rank', aggfunc='first'
    ).reset_index()
    if arm not in pair or baseline not in pair:
        raise RuntimeError(f'missing paired arm {arm}/{baseline}')
    pair['_delta'] = (pair[arm] <= 1).astype(float) - (pair[baseline] <= 1).astype(float)
    per_video = pair.groupby('video')['_delta'].sum().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(per_video, size=(iterations, len(per_video)), replace=True).mean(axis=1)
    return {
        'arm': arm, 'baseline': baseline, 'videos': int(len(per_video)),
        'events': int(len(pair)), 'mean_delta_events_per_video': float(per_video.mean()),
        'ci95_low': float(np.quantile(samples, 0.025)),
        'ci95_high': float(np.quantile(samples, 0.975)),
        'positive_videos': int(np.sum(per_video > 0)),
        'zero_videos': int(np.sum(per_video == 0)),
        'negative_videos': int(np.sum(per_video < 0)),
    }


arm_scores = {
    'geometry_1_05_focus0': ('fixed', BASE_FEATURES, fixed_scores(rank_frame, 0.5, 0.0)),
    'logistic_base': ('logistic', GROUPS['base'], oof_scores(rank_frame, GROUPS['base'])),
    'logistic_mask_geometry': ('logistic', GROUPS['base_plus_mask_geometry'], oof_scores(rank_frame, GROUPS['base_plus_mask_geometry'])),
    'logistic_mask_quality': ('logistic', GROUPS['base_plus_mask_quality'], oof_scores(rank_frame, GROUPS['base_plus_mask_quality'])),
    'logistic_mask_source': ('logistic', GROUPS['base_plus_mask_source'], oof_scores(rank_frame, GROUPS['base_plus_mask_source'])),
    'logistic_rich_all': ('logistic', GROUPS['rich_all'], oof_scores(rank_frame, GROUPS['rich_all'])),
}
permuted_mask = pd.DataFrame(permute_columns_within_groups(
    rank_frame.to_dict('records'), columns=PURE_MASK_FEATURES,
    group_key='video', seed=20260919,
))
arm_scores['logistic_mask_geometry_permuted'] = (
    'logistic', GROUPS['base_plus_mask_geometry'], oof_scores(permuted_mask, GROUPS['base_plus_mask_geometry'])
)

arm_rows = []
score_rows = []
selected_rows = []
rank_rows = []
for arm, (_, features, scores) in arm_scores.items():
    rank_metrics = event_ranking_metrics(rank_frame.event_id, rank_frame.label, scores)
    structure, chosen = structural_metrics(rank_frame, scores)
    arm_rows.append({
        'arm': arm, 'features': json.dumps(features),
        **rank_metrics, **structure,
    })
    ranks = rank_table(rank_frame, scores)
    ranks['arm'] = arm
    rank_rows.append(ranks)
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
event_ranks_df = pd.concat(rank_rows, ignore_index=True)
bootstrap_df = pd.DataFrame([
    bootstrap_video_delta(event_ranks_df, arm, 'logistic_base')
    for arm in [
        'logistic_mask_geometry', 'logistic_mask_quality',
        'logistic_mask_source', 'logistic_rich_all',
        'logistic_mask_geometry_permuted',
    ]
])
print('EXP031 grouped FOCUS3D mask ablation')
display(arm_summary_df)
display(bootstrap_df)
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp031_focus_mask_group_ablation')
out.mkdir(parents=True, exist_ok=True)
candidate_df.to_csv(out / 'candidate_rows.csv', index=False)
endpoint_df.to_csv(out / 'endpoint_rows.csv', index=False)
event_df.to_csv(out / 'event_rows.csv', index=False)
selection_df.to_csv(out / 'geometry_selection_summary.csv', index=False)
selected_df.to_csv(out / 'geometry_selected_candidates.csv', index=False)
arm_summary_df.to_csv(out / 'arm_summary.csv', index=False)
rank_scores_df.to_csv(out / 'candidate_oof_scores.csv', index=False)
rank_selected_df.to_csv(out / 'ranker_selected_candidates.csv', index=False)
event_ranks_df.to_csv(out / 'event_ranks.csv', index=False)
bootstrap_df.to_csv(out / 'paired_video_bootstrap.csv', index=False)
summary = {
    'experiment': 'EXP031', 'official_cv': None, 'submission_generated': False,
    'selected_videos': selected_stems, 'events': int(event_df.event_id.nunique()),
    'union_candidates': int(len(rank_frame)), 'feature_groups': GROUPS,
    'oof_group': 'video', 'bootstrap_iterations': 10_000,
    'candidate_gate_um': {'parent': PARENT_RADIUS_UM, 'daughter': DAUGHTER_RADIUS_UM},
    'arms': list(arm_scores),
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = [
    '# EXP031 grouped FOCUS3D mask ablation', '',
    'Diagnostic only; no submission and no official CV.', '',
    f"Videos={len(selected_stems)}; events={summary['events']}; union candidates={summary['union_candidates']}.", '',
    '```text', arm_summary_df.to_string(index=False), '```', '',
    'The original Pilkwang and FOCUS3D weights are frozen. The experiment separates '
    'mask geometry, FOCUS instance quality, and source-composition evidence under '
    'leave-one-video-out scoring. The paired bootstrap is clustered by video.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    notebook = {
        "cells": [
            cell(
                "# EXP031 grouped FOCUS3D mask ablation\n\n"
                "Separate FOCUS3D mask geometry, instance quality, and source composition "
                "under leave-one-video-out candidate ranking.",
                "markdown", "title",
            ),
            cell(EMBEDDED_MODULES(), "code", "embedded-modules"),
            cell(WIDE_SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(CANDIDATE_AND_MASK_FEATURES, "code", "candidate-and-mask-features"),
            cell(RANKER, "code", "grouped-oof-rankers"),
            cell(OUTPUTS, "code", "write-diagnostics"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP031 grouped FOCUS3D mask ablation"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
