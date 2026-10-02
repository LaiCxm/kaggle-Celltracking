"""Create EXP029: wider-video OOF training of a division candidate ranker."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .create_exp027_structural_selection import (
        CANDIDATE_AND_SELECTION, FOCUS, PILKWANG, ROOT, SETUP, cell, embedded_modules,
    )
except ImportError:
    from create_exp027_structural_selection import (  # type: ignore
        CANDIDATE_AND_SELECTION, FOCUS, PILKWANG, ROOT, SETUP, cell, embedded_modules,
    )


OUT = ROOT / "EXP" / "EXP029" / "CELL_train_oof_division_candidate_ranker.ipynb"

# Use twice the EXP027 video panel and cap events per video so the validation
# remains broader without turning the diagnostic into a full-movie submission.
WIDE_SETUP = SETUP.replace("TARGET_VIDEO_COUNT = 12", "TARGET_VIDEO_COUNT = 24").replace(
    "MAX_EVENTS_PER_VIDEO = 3", "MAX_EVENTS_PER_VIDEO = 2"
)

RANKER = r'''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from focus_first.candidate_scoring import event_ranking_metrics, leave_one_group_out_splits

rank_frame = candidate_df[candidate_df["mode"] == "union"].copy().reset_index(drop=True)
RANK_FEATURES = ["parent_distance_um", "daughter_distance_um", "focus_only_count"]
if rank_frame.empty:
    raise RuntimeError("EXP029 union candidate pool is empty")
if int(rank_frame["label"].sum()) != int(rank_frame["event_id"].nunique()):
    raise RuntimeError("EXP029 expects one positive union candidate per event")


def _fixed_scores(frame, daughter_weight, focus_cost):
    return -(
        frame["parent_distance_um"].astype(float).to_numpy()
        + float(daughter_weight) * frame["daughter_distance_um"].astype(float).to_numpy()
        + float(focus_cost) * frame["focus_only_count"].astype(float).to_numpy()
    )


def _oof_scores(frame):
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    groups = frame["video"].astype(str).tolist()
    for fold, (train_idx, valid_idx) in enumerate(leave_one_group_out_splits(groups)):
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        if train["label"].nunique() != 2:
            raise RuntimeError(f"OOF fold {fold} has only one class")
        train_x = train[RANK_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
        valid_x = valid[RANK_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
        for feature in RANK_FEATURES:
            median = train_x[feature].median()
            if not np.isfinite(median):
                median = 0.0
            train_x[feature] = train_x[feature].fillna(float(median))
            valid_x[feature] = valid_x[feature].fillna(float(median))
        model = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                random_state=20260917, solver="liblinear",
            )),
        ])
        model.fit(train_x, train["label"].astype(int).to_numpy())
        scores[valid_idx] = model.predict_proba(valid_x)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("OOF scores contain NaN")
    return scores


def _items(frame, scores):
    items = []
    for row, score in zip(frame.itertuples(index=False), scores):
        items.append(DivisionSelectionCandidate(
            candidate_id=str(row.candidate_id), parent_id=str(row.parent_id),
            daughter1_id=str(row.daughter1_id), daughter2_id=str(row.daughter2_id),
            event_id=str(row.event_id), geometry_cost=float(-score),
            focus_only_count=0, label=int(row.label),
        ))
    return items


def _selection_metrics(frame, scores):
    selected = []
    for video, group in frame.assign(_score=scores).groupby("video", sort=True):
        chosen = select_nonconflicting_divisions(
            _items(group, group["_score"].to_numpy(dtype=float)),
            focus_activation_cost=0.0,
        )
        for item in chosen:
            selected.append({
                "video": video, "event_id": item.event_id,
                "candidate_id": item.candidate_id, "label": int(item.label),
            })
    selected_count = len(selected)
    selected_positive = sum(int(row["label"] == 1) for row in selected)
    events = int(frame["event_id"].nunique())
    return {
        "selected_count": selected_count,
        "selected_positive_count": selected_positive,
        "selected_negative_count": selected_count - selected_positive,
        "selection_precision": float(selected_positive / selected_count) if selected_count else 0.0,
        "event_hit_rate": float(selected_positive / events) if events else 0.0,
    }, selected


ARMS = {
    "geometry_1_05_focus0": _fixed_scores(rank_frame, 0.5, 0.0),
    "geometry_1_05_focus4": _fixed_scores(rank_frame, 0.5, 4.0),
    "geometry_1_00_focus4": _fixed_scores(rank_frame, 1.0, 4.0),
    "oof_logistic_3feat": _oof_scores(rank_frame),
}
arm_rows = []
score_rows = []
selected_rows = []
for arm, scores in ARMS.items():
    rank_metrics = event_ranking_metrics(rank_frame.event_id, rank_frame.label, scores)
    select_metrics, chosen_rows = _selection_metrics(rank_frame, scores)
    arm_rows.append({"arm": arm, **rank_metrics, **select_metrics})
    for index, score in enumerate(scores):
        row = rank_frame.iloc[index]
        score_rows.append({
            "arm": arm, "video": row.video, "event_id": row.event_id,
            "candidate_id": row.candidate_id, "label": int(row.label), "score": float(score),
        })
    for row in chosen_rows:
        row["arm"] = arm
        selected_rows.append(row)

arm_summary_df = pd.DataFrame(arm_rows)
rank_scores_df = pd.DataFrame(score_rows)
rank_selected_df = pd.DataFrame(selected_rows)
print("EXP029 OOF ranking summary")
display(arm_summary_df)
'''


OUTPUTS = r'''out = Path('/kaggle/working/exp029_oof_candidate_ranker')
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
    'experiment': 'EXP029', 'official_cv': None, 'submission_generated': False,
    'selected_videos': selected_stems, 'events': int(event_df.event_id.nunique()),
    'union_candidates': int(len(rank_frame)), 'rank_features': RANK_FEATURES,
    'oof_group': 'video', 'arms': list(ARMS),
    'candidate_gate_um': {'parent': PARENT_RADIUS_UM, 'daughter': DAUGHTER_RADIUS_UM},
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
report = [
    '# EXP029 wider-video OOF division candidate ranker', '',
    'Diagnostic only; no submission and no official CV.', '',
    f"Videos={len(selected_stems)}; events={summary['events']}; union candidates={summary['union_candidates']}.", '',
    '```text', arm_summary_df.to_string(index=False), '```', '',
    'The learned arm is trained only within leave-one-video-out folds. '
    'Pilkwang and FOCUS3D weights remain frozen.',
]
(out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Wrote:', sorted(path.name for path in out.iterdir()))
'''


def main() -> None:
    notebook = {
        "cells": [
            cell(
                "# EXP029 wider-video OOF division candidate ranker\n\n"
                "Train a lightweight candidate ranker on a wider video panel while keeping "
                "Pilkwang and FOCUS3D weights frozen.",
                "markdown", "title",
            ),
            cell(embedded_modules(), "code", "embedded-modules"),
            cell(WIDE_SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(CANDIDATE_AND_SELECTION, "code", "candidate-pool-and-geometry-baseline"),
            cell(RANKER, "code", "oof-ranker-and-structural-selection"),
            cell(OUTPUTS, "code", "write-diagnostics"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP029 wider-video OOF division candidate ranker"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
