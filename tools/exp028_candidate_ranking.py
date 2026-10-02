"""EXP028: grouped OOF ranking on the EXP027 expanded division pool.

This is a local, label-audited diagnostic.  Labels are used only inside each
leave-one-video-out training fold and for the held-out evaluation; the fixed
arms never use labels at scoring time.  No submission or official CV is made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.focus_first.candidate_scoring import event_ranking_metrics, leave_one_group_out_splits
from tools.focus_first.global_selection import DivisionSelectionCandidate, select_nonconflicting_divisions


INPUT = ROOT / "EXP" / "EXP027" / "outputs" / "kaggle_v2" / "exp027_structural_selection" / "candidate_rows.csv"
OUT = ROOT / "EXP" / "EXP028" / "outputs" / "local_oof"
SEED = 20260917

FEATURES = ["parent_distance_um", "daughter_distance_um", "focus_only_count"]


def load_candidates(path: Path = INPUT) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "video", "event_id", "mode", "candidate_id", "label", "parent_id",
        "daughter1_id", "daughter2_id", "parent_distance_um", "daughter_distance_um",
        "focus_only_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate input is missing columns: {missing}")
    frame = frame[frame["mode"].astype(str) == "union"].copy()
    if frame.empty:
        raise ValueError("EXP028 requires the EXP027 union candidate pool")
    frame["video"] = frame["video"].astype(str)
    frame["event_id"] = frame["event_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    if frame["label"].sum() != frame["event_id"].nunique():
        raise ValueError("EXP028 expects exactly one positive candidate per union event")
    return frame.reset_index(drop=True)


def fixed_cost(frame: pd.DataFrame, *, daughter_weight: float, focus_cost: float) -> np.ndarray:
    return (
        frame["parent_distance_um"].to_numpy(dtype=float)
        + float(daughter_weight) * frame["daughter_distance_um"].to_numpy(dtype=float)
        + float(focus_cost) * frame["focus_only_count"].to_numpy(dtype=float)
    )


def fit_oof_logistic(frame: pd.DataFrame) -> np.ndarray:
    scores = np.full(len(frame), np.nan, dtype=float)
    groups = frame["video"].tolist()
    for fold, (train_idx, valid_idx) in enumerate(leave_one_group_out_splits(groups)):
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        x_train = train[FEATURES].copy()
        x_valid = valid[FEATURES].copy()
        for feature in FEATURES:
            median = x_train[feature].median()
            if not np.isfinite(median):
                median = 0.0
            x_train[feature] = x_train[feature].fillna(float(median))
            x_valid[feature] = x_valid[feature].fillna(float(median))
        if train["label"].nunique() != 2:
            raise ValueError(f"LOO fold {fold} does not contain both classes")
        model = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                random_state=SEED, solver="liblinear",
            )),
        ])
        model.fit(x_train, train["label"].to_numpy(dtype=int))
        scores[valid_idx] = model.predict_proba(x_valid)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("OOF scoring left missing candidate rows")
    return scores


def _selection_items(rows: pd.DataFrame, scores: Iterable[float]) -> list[DivisionSelectionCandidate]:
    result = []
    for row, score in zip(rows.itertuples(index=False), scores):
        result.append(DivisionSelectionCandidate(
            candidate_id=str(row.candidate_id),
            parent_id=str(row.parent_id),
            daughter1_id=str(row.daughter1_id),
            daughter2_id=str(row.daughter2_id),
            event_id=str(row.event_id),
            geometry_cost=float(-score),
            focus_only_count=0,
            label=int(row.label),
        ))
    return result


def _structural_metrics(frame: pd.DataFrame, scores: np.ndarray) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    chosen_rows: list[dict[str, object]] = []
    for video, group in frame.assign(_score=scores).groupby("video", sort=True):
        items = _selection_items(group, group["_score"].to_numpy(dtype=float))
        selected = select_nonconflicting_divisions(items, focus_activation_cost=0.0)
        for item in selected:
            chosen_rows.append({
                "video": video,
                "event_id": item.event_id,
                "candidate_id": item.candidate_id,
                "label": int(item.label),
            })
    selected_count = len(chosen_rows)
    selected_positive = sum(int(row["label"] == 1) for row in chosen_rows)
    events = int(frame["event_id"].nunique())
    return ({
        "events": events,
        "selected_count": selected_count,
        "selected_positive_count": selected_positive,
        "selected_negative_count": selected_count - selected_positive,
        "selection_precision": float(selected_positive / selected_count) if selected_count else 0.0,
        "event_hit_rate": float(selected_positive / events) if events else 0.0,
    }, chosen_rows)


def _video_metrics(frame: pd.DataFrame, scores: np.ndarray, arm: str) -> pd.DataFrame:
    scored = frame[["video", "event_id", "label"]].copy()
    scored["score"] = scores
    rows: list[dict[str, object]] = []
    for video, group in scored.groupby("video", sort=True):
        top1 = 0
        for _, event in group.groupby("event_id", sort=True):
            ordered = event.sort_values(["score"], ascending=[False], kind="mergesort")
            top1 += int(int(ordered.iloc[0]["label"]) == 1)
        selected = select_nonconflicting_divisions(
            _selection_items(frame.loc[group.index], group["score"].to_numpy(dtype=float)),
            focus_activation_cost=0.0,
        )
        selected_positive = sum(int(item.label == 1) for item in selected)
        rows.append({
            "arm": arm,
            "video": video,
            "events": int(group["event_id"].nunique()),
            "top1_positive": top1,
            "selected_count": len(selected),
            "selected_positive_count": selected_positive,
        })
    return pd.DataFrame(rows)


def run(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    arms = {
        "geometry_1_05_focus0": ("fixed", {"daughter_weight": 0.5, "focus_cost": 0.0}),
        "geometry_1_05_focus4": ("fixed", {"daughter_weight": 0.5, "focus_cost": 4.0}),
        "geometry_1_00_focus4": ("fixed", {"daughter_weight": 1.0, "focus_cost": 4.0}),
        "oof_logistic_3feat": ("learned", {}),
    }
    score_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    video_frames: list[pd.DataFrame] = []
    for arm, (kind, params) in arms.items():
        if kind == "fixed":
            cost = fixed_cost(frame, **params)
            scores = -cost
        else:
            scores = fit_oof_logistic(frame)
        scored = frame[["video", "event_id", "candidate_id", "label"]].copy()
        scored["arm"] = arm
        scored["score"] = scores
        score_rows.append(scored)
        rank = event_ranking_metrics(scored["event_id"], scored["label"], scored["score"])
        structure, chosen = _structural_metrics(frame, scores)
        video_frames.append(_video_metrics(frame, scores, arm))
        for row in chosen:
            row["arm"] = arm
        selected_rows.extend(chosen)
        summary_rows.append({"arm": arm, **rank, **structure})
    score_df = pd.concat(score_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    selected_df = pd.DataFrame(selected_rows)
    video_df = pd.concat(video_frames, ignore_index=True)
    manifest = {
        "experiment": "EXP028",
        "input": str(INPUT),
        "candidate_mode": "union",
        "videos": int(frame["video"].nunique()),
        "events": int(frame["event_id"].nunique()),
        "candidates": int(len(frame)),
        "features": FEATURES,
        "oof_group": "video",
        "arms": list(arms),
        "official_cv": None,
        "submission_generated": False,
    }
    return summary_df, selected_df, score_df, video_df, manifest


def main() -> None:
    frame = load_candidates()
    summary_df, selected_df, score_df, video_df, manifest = run(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT / "arm_summary.csv", index=False)
    selected_df.to_csv(OUT / "selected_candidates.csv", index=False)
    score_df.to_csv(OUT / "candidate_oof_scores.csv", index=False)
    video_df.to_csv(OUT / "video_summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = [
        "# EXP028 候选级 OOF 排序与结构选择诊断", "",
        "本实验使用 EXP027 的 union 候选池，按视频留一做 OOF；不生成 submission.csv，不运行官方 scorer。", "",
        "```text", summary_df.to_string(index=False), "```", "",
        "固定排序臂不读取标签；逻辑回归只在训练视频上读取标签，并在留出视频上预测。"
        "结构选择在每个留出视频内执行，避免跨视频节点冲突。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
