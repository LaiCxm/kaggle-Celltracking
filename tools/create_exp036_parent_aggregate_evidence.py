"""Create EXP036 by replacing EXP035 parent evidence with aggregate evidence."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp035_hierarchical_parent_gate as source
except ImportError:  # pragma: no cover
    import create_exp035_hierarchical_parent_gate as source  # type: ignore


ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP036" / "CELL_train_parent_aggregate_evidence.ipynb"
SETUP = source.SETUP
EMBEDDED_MODULES = source.EMBEDDED_MODULES
PILKWANG = source.PILKWANG
FOCUS = source.FOCUS
PARENT_CANDIDATES = source.PARENT_CANDIDATES
cell = source.cell

PARENT_AGGREGATE_FUNCTION = r'''def parent_frame(frame):
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
            'parent_center_score': float(group['parent_center_score'].iloc[0]),
            'parent_is_consensus': int(group['parent_is_consensus'].iloc[0]),
            'parent_focus_quality': float(group['parent_focus_quality'].iloc[0]),
            'parent_backward_distance_um': float(group['parent_backward_distance_um'].iloc[0]),
            'parent_forward_count': int(group['parent_forward_count'].iloc[0]),
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
            'label': int(group['label'].max()),
        })
    return pd.DataFrame(rows)

'''


def build_ranker() -> str:
    text = source.HIERARCHICAL_RANKER
    text = text.replace(
        "K_VALUES = [1, 2, 4, 8, 16, 32, 64]",
        "K_VALUES = [8, 16, 32, 64, 128]",
    )
    text = text.replace(
        "GATE_FEATURES = PARENT_FEATURES\n",
        "GATE_FEATURES = [\n"
        "    'parent_center_score', 'parent_is_consensus', 'parent_focus_quality',\n"
        "    'parent_backward_distance_um', 'parent_forward_count',\n"
        "    'candidate_count', 'best_geometry_cost', 'second_geometry_cost',\n"
        "    'geometry_gap', 'best_volume_balance', 'best_volume_conservation',\n"
        "    'best_parent_union_overlap', 'best_parent_union_parent_coverage',\n"
        "    'best_parent_union_daughter_coverage', 'best_focus_only_count',\n"
        "]\n",
    )
    start = text.index("def parent_frame(frame):")
    end = text.index("def oof_predictions(frame):")
    text = text[:start] + PARENT_AGGREGATE_FUNCTION + text[end:]
    text = text.replace("EXP035", "EXP036")
    text = text.replace("hierarchical parent gate", "parent aggregate evidence")
    return text


HIERARCHICAL_RANKER = build_ranker()
OUTPUTS = source.OUTPUTS.replace(
    "Path('/kaggle/working/exp035_hierarchical_parent_gate')",
    "Path('/kaggle/working/exp036_parent_aggregate_evidence')",
).replace('EXP035', 'EXP036').replace(
    'parent_top_k_then_candidate_rank', 'parent_aggregate_top_k_then_candidate_rank'
)


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
            "kaggle": {"title": "EXP036 parent aggregate evidence"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
