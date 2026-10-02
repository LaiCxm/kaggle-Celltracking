"""Create EXP034: parent-blind ranking with temporal parent evidence."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp033_parent_blind_division as source
except ImportError:  # pragma: no cover
    import create_exp033_parent_blind_division as source  # type: ignore


ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP034" / "CELL_train_parent_evidence_ranker.ipynb"
SETUP = source.SETUP
EMBEDDED_MODULES = source.EMBEDDED_MODULES
PILKWANG = source.PILKWANG
FOCUS = source.FOCUS
cell = source.cell


PARENT_EVIDENCE_CANDIDATES = source.PARENT_BLIND_CANDIDATES
PARENT_EVIDENCE_CANDIDATES = PARENT_EVIDENCE_CANDIDATES.replace('EXP033', 'EXP034')
PARENT_EVIDENCE_CANDIDATES = PARENT_EVIDENCE_CANDIDATES.replace(
    "    for t in transitions:\n        transition_id = f'{stem}:{t}'\n",
    """    for t in transitions:
        transition_id = f'{stem}:{t}'
""",
)
PARENT_EVIDENCE_CANDIDATES = PARENT_EVIDENCE_CANDIDATES.replace(
    "        objects = enumerate_divisions(\n            parent_nodes, daughter_nodes, instances_by_label=instance_lookup,\n",
    """        previous_nodes = nodes_by_t.get(t - 1, [])
        next_nodes = nodes_by_t.get(t + 1, [])

        def distance_um(left, right):
            delta = np.asarray(left.point, dtype=float) - np.asarray(right.point, dtype=float)
            scale = np.asarray(SCALE_UM, dtype=float)
            return float(np.sqrt(np.sum((delta * scale) ** 2)))

        parent_evidence = {}
        for parent in parent_nodes:
            backward = min(
                (distance_um(parent, previous) for previous in previous_nodes),
                default=float(PARENT_RADIUS_UM),
            )
            forward_count = sum(
                int(distance_um(parent, target) <= PARENT_RADIUS_UM)
                for target in next_nodes
            )
            parent_evidence[parent.proposal_id] = {
                'parent_center_score': float(parent.center_score),
                'parent_is_consensus': int(parent.kind == 'consensus'),
                'parent_focus_quality': float(parent.focus_quality),
                'parent_backward_distance_um': float(backward),
                'parent_forward_count': int(forward_count),
            }

        objects = enumerate_divisions(
            parent_nodes, daughter_nodes, instances_by_label=instance_lookup,
""",
)
PARENT_EVIDENCE_CANDIDATES = PARENT_EVIDENCE_CANDIDATES.replace(
    "                'focus_quality_mean': focus_quality_mean,\n            }\n",
    "                'focus_quality_mean': focus_quality_mean,\n                **parent_evidence[str(candidate.parent_id)],\n            }\n",
)


PARENT_EVIDENCE_RANKER = source.PARENT_BLIND_RANKER
PARENT_EVIDENCE_RANKER = PARENT_EVIDENCE_RANKER.replace('EXP033', 'EXP034')
PARENT_EVIDENCE_RANKER = PARENT_EVIDENCE_RANKER.replace(
    "BASE_FEATURES = ['parent_distance_um', 'daughter_distance_um', 'focus_only_count']\n",
    "BASE_FEATURES = ['parent_distance_um', 'daughter_distance_um', 'focus_only_count']\nPARENT_FEATURES = [\n    'parent_center_score', 'parent_is_consensus', 'parent_focus_quality',\n    'parent_backward_distance_um', 'parent_forward_count',\n]\nEVIDENCE_FEATURES = BASE_FEATURES + PARENT_FEATURES\n",
)
PARENT_EVIDENCE_RANKER = PARENT_EVIDENCE_RANKER.replace(
    "RICH_FEATURES = BASE_FEATURES + MASK_GEOMETRY_FEATURES\n",
    "RICH_FEATURES = EVIDENCE_FEATURES + MASK_GEOMETRY_FEATURES\n",
)
PARENT_EVIDENCE_RANKER = PARENT_EVIDENCE_RANKER.replace(
    "    'logistic_base': oof_scores(candidate_df, BASE_FEATURES),\n    'logistic_mask_geometry': oof_scores(candidate_df, RICH_FEATURES),\n",
    "    'logistic_base': oof_scores(candidate_df, BASE_FEATURES),\n    'logistic_parent_evidence': oof_scores(candidate_df, EVIDENCE_FEATURES),\n    'logistic_parent_mask_geometry': oof_scores(candidate_df, RICH_FEATURES),\n",
)


OUTPUTS = source.OUTPUTS.replace(
    "Path('/kaggle/working/exp033_parent_blind_division')",
    "Path('/kaggle/working/exp034_parent_evidence')",
).replace(
    "    'features': {'base': BASE_FEATURES, 'mask_geometry': MASK_GEOMETRY_FEATURES},\n",
    "    'features': {'base': BASE_FEATURES, 'parent_evidence': EVIDENCE_FEATURES, 'parent_mask_geometry': RICH_FEATURES},\n",
).replace(
    "    'experiment': 'EXP033',",
    "    'experiment': 'EXP034',",
).replace(
    "# EXP033 parent-blind division discovery",
    "# EXP034 parent evidence division discovery",
).replace(
    "The correct parent is not supplied to candidate generation.",
    "The correct parent is not supplied to candidate generation. Parent evidence is computed from observations in adjacent frames and the current frame.",
)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            cell(EMBEDDED_MODULES(), "code", "embedded-modules"),
            cell(SETUP, "code", "setup-and-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-inference"),
            cell(FOCUS, "code", "focus3d-inference"),
            cell(PARENT_EVIDENCE_CANDIDATES, "code", "parent-evidence-candidates"),
            cell(PARENT_EVIDENCE_RANKER, "code", "parent-evidence-ranker"),
            cell(OUTPUTS, "code", "outputs"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP034 parent evidence division ranker"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
