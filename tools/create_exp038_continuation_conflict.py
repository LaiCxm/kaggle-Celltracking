"""Create EXP038: continuation occupancy and parent-competition evidence."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp037_soft_parent_evidence as source
except ImportError:  # pragma: no cover
    import create_exp037_soft_parent_evidence as source  # type: ignore

ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP038" / "CELL_train_continuation_conflict_evidence.ipynb"


def continuation_candidates() -> str:
    text = source.PARENT_CANDIDATES
    needle = "        previous_nodes = nodes_by_t.get(t - 1, [])\n        next_nodes = nodes_by_t.get(t + 1, [])\n"
    if needle not in text:
        raise RuntimeError("EXP037 candidate cell changed: frame-node anchor missing")
    text = text.replace(needle, needle + "        future_nodes = nodes_by_t.get(t + 2, [])\n", 1)

    needle = "        objects = enumerate_divisions(\n"
    insertion = "\n".join([
        "        incoming_evidence = {}",
        "        for daughter in daughter_nodes:",
        "            ranked = sorted(",
        "                (distance_um(parent, daughter), str(parent.proposal_id))",
        "                for parent in parent_nodes",
        "            )",
        "            for rank, (candidate_distance, parent_id) in enumerate(ranked):",
        "                alternative = min(",
        "                    (distance for distance, other_id in ranked if other_id != parent_id),",
        "                    default=float(PARENT_RADIUS_UM),",
        "                )",
        "                incoming_evidence[(parent_id, str(daughter.proposal_id))] = {",
        "                    'incoming_rank': int(rank),",
        "                    'incoming_margin_um': float(alternative - candidate_distance),",
        "                }",
        "",
        "        daughter_future_evidence = {}",
        "        for daughter in daughter_nodes:",
        "            future_distances = sorted(distance_um(daughter, target) for target in future_nodes)",
        "            daughter_future_evidence[str(daughter.proposal_id)] = {",
        "                'daughter_forward_distance_um': float(",
        "                    future_distances[0] if future_distances else PARENT_RADIUS_UM",
        "                ),",
        "                'daughter_forward_count': int(",
        "                    sum(distance <= float(PARENT_RADIUS_UM) for distance in future_distances)",
        "                ),",
        "            }",
        "",
    ])
    if needle not in text:
        raise RuntimeError("EXP037 candidate cell changed: division enumeration anchor missing")
    text = text.replace(needle, insertion + "\n" + needle, 1)

    needle = "            key = (\n                str(candidate.parent_id),\n                frozenset((str(candidate.daughter1_id), str(candidate.daughter2_id))),\n            )\n"
    insertion = "\n".join([
        "            parent_id = str(candidate.parent_id)",
        "            daughter1_id = str(candidate.daughter1_id)",
        "            daughter2_id = str(candidate.daughter2_id)",
        "            incoming1 = incoming_evidence[(parent_id, daughter1_id)]",
        "            incoming2 = incoming_evidence[(parent_id, daughter2_id)]",
        "            future1 = daughter_future_evidence[daughter1_id]",
        "            future2 = daughter_future_evidence[daughter2_id]",
        "            incoming_margins = (float(incoming1['incoming_margin_um']), float(incoming2['incoming_margin_um']))",
        "            forward_distances = (float(future1['daughter_forward_distance_um']), float(future2['daughter_forward_distance_um']))",
    ])
    if needle not in text:
        raise RuntimeError("EXP037 candidate cell changed: candidate-key anchor missing")
    text = text.replace(needle, needle + insertion + "\n", 1)

    needle = "                **parent_evidence[str(candidate.parent_id)],\n            }\n"
    insertion = "\n".join([
        "                'daughter1_incoming_rank': int(incoming1['incoming_rank']),",
        "                'daughter2_incoming_rank': int(incoming2['incoming_rank']),",
        "                'daughter1_incoming_margin_um': incoming_margins[0],",
        "                'daughter2_incoming_margin_um': incoming_margins[1],",
        "                'incoming_margin_min_um': float(min(incoming_margins)),",
        "                'incoming_margin_mean_um': float(np.mean(incoming_margins)),",
        "                'candidate_parent_closest_both': int(incoming1['incoming_rank'] == 0 and incoming2['incoming_rank'] == 0),",
        "                'daughter1_forward_distance_um': forward_distances[0],",
        "                'daughter2_forward_distance_um': forward_distances[1],",
        "                'daughter_forward_distance_mean_um': float(np.mean(forward_distances)),",
        "                'daughter_forward_distance_max_um': float(max(forward_distances)),",
        "                'daughter_forward_count': int(future1['daughter_forward_count'] + future2['daughter_forward_count']),",
    ])
    if needle not in text:
        raise RuntimeError("EXP037 candidate cell changed: row-feature anchor missing")
    return text.replace(needle, "                **parent_evidence[str(candidate.parent_id)],\n" + insertion + "\n            }\n", 1)


def continuation_ranker() -> str:
    text = source.SOFT_RANKER
    old = "PARENT_FEATURES = [\n    'parent_center_score', 'parent_is_consensus', 'parent_focus_quality',\n    'parent_backward_distance_um', 'parent_forward_count',\n]\n"
    new = "STRUCTURAL_FEATURES = [\n    'daughter1_incoming_rank', 'daughter2_incoming_rank',\n    'daughter1_incoming_margin_um', 'daughter2_incoming_margin_um',\n    'incoming_margin_min_um', 'incoming_margin_mean_um',\n    'candidate_parent_closest_both',\n    'daughter1_forward_distance_um', 'daughter2_forward_distance_um',\n    'daughter_forward_distance_mean_um', 'daughter_forward_distance_max_um',\n    'daughter_forward_count',\n]\n"
    if old not in text:
        raise RuntimeError("EXP037 ranker changed: parent feature block missing")
    text = text.replace(old, new, 1)
    start = text.index("PARENT_AGGREGATE_FEATURES = [")
    end = text.index("HARD_NEGATIVES_PER_TRANSITION = 256\n", start)
    text = text[:start] + "CONTINUATION_FEATURES = BASE_FEATURES + STRUCTURAL_FEATURES\nCONTINUATION_MASK_FEATURES = CONTINUATION_FEATURES + MASK_GEOMETRY_FEATURES\n" + text[end:]
    text = text.replace("'soft_parent': np.full(len(frame), np.nan, dtype=float),", "'continuation_conflict': np.full(len(frame), np.nan, dtype=float),", 1)
    text = text.replace("'soft_parent_mask_geometry': np.full(len(frame), np.nan, dtype=float),", "'continuation_conflict_mask_geometry': np.full(len(frame), np.nan, dtype=float),", 1)
    for line in [
        "        train_aggregates = parent_frame(frame.loc[train_pool])\n",
        "        valid_aggregates = parent_frame(valid_candidates)\n",
        "        train_aug = add_parent_aggregates(train_candidates, train_aggregates)\n",
        "        valid_aug = add_parent_aggregates(valid_candidates, valid_aggregates)\n",
    ]:
        text = text.replace(line, "", 1)
    text = text.replace("scores['soft_parent'][valid_idx]", "scores['continuation_conflict'][valid_idx]", 1)
    text = text.replace("scores['soft_parent_mask_geometry'][valid_idx]", "scores['continuation_conflict_mask_geometry'][valid_idx]", 1)
    text = text.replace("train_aug, valid_aug, SOFT_PARENT_FEATURES", "train_candidates, valid_candidates, CONTINUATION_FEATURES", 1)
    text = text.replace("train_aug, valid_aug, SOFT_PARENT_MASK_FEATURES", "train_candidates, valid_candidates, CONTINUATION_MASK_FEATURES", 1)
    return text


def outputs() -> str:
    text = source.OUTPUTS
    text = text.replace("exp037_soft_parent_evidence", "exp038_continuation_conflict_evidence")
    text = text.replace("'experiment': 'EXP037'", "'experiment': 'EXP038'")
    text = text.replace("'selection': 'soft_parent_features_full_candidate_rank'", "'selection': 'continuation_conflict_soft_rank_full_candidate_pool'")
    text = text.replace("# EXP037 soft parent evidence", "# EXP038 continuation conflict evidence")
    return text.replace(
        "Parent aggregate evidence is used as a soft candidate feature. No parent Top-K gate is applied.",
        "Continuation occupancy and parent-competition evidence are soft features; no candidates are gated out.",
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            source.cell(source.EMBEDDED_MODULES(), "code", "embedded-modules"),
            source.cell(source.SETUP, "code", "setup-and-event-sampling"),
            source.cell(source.PILKWANG, "code", "pilkwang-centers"),
            source.cell(source.FOCUS, "code", "focus3d-instances"),
            source.cell(continuation_candidates(), "code", "continuation-conflict-candidates"),
            source.cell(continuation_ranker(), "code", "continuation-conflict-ranker"),
            source.cell(outputs(), "code", "outputs"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP038 continuation conflict evidence"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
