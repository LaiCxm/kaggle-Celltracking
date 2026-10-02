"""Precision-first post-ILP division proposals for EXP049."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _exp049_position_um(node: dict[str, object], scale: np.ndarray) -> np.ndarray:
    return np.asarray([node["z"], node["y"], node["x"]], dtype=np.float64) * scale


def apply_post_ilp_division_proposals(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    proposals: np.ndarray,
    *,
    cosine_max: float,
    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    min_primary_probability: float = 0.60,
    min_second_probability: float = 0.18,
    max_parent_um: float = 7.0,
    min_sister_um: float = 8.0,
    max_sister_um: float = 14.0,
    max_distance_asymmetry: float = 0.60,
    min_divergence_gain_um: float = 2.25,
    max_per_frame: int = 1,
    max_total: int = 64,
    accept_candidate: Callable[[dict[str, object]], bool] | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    """Promote a model-ranked second edge after ILP under strict structure gates.

    ``proposals`` columns are source id, primary target id, second target id,
    primary probability, and second probability. Existing incoming edges to a
    selected second target are replaced, never duplicated.
    """
    stats = stats if stats is not None else {}
    for key in (
        "exp049_saved_proposals",
        "exp049_structural_candidates",
        "exp049_deepcenter_rejected",
        "exp049_divergence_rejected",
        "exp049_added",
        "exp049_rewired",
        "exp049_cap_rejected",
    ):
        stats.setdefault(key, 0)

    rows = np.asarray(proposals, dtype=np.float64).reshape((-1, 5))
    stats["exp049_saved_proposals"] += int(len(rows))
    if not len(rows) or not edges:
        return edges

    scale = np.asarray(scale_um, dtype=np.float64)
    outgoing: dict[int, list[dict[str, object]]] = {}
    incoming: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        outgoing.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.setdefault(int(edge["target_id"]), []).append(edge)

    candidates: list[tuple[tuple[float, float, float, int, int], int, int, int]] = []
    for source_raw, primary_raw, second_raw, p1_raw, p2_raw in rows:
        source_id, primary_id, second_id = int(source_raw), int(primary_raw), int(second_raw)
        p1, p2 = float(p1_raw), float(p2_raw)
        if p1 < min_primary_probability or p2 < min_second_probability:
            continue
        source_edges = outgoing.get(source_id, [])
        if len(source_edges) != 1:
            continue
        current_child_id = int(source_edges[0]["target_id"])
        # The model's primary edge must still be the selected continuation.
        if current_child_id != primary_id or second_id == current_child_id:
            continue
        source = nodes_by_id.get(source_id)
        current_child = nodes_by_id.get(current_child_id)
        second_child = nodes_by_id.get(second_id)
        if source is None or current_child is None or second_child is None:
            continue
        if int(current_child["t"]) != int(source["t"]) + 1:
            continue
        if int(second_child["t"]) != int(source["t"]) + 1:
            continue

        parent_pos = _exp049_position_um(source, scale)
        v1 = _exp049_position_um(current_child, scale) - parent_pos
        v2 = _exp049_position_um(second_child, scale) - parent_pos
        d1, d2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if d1 <= 1e-8 or d2 <= 1e-8 or max(d1, d2) > max_parent_um:
            continue
        sister = float(np.linalg.norm(v1 - v2))
        if sister < min_sister_um or sister > max_sister_um:
            continue
        cosine = float(np.dot(v1, v2) / (d1 * d2 + 1e-12))
        if cosine > cosine_max:
            continue
        asymmetry = abs(d1 - d2) / max((d1 + d2) / 2.0, 1e-8)
        if asymmetry > max_distance_asymmetry:
            continue

        current_next = outgoing.get(current_child_id, [])
        second_next = outgoing.get(second_id, [])
        if len(current_next) != 1 or len(second_next) != 1:
            stats["exp049_divergence_rejected"] += 1
            continue
        current_grandchild = nodes_by_id.get(int(current_next[0]["target_id"]))
        second_grandchild = nodes_by_id.get(int(second_next[0]["target_id"]))
        if current_grandchild is None or second_grandchild is None:
            stats["exp049_divergence_rejected"] += 1
            continue
        grandchild_distance = float(np.linalg.norm(
            _exp049_position_um(current_grandchild, scale)
            - _exp049_position_um(second_grandchild, scale)
        ))
        if grandchild_distance - sister < min_divergence_gain_um:
            stats["exp049_divergence_rejected"] += 1
            continue

        stats["exp049_structural_candidates"] += 1
        if accept_candidate is not None and not accept_candidate(second_child):
            stats["exp049_deepcenter_rejected"] += 1
            continue
        # More opposing and more symmetric proposals rank first. Probability is
        # only the third tie-breaker so a low-score daughter is not discarded.
        rank_key = (cosine, asymmetry, -p2, source_id, second_id)
        candidates.append((rank_key, source_id, second_id, int(source["t"])))

    candidates.sort(key=lambda item: item[0])
    selected: list[tuple[int, int]] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    per_frame: dict[int, int] = {}
    for _, source_id, second_id, frame in candidates:
        if len(selected) >= max_total:
            stats["exp049_cap_rejected"] += 1
            continue
        if source_id in used_sources or second_id in used_targets:
            continue
        if per_frame.get(frame, 0) >= max_per_frame:
            stats["exp049_cap_rejected"] += 1
            continue
        selected.append((source_id, second_id))
        used_sources.add(source_id)
        used_targets.add(second_id)
        per_frame[frame] = per_frame.get(frame, 0) + 1

    if not selected:
        return edges

    selected_targets = {target for _, target in selected}
    kept: list[dict[str, object]] = []
    for edge in edges:
        target_id = int(edge["target_id"])
        source_id = int(edge["source_id"])
        if target_id in selected_targets and (source_id, target_id) not in selected:
            stats["exp049_rewired"] += 1
            continue
        kept.append(edge)
    for source_id, second_id in selected:
        source = nodes_by_id[source_id]
        second = nodes_by_id[second_id]
        distance = float(np.linalg.norm(
            _exp049_position_um(source, scale) - _exp049_position_um(second, scale)
        ))
        kept.append({
            "source_id": source_id,
            "target_id": second_id,
            "edge_prob": None,
            "distance_um": distance,
            "exp049_post_ilp_division": 1,
        })
    stats["exp049_added"] += len(selected)
    return kept
