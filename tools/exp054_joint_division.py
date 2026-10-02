"""Joint parent-to-two-daughters selection for EXP054.

This selector is deliberately narrow.  It keeps EXP049's probability and
geometry gates, but evaluates a fork as one object and may replace a single
wrong outgoing continuation from the same parent.  It never changes an
existing fork and uses deterministic greedy arbitration for overlapping
joint candidates.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _exp054_position_um(node: dict[str, object], scale: np.ndarray) -> np.ndarray:
    return np.asarray([node["z"], node["y"], node["x"]], dtype=np.float64) * scale


def apply_exp054_joint_division_proposals(
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
    """Select structurally valid forks as atomic parent/daughter pairs.

    A source with zero or one outgoing edge can be considered.  A source with
    an existing fork is skipped to avoid rewriting an already selected branch.
    If selected, all old outgoing edges from that source and all competing
    incoming edges to either daughter are removed before the two joint edges
    are written.  This is the only intentional difference from EXP049.
    """
    stats = stats if stats is not None else {}
    keys = (
        "exp054_joint_saved_proposals",
        "exp054_joint_structural_candidates",
        "exp054_joint_deepcenter_rejected",
        "exp054_joint_divergence_rejected",
        "exp054_joint_existing_fork_skipped",
        "exp054_joint_added",
        "exp054_joint_rewired",
        "exp054_joint_source_replaced",
        "exp054_joint_cap_rejected",
    )
    for key in keys:
        stats.setdefault(key, 0)

    rows = np.asarray(proposals, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError(f"EXP054 proposals must have shape (N, 5), got {rows.shape}")
    stats["exp054_joint_saved_proposals"] += int(len(rows))
    if not len(rows):
        return edges

    scale = np.asarray(scale_um, dtype=np.float64)
    outgoing: dict[int, list[dict[str, object]]] = {}
    incoming: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        outgoing.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.setdefault(int(edge["target_id"]), []).append(edge)

    candidates: list[dict[str, object]] = []
    for source_raw, primary_raw, second_raw, p1_raw, p2_raw in rows:
        source_id = int(source_raw)
        primary_id = int(primary_raw)
        second_id = int(second_raw)
        p1 = float(p1_raw)
        p2 = float(p2_raw)
        if p1 < min_primary_probability or p2 < min_second_probability:
            continue
        if primary_id == second_id or source_id in (primary_id, second_id):
            continue

        source = nodes_by_id.get(source_id)
        primary = nodes_by_id.get(primary_id)
        second = nodes_by_id.get(second_id)
        if source is None or primary is None or second is None:
            continue
        if int(primary["t"]) != int(source["t"]) + 1:
            continue
        if int(second["t"]) != int(source["t"]) + 1:
            continue

        source_edges = outgoing.get(source_id, [])
        if len(source_edges) > 1:
            stats["exp054_joint_existing_fork_skipped"] += 1
            continue

        parent_pos = _exp054_position_um(source, scale)
        v1 = _exp054_position_um(primary, scale) - parent_pos
        v2 = _exp054_position_um(second, scale) - parent_pos
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))
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

        primary_next = outgoing.get(primary_id, [])
        second_next = outgoing.get(second_id, [])
        if len(primary_next) != 1 or len(second_next) != 1:
            stats["exp054_joint_divergence_rejected"] += 1
            continue
        primary_grandchild = nodes_by_id.get(int(primary_next[0]["target_id"]))
        second_grandchild = nodes_by_id.get(int(second_next[0]["target_id"]))
        if primary_grandchild is None or second_grandchild is None:
            stats["exp054_joint_divergence_rejected"] += 1
            continue
        grandchild_distance = float(np.linalg.norm(
            _exp054_position_um(primary_grandchild, scale)
            - _exp054_position_um(second_grandchild, scale)
        ))
        if grandchild_distance - sister < min_divergence_gain_um:
            stats["exp054_joint_divergence_rejected"] += 1
            continue

        stats["exp054_joint_structural_candidates"] += 1
        if accept_candidate is not None and not accept_candidate(second):
            stats["exp054_joint_deepcenter_rejected"] += 1
            continue

        # Keep EXP049's deterministic structural ordering.  The experiment
        # changes only the atomic conflict handling, not the ranking score.
        candidates.append({
            "source_id": source_id,
            "primary_id": primary_id,
            "second_id": second_id,
            "frame": int(source["t"]),
            "p1": p1,
            "p2": p2,
            "rank_key": (cosine, asymmetry, -p2, source_id, primary_id, second_id),
        })

    candidates.sort(key=lambda item: item["rank_key"])
    selected: list[dict[str, object]] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    per_frame: dict[int, int] = {}
    for candidate in candidates:
        source_id = int(candidate["source_id"])
        primary_id = int(candidate["primary_id"])
        second_id = int(candidate["second_id"])
        frame = int(candidate["frame"])
        if len(selected) >= max_total:
            stats["exp054_joint_cap_rejected"] += 1
            continue
        if source_id in used_sources or primary_id in used_targets or second_id in used_targets:
            continue
        if per_frame.get(frame, 0) >= max_per_frame:
            stats["exp054_joint_cap_rejected"] += 1
            continue
        selected.append(candidate)
        used_sources.add(source_id)
        used_targets.update((primary_id, second_id))
        per_frame[frame] = per_frame.get(frame, 0) + 1

    if not selected:
        return edges

    selected_sources = {int(item["source_id"]) for item in selected}
    selected_targets = {
        target_id
        for item in selected
        for target_id in (int(item["primary_id"]), int(item["second_id"]))
    }
    selected_pairs = {
        (int(item["source_id"]), int(item["primary_id"]))
        for item in selected
    } | {
        (int(item["source_id"]), int(item["second_id"]))
        for item in selected
    }

    kept: list[dict[str, object]] = []
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        remove = False
        if source_id in selected_sources and (source_id, target_id) not in selected_pairs:
            remove = True
            stats["exp054_joint_rewired"] += 1
            stats["exp054_joint_source_replaced"] += 1
        elif target_id in selected_targets and (source_id, target_id) not in selected_pairs:
            remove = True
            stats["exp054_joint_rewired"] += 1
        if not remove:
            kept.append(edge)

    existing_pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in kept}
    for item in selected:
        source_id = int(item["source_id"])
        source = nodes_by_id[source_id]
        for target_id, probability in (
            (int(item["primary_id"]), float(item["p1"])),
            (int(item["second_id"]), float(item["p2"])),
        ):
            pair = (source_id, target_id)
            if pair in existing_pairs:
                continue
            target = nodes_by_id[target_id]
            distance = float(np.linalg.norm(
                _exp054_position_um(source, scale)
                - _exp054_position_um(target, scale)
            ))
            kept.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": probability,
                "distance_um": distance,
                "exp054_joint_division": 1,
            })
            existing_pairs.add(pair)
            stats["exp054_joint_added"] += 1
    return kept
