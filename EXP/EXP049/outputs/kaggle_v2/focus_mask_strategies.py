"""FOCUS3D mask-use strategies for the EXP019 mechanism comparison."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from focus_mask_arbitration import (
    FocusMaskProvider,
    MaskInstance,
    apply_focus_mask_arbitration,
    build_mask_frame_matches,
    continuation_mask_score,
    division_mask_score,
)


Conflict = tuple[float, int, int, int, int]  # rank, parent, daughter1, old_parent, daughter2


def _distance(
    a: dict[str, object],
    b: dict[str, object],
    fn: Callable[[dict[str, object], dict[str, object]], float] | None,
) -> float:
    if fn is not None:
        return float(fn(a, b))
    return float(math.dist([float(a[k]) for k in ("z", "y", "x")], [float(b[k]) for k in ("z", "y", "x")]))


def rank_occupied_daughter_conflicts(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    *,
    max_parent_um: float = 14.0,
    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,
) -> list[Conflict]:
    """Rank compact P->D1, Q->D conflicts without consulting labels."""
    by_source: dict[int, list[dict[str, object]]] = {}
    by_target: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(edge)
        by_target.setdefault(int(edge["target_id"]), []).append(edge)

    ranked: list[Conflict] = []
    for parent, outgoing in by_source.items():
        if len(outgoing) != 1 or parent not in nodes_by_id:
            continue
        daughter1 = int(outgoing[0]["target_id"])
        if daughter1 not in nodes_by_id:
            continue
        pn, d1n = nodes_by_id[parent], nodes_by_id[daughter1]
        if int(d1n["t"]) != int(pn["t"]) + 1:
            continue
        for daughter2, incoming in by_target.items():
            if daughter2 == daughter1 or len(incoming) != 1 or daughter2 not in nodes_by_id:
                continue
            old_parent = int(incoming[0]["source_id"])
            if old_parent == parent or old_parent not in nodes_by_id:
                continue
            d2n, qn = nodes_by_id[daughter2], nodes_by_id[old_parent]
            if int(d2n["t"]) != int(d1n["t"]) or int(qn["t"]) != int(pn["t"]):
                continue
            parent_distance = _distance(pn, d2n, edge_distance_um)
            sister_distance = _distance(d1n, d2n, edge_distance_um)
            if parent_distance <= max_parent_um and sister_distance <= max_parent_um:
                ranked.append((parent_distance + 0.35 * sister_distance, parent, daughter1, old_parent, daughter2))
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return ranked


def _select_with_frame_budget(
    ranked: list[Conflict],
    nodes_by_id: dict[int, dict[str, object]],
    *,
    max_items: int,
    max_frames: int,
) -> tuple[list[Conflict], set[int]]:
    selected: list[Conflict] = []
    frames: set[int] = set()
    for item in ranked:
        if len(selected) >= max(0, int(max_items)):
            break
        _, parent, _, _, daughter2 = item
        needed = {int(nodes_by_id[parent]["t"]), int(nodes_by_id[daughter2]["t"])}
        if max_frames > 0 and len(frames | needed) > int(max_frames):
            continue
        selected.append(item)
        frames.update(needed)
    return selected, frames


def _load_matches(
    nodes_by_id: dict[int, dict[str, object]],
    *,
    dataset: str,
    frames: set[int],
    provider: FocusMaskProvider,
    frame_loader: Callable[[int], np.ndarray],
    scale_um: tuple[float, float, float],
    radius_um: float,
) -> tuple[dict[int, dict[int, MaskInstance]], int]:
    maps: dict[int, np.ndarray] = {}
    for frame in sorted(frames):
        instance_map = provider.get(dataset, frame, frame_loader)
        if instance_map is not None:
            maps[frame] = instance_map
    if not maps:
        return {}, 0
    return build_mask_frame_matches(nodes_by_id, maps, scale_um, radius_um), len(maps)


def apply_selected_rescue(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    *,
    dataset: str,
    provider: FocusMaskProvider | None,
    frame_loader: Callable[[int], np.ndarray],
    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    radius_um: float = 7.0,
    max_parent_um: float = 14.0,
    min_division_score: float = 0.20,
    min_margin: float = 0.10,
    max_conflicts: int = 4,
    max_frames: int = 8,
    stats: dict[str, int] | None = None,
    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,
) -> list[dict[str, object]]:
    """Evaluate and modify only the explicitly selected conflict tuples."""
    if provider is None or not provider.available:
        return edges
    ranked = rank_occupied_daughter_conflicts(
        nodes_by_id, edges, max_parent_um=max_parent_um, edge_distance_um=edge_distance_um
    )
    selected, frames = _select_with_frame_budget(
        ranked, nodes_by_id, max_items=max_conflicts, max_frames=max_frames
    )
    if stats is not None:
        stats["focus_ranked_conflicts"] = stats.get("focus_ranked_conflicts", 0) + len(ranked)
        stats["focus_selected_conflicts"] = stats.get("focus_selected_conflicts", 0) + len(selected)
    if not selected:
        return edges
    matches, loaded = _load_matches(
        nodes_by_id, dataset=dataset, frames=frames, provider=provider,
        frame_loader=frame_loader, scale_um=scale_um, radius_um=radius_um,
    )
    if stats is not None:
        stats["focus_frames_loaded"] = stats.get("focus_frames_loaded", 0) + loaded
        stats["focus_nodes_matched"] = stats.get("focus_nodes_matched", 0) + sum(len(v) for v in matches.values())

    out = [dict(edge) for edge in edges]
    active = {(int(edge["source_id"]), int(edge["target_id"])) for edge in out}
    used_parents: set[int] = set()
    for _, parent, daughter1, old_parent, daughter2 in selected:
        if stats is not None:
            stats["focus_checked"] = stats.get("focus_checked", 0) + 1
        if parent in used_parents or (parent, daughter1) not in active or (old_parent, daughter2) not in active:
            continue
        pn, d1n = nodes_by_id[parent], nodes_by_id[daughter1]
        qn, d2n = nodes_by_id[old_parent], nodes_by_id[daughter2]
        pm = matches.get(int(pn["t"]), {}).get(parent)
        d1m = matches.get(int(d1n["t"]), {}).get(daughter1)
        qm = matches.get(int(qn["t"]), {}).get(old_parent)
        d2m = matches.get(int(d2n["t"]), {}).get(daughter2)
        if any(mask is None for mask in (pm, d1m, qm, d2m)):
            if stats is not None:
                stats["focus_missing_mask"] = stats.get("focus_missing_mask", 0) + 1
            continue
        h0 = continuation_mask_score(qn, d2n, qm, d2m)
        h1, details = division_mask_score(pn, d1n, d2n, pm, d1m, d2m)
        if h1 < min_division_score or h1 <= h0 + min_margin:
            if stats is not None:
                stats["focus_rejected_margin"] = stats.get("focus_rejected_margin", 0) + 1
            continue
        out = [
            edge for edge in out
            if not (int(edge["source_id"]) == old_parent and int(edge["target_id"]) == daughter2)
        ]
        out.append({
            "source_id": parent,
            "target_id": daughter2,
            "edge_prob": None,
            "distance_um": _distance(pn, d2n, edge_distance_um),
            "focus_mask_division": 1,
            "focus_h0": float(h0),
            "focus_h1": float(h1),
            "focus_volume_balance": float(details["volume_balance"]),
        })
        active.remove((old_parent, daughter2))
        active.add((parent, daughter2))
        used_parents.add(parent)
        if stats is not None:
            stats["focus_accepted"] = stats.get("focus_accepted", 0) + 1
    return out


def apply_safe_division_veto(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    *,
    dataset: str,
    provider: FocusMaskProvider | None,
    frame_loader: Callable[[int], np.ndarray],
    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    radius_um: float = 7.0,
    min_score: float = 0.18,
    min_volume_balance: float = 0.25,
    max_divisions: int = 8,
    max_frames: int = 8,
    stats: dict[str, int] | None = None,
    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,
) -> list[dict[str, object]]:
    """Remove only a safe-div-added edge when two mask tests strongly disagree."""
    if provider is None or not provider.available:
        return edges
    by_source: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(edge)
    candidates: list[tuple[float, int, dict[str, object], dict[str, object]]] = []
    for parent, outgoing in by_source.items():
        if len(outgoing) != 2 or parent not in nodes_by_id:
            continue
        added = [edge for edge in outgoing if int(edge.get("safe_division", 0)) == 1]
        existing = [edge for edge in outgoing if int(edge.get("safe_division", 0)) != 1]
        if len(added) != 1 or len(existing) != 1:
            continue
        d1, d2 = int(existing[0]["target_id"]), int(added[0]["target_id"])
        if d1 not in nodes_by_id or d2 not in nodes_by_id:
            continue
        pn, d1n, d2n = nodes_by_id[parent], nodes_by_id[d1], nodes_by_id[d2]
        a = _distance(pn, d1n, edge_distance_um)
        b = _distance(pn, d2n, edge_distance_um)
        asymmetry = abs(a - b) / max((a + b) / 2.0, 1e-6)
        candidates.append((-asymmetry, parent, existing[0], added[0]))
    candidates.sort(key=lambda item: (item[0], item[1]))

    selected: list[tuple[float, int, dict[str, object], dict[str, object]]] = []
    frames: set[int] = set()
    for item in candidates:
        if len(selected) >= max(0, int(max_divisions)):
            break
        _, parent, _, added = item
        daughter2 = int(added["target_id"])
        needed = {int(nodes_by_id[parent]["t"]), int(nodes_by_id[daughter2]["t"])}
        if max_frames > 0 and len(frames | needed) > int(max_frames):
            continue
        selected.append(item)
        frames.update(needed)
    if stats is not None:
        stats["focus_veto_candidates"] = stats.get("focus_veto_candidates", 0) + len(candidates)
        stats["focus_veto_selected"] = stats.get("focus_veto_selected", 0) + len(selected)
    if not selected:
        return edges
    matches, loaded = _load_matches(
        nodes_by_id, dataset=dataset, frames=frames, provider=provider,
        frame_loader=frame_loader, scale_um=scale_um, radius_um=radius_um,
    )
    if stats is not None:
        stats["focus_frames_loaded"] = stats.get("focus_frames_loaded", 0) + loaded
        stats["focus_nodes_matched"] = stats.get("focus_nodes_matched", 0) + sum(len(v) for v in matches.values())

    remove_pairs: set[tuple[int, int]] = set()
    for _, parent, existing, added in selected:
        daughter1, daughter2 = int(existing["target_id"]), int(added["target_id"])
        pn, d1n, d2n = nodes_by_id[parent], nodes_by_id[daughter1], nodes_by_id[daughter2]
        pm = matches.get(int(pn["t"]), {}).get(parent)
        d1m = matches.get(int(d1n["t"]), {}).get(daughter1)
        d2m = matches.get(int(d2n["t"]), {}).get(daughter2)
        if stats is not None:
            stats["focus_veto_checked"] = stats.get("focus_veto_checked", 0) + 1
        if any(mask is None for mask in (pm, d1m, d2m)):
            if stats is not None:
                stats["focus_veto_missing_mask"] = stats.get("focus_veto_missing_mask", 0) + 1
            continue
        score, details = division_mask_score(pn, d1n, d2n, pm, d1m, d2m)
        # Conservative contradiction: both the combined shape score and the
        # daughter-volume balance must fail. Missing masks never cause deletion.
        if score < min_score and float(details["volume_balance"]) < min_volume_balance:
            remove_pairs.add((parent, daughter2))
            if stats is not None:
                stats["focus_veto_removed"] = stats.get("focus_veto_removed", 0) + 1
    return [
        edge for edge in edges
        if (int(edge["source_id"]), int(edge["target_id"])) not in remove_pairs
    ]


def apply_focus_strategy(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    *,
    mode: str,
    dataset: str,
    provider: FocusMaskProvider | None,
    frame_loader: Callable[[int], np.ndarray],
    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    radius_um: float = 7.0,
    max_parent_um: float = 14.0,
    min_division_score: float = 0.20,
    min_margin: float = 0.10,
    max_conflicts: int = 4,
    max_frames: int = 8,
    stats: dict[str, int] | None = None,
    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,
) -> list[dict[str, object]]:
    """Dispatch one pre-registered mask-use strategy."""
    mode = str(mode).strip().lower()
    if mode == "off":
        return edges
    if mode == "broad_rescue":
        return apply_focus_mask_arbitration(
            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,
            scale_um=scale_um, radius_um=radius_um, max_parent_um=max_parent_um,
            min_division_score=min_division_score, min_margin=min_margin,
            # Preserve the scored EXP018-v3 behaviour: its value 2 means two
            # conflict frame-pairs and resulted in at most four loaded frames.
            max_conflicts=max_conflicts, max_frames_per_dataset=2,
            stats=stats, edge_distance_um=edge_distance_um,
        )
    if mode == "selected_rescue":
        return apply_selected_rescue(
            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,
            scale_um=scale_um, radius_um=radius_um, max_parent_um=max_parent_um,
            min_division_score=min_division_score, min_margin=min_margin,
            max_conflicts=max_conflicts, max_frames=max_frames,
            stats=stats, edge_distance_um=edge_distance_um,
        )
    if mode == "veto_only":
        return apply_safe_division_veto(
            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,
            scale_um=scale_um, radius_um=radius_um, max_frames=max_frames,
            stats=stats, edge_distance_um=edge_distance_um,
        )
    if mode == "hybrid_selected":
        rescued = apply_selected_rescue(
            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,
            scale_um=scale_um, radius_um=radius_um, max_parent_um=max_parent_um,
            min_division_score=min_division_score, min_margin=min_margin,
            max_conflicts=max_conflicts, max_frames=max_frames,
            stats=stats, edge_distance_um=edge_distance_um,
        )
        return apply_safe_division_veto(
            nodes_by_id, rescued, dataset=dataset, provider=provider, frame_loader=frame_loader,
            scale_um=scale_um, radius_um=radius_um, max_frames=max_frames,
            stats=stats, edge_distance_um=edge_distance_um,
        )
    raise ValueError(f"unknown FOCUS strategy: {mode}")
