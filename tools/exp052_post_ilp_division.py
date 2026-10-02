"""EXP052 Top-3 delayed division promotion built on the EXP049 selector."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from exp049_post_ilp_division import apply_post_ilp_division_proposals


def apply_exp052_top3_division_proposals(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    proposals_top2: np.ndarray,
    proposals_rank3: np.ndarray,
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
    """Apply EXP049 rank-2 promotion, then a delayed rank-3 fallback.

    Both proposal arrays use the original EXP049 five-column contract:
    ``source, primary, backup, primary_probability, backup_probability``.
    Rank-3 rows are evaluated only for sources whose rank-2 row was not
    promoted and for frames not already consumed by a rank-2 promotion.
    """
    stats = stats if stats is not None else {}
    top2 = np.asarray(proposals_top2, dtype=np.float64)
    rank3 = np.asarray(proposals_rank3, dtype=np.float64)
    if top2.size == 0:
        top2 = np.empty((0, 5), dtype=np.float64)
    elif top2.ndim != 2 or top2.shape[1] != 5:
        raise ValueError(f"EXP052 top-2 proposals must have shape (N, 5), got {top2.shape}")
    if rank3.size == 0:
        rank3 = np.empty((0, 5), dtype=np.float64)
    elif rank3.ndim != 2 or rank3.shape[1] != 5:
        raise ValueError(f"EXP052 rank-3 proposals must have shape (N, 5), got {rank3.shape}")

    stats.setdefault("exp052_rank3_saved_proposals", 0)
    stats.setdefault("exp052_rank3_frame_blocked", 0)
    for suffix in (
        "structural_candidates",
        "deepcenter_rejected",
        "divergence_rejected",
        "added",
        "rewired",
        "cap_rejected",
    ):
        stats.setdefault(f"exp052_rank3_{suffix}", 0)

    # This call is intentionally the unchanged EXP049 selector.  It keeps the
    # Top-2 arm byte-compatible with the prior experiment.
    before_top2_added = int(stats.get("exp049_added", 0))
    edges = apply_post_ilp_division_proposals(
        nodes_by_id,
        edges,
        top2,
        cosine_max=cosine_max,
        scale_um=scale_um,
        min_primary_probability=min_primary_probability,
        min_second_probability=min_second_probability,
        max_parent_um=max_parent_um,
        min_sister_um=min_sister_um,
        max_sister_um=max_sister_um,
        max_distance_asymmetry=max_distance_asymmetry,
        min_divergence_gain_um=min_divergence_gain_um,
        max_per_frame=max_per_frame,
        max_total=max_total,
        accept_candidate=accept_candidate,
        stats=stats,
    )
    top2_added = int(stats.get("exp049_added", 0)) - before_top2_added

    if not len(rank3):
        return edges

    # EXP049 enforces one promoted fork per source and per source frame.  Keep
    # those limits across both stages and leave rank-3 as a true fallback.
    promoted_frames = {
        int(nodes_by_id[int(edge["source_id"])]["t"])
        for edge in edges
        if edge.get("exp049_post_ilp_division") == 1
        and int(edge["source_id"]) in nodes_by_id
    }
    keep_rows = []
    for row in rank3:
        source_id = int(row[0])
        source = nodes_by_id.get(source_id)
        if source is not None and int(source["t"]) in promoted_frames:
            stats["exp052_rank3_frame_blocked"] += 1
            continue
        keep_rows.append(row)
    rank3 = np.asarray(keep_rows, dtype=np.float64).reshape((-1, 5))
    stats["exp052_rank3_saved_proposals"] += int(len(rank3))
    if not len(rank3):
        return edges

    rank3_stats: dict[str, int] = {}
    edges = apply_post_ilp_division_proposals(
        nodes_by_id,
        edges,
        rank3,
        cosine_max=cosine_max,
        scale_um=scale_um,
        min_primary_probability=min_primary_probability,
        min_second_probability=min_second_probability,
        max_parent_um=max_parent_um,
        min_sister_um=min_sister_um,
        max_sister_um=max_sister_um,
        max_distance_asymmetry=max_distance_asymmetry,
        min_divergence_gain_um=min_divergence_gain_um,
        max_per_frame=max_per_frame,
        max_total=max(0, max_total - top2_added),
        accept_candidate=accept_candidate,
        stats=rank3_stats,
    )
    for old_suffix in (
        "structural_candidates",
        "deepcenter_rejected",
        "divergence_rejected",
        "added",
        "rewired",
        "cap_rejected",
    ):
        stats[f"exp052_rank3_{old_suffix}"] += int(
            rank3_stats.get(f"exp049_{old_suffix}", 0)
        )
    return edges
