"""Recall-first geometry sweeps for explicit division candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

from .observations import DEFAULT_SCALE_UM, UnifiedNode


@dataclass(frozen=True)
class GeometryDivisionCandidate:
    parent_id: str
    daughter1_id: str
    daughter2_id: str
    parent_distance_um: float
    daughter_distance_um: float
    contains_focus_only: bool


def _distance_um(
    a: UnifiedNode,
    b: UnifiedNode,
    scale_um: Sequence[float],
) -> float:
    delta = np.asarray(a.point, dtype=np.float64) - np.asarray(b.point, dtype=np.float64)
    scale = np.asarray(tuple(scale_um), dtype=np.float64)
    return float(np.linalg.norm(delta * scale))


def enumerate_geometry_divisions(
    parent_nodes: Iterable[UnifiedNode],
    daughter_nodes: Iterable[UnifiedNode],
    *,
    max_parent_distance_um: float,
    max_daughter_distance_um: float,
    min_daughter_distance_um: float = 0.0,
    scale_um: Sequence[float] = DEFAULT_SCALE_UM,
) -> list[GeometryDivisionCandidate]:
    """Enumerate the same geometric triples as ``enumerate_divisions``.

    This lightweight path deliberately skips mask overlap calculations so one
    large superset can be generated once and replayed through many gate pairs.
    """

    if max_parent_distance_um <= 0 or max_daughter_distance_um <= 0:
        raise ValueError("maximum distances must be positive")
    if min_daughter_distance_um < 0:
        raise ValueError("minimum daughter distance cannot be negative")
    daughters = list(daughter_nodes)
    result: list[GeometryDivisionCandidate] = []
    for parent in parent_nodes:
        eligible = [
            daughter
            for daughter in daughters
            if int(daughter.t) == int(parent.t) + 1
            and _distance_um(parent, daughter, scale_um) <= float(max_parent_distance_um)
        ]
        for daughter1, daughter2 in combinations(eligible, 2):
            daughter_distance = _distance_um(daughter1, daughter2, scale_um)
            if not float(min_daughter_distance_um) <= daughter_distance <= float(max_daughter_distance_um):
                continue
            result.append(
                GeometryDivisionCandidate(
                    parent_id=str(parent.proposal_id),
                    daughter1_id=str(daughter1.proposal_id),
                    daughter2_id=str(daughter2.proposal_id),
                    parent_distance_um=max(
                        _distance_um(parent, daughter1, scale_um),
                        _distance_um(parent, daughter2, scale_um),
                    ),
                    daughter_distance_um=float(daughter_distance),
                    contains_focus_only=any(
                        node.kind == "focus_only" for node in (parent, daughter1, daughter2)
                    ),
                )
            )
    return result


def candidate_to_dict(candidate: GeometryDivisionCandidate) -> dict[str, object]:
    return asdict(candidate)


def summarize_gate_grid(
    rows: Iterable[Mapping[str, object]],
    *,
    event_ids: Iterable[str],
    parent_radii_um: Iterable[float],
    daughter_radii_um: Iterable[float],
) -> list[dict[str, object]]:
    """Summarize event recall and candidate burden for every gate pair."""

    records = [dict(row) for row in rows]
    events = tuple(sorted(set(map(str, event_ids))))
    if not events:
        raise ValueError("event_ids cannot be empty")
    result: list[dict[str, object]] = []
    for parent_radius in parent_radii_um:
        for daughter_radius in daughter_radii_um:
            kept = [
                row for row in records
                if float(row["parent_distance_um"]) <= float(parent_radius)
                and float(row["daughter_distance_um"]) <= float(daughter_radius)
            ]
            per_event = {event_id: 0 for event_id in events}
            recalled = set()
            focus_only_count = 0
            for row in kept:
                event_id = str(row["event_id"])
                if event_id not in per_event:
                    raise ValueError(f"candidate references unknown event {event_id}")
                per_event[event_id] += 1
                if int(row["label"]) == 1:
                    recalled.add(event_id)
                focus_only_count += int(bool(row.get("contains_focus_only", False)))
            counts = np.asarray(list(per_event.values()), dtype=np.float64)
            result.append({
                "parent_radius_um": float(parent_radius),
                "daughter_radius_um": float(daughter_radius),
                "recalled_events": int(len(recalled)),
                "total_events": int(len(events)),
                "candidate_recall": float(len(recalled) / len(events)),
                "candidate_count": int(len(kept)),
                "focus_only_candidate_count": int(focus_only_count),
                "mean_candidates_per_event": float(counts.mean()),
                "median_candidates_per_event": float(np.median(counts)),
                "p95_candidates_per_event": float(np.quantile(counts, 0.95)),
                "max_candidates_per_event": int(counts.max()),
            })
    return result


def select_minimum_budget_full_recall(
    summaries: Iterable[Mapping[str, object]],
) -> dict[str, object] | None:
    """Select the least expensive full-recall gate with fixed tie breakers."""

    eligible = [
        dict(row) for row in summaries
        if int(row["recalled_events"]) == int(row["total_events"])
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            int(row["candidate_count"]),
            max(float(row["parent_radius_um"]), float(row["daughter_radius_um"])),
            float(row["parent_radius_um"]) + float(row["daughter_radius_um"]),
            float(row["parent_radius_um"]),
            float(row["daughter_radius_um"]),
        ),
    )
