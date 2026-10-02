"""Label-aware division candidate audit primitives.

This module is deliberately independent of the competition submission graph.
It answers two separate questions for a labelled division event:

1. were the parent and both daughters detected by each observation source?
2. after those nodes were detected, did the source-independent candidate layer
   contain the complete parent/daughter triple?

The functions operate on plain mappings and small immutable records so they can
be unit-tested without a GPU or a Kaggle-only dependency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class DivisionEvent:
    """A ground-truth parent with exactly two daughter endpoints."""

    event_index: int
    parent_id: int
    daughter1_id: int
    daughter2_id: int
    parent_t: int


@dataclass(frozen=True)
class EventSourceAudit:
    """Source-specific result for one division event."""

    source: str
    parent_pred_id: str | None
    daughter1_pred_id: str | None
    daughter2_pred_id: str | None
    parent_distance_um: float | None
    daughter1_distance_um: float | None
    daughter2_distance_um: float | None
    candidate_count: int
    complete_candidate: bool

    @property
    def detected_count(self) -> int:
        return sum(value is not None for value in (
            self.parent_pred_id,
            self.daughter1_pred_id,
            self.daughter2_pred_id,
        ))


def _distance_um(
    a: Sequence[float],
    b: Sequence[float],
    scale_um: Sequence[float],
) -> float:
    delta = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    scale = np.asarray(scale_um, dtype=np.float64)
    return float(np.sqrt(np.sum((delta * scale) ** 2)))


def extract_division_events(
    nodes: Mapping[int, Sequence[float]],
    edges: Iterable[tuple[int, int]],
    *,
    require_next_frame: bool = True,
) -> list[DivisionEvent]:
    """Extract deterministic two-daughter events from a GEFF-like graph.

    ``nodes[node_id]`` must start with the integer frame ``t``.  The default
    matches the official division convention: exactly two outgoing edges and
    both daughters at ``t+1``.  Out-degree >2 is intentionally excluded rather
    than silently truncated.
    """

    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[int(source)].append(int(target))
    result: list[DivisionEvent] = []
    for index, parent_id in enumerate(sorted(outgoing), 1):
        children = sorted(set(outgoing[parent_id]))
        if len(children) != 2 or parent_id not in nodes:
            continue
        if any(child not in nodes for child in children):
            continue
        if require_next_frame and any(int(nodes[child][0]) != int(nodes[parent_id][0]) + 1 for child in children):
            continue
        result.append(DivisionEvent(
            event_index=index,
            parent_id=int(parent_id),
            daughter1_id=int(children[0]),
            daughter2_id=int(children[1]),
            parent_t=int(nodes[parent_id][0]),
        ))
    return result


def match_nodes_by_frame(
    gt_nodes: Mapping[int, Sequence[float]],
    predicted_nodes: Iterable[tuple[str, Sequence[float]]],
    *,
    scale_um: Sequence[float],
    radius_um: float,
) -> tuple[dict[int, str], dict[int, float]]:
    """Match predicted nodes to GT nodes one-to-one within each frame.

    Both coordinate records begin with ``t,z,y,x``.  The returned mapping is
    from GT node id to the prediction id and physical matching distance.
    """

    pred = [(str(pid), tuple(float(v) for v in value)) for pid, value in predicted_nodes]
    gt_by_t: dict[int, list[int]] = defaultdict(list)
    pred_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, value in gt_nodes.items():
        gt_by_t[int(value[0])].append(int(node_id))
    for index, (_, value) in enumerate(pred):
        pred_by_t[int(value[0])].append(index)

    matched: dict[int, str] = {}
    distances: dict[int, float] = {}
    for t in sorted(set(gt_by_t) | set(pred_by_t)):
        gt_ids = sorted(gt_by_t.get(t, []))
        pred_indices = pred_by_t.get(t, [])
        if not gt_ids or not pred_indices:
            continue
        cost = np.asarray([
            [_distance_um(gt_nodes[gt_id][1:], pred[index][1][1:], scale_um)
             for index in pred_indices]
            for gt_id in gt_ids
        ], dtype=np.float64)
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            distance = float(cost[int(row), int(col)])
            if distance <= float(radius_um):
                gt_id = gt_ids[int(row)]
                pred_id = pred[pred_indices[int(col)]][0]
                matched[gt_id] = pred_id
                distances[gt_id] = distance
    return matched, distances


def has_complete_division_candidate(
    candidates: Iterable[tuple[str, str, str]],
    parent_id: str | None,
    daughter1_id: str | None,
    daughter2_id: str | None,
) -> bool:
    """Check whether an unordered-daughter candidate contains all endpoints."""

    if parent_id is None or daughter1_id is None or daughter2_id is None:
        return False
    daughters = frozenset((str(daughter1_id), str(daughter2_id)))
    if len(daughters) != 2:
        return False
    return any(
        str(parent) == str(parent_id)
        and frozenset((str(child1), str(child2))) == daughters
        for parent, child1, child2 in candidates
    )


def candidate_pairs_for_parent(
    parent_id: str,
    parent_point: Sequence[float],
    daughter_nodes: Iterable[tuple[str, Sequence[float]]],
    *,
    scale_um: Sequence[float],
    max_parent_distance_um: float,
    max_daughter_distance_um: float,
    min_daughter_distance_um: float = 0.0,
) -> list[tuple[str, str, str]]:
    """Build a small geometry-only division candidate list for one parent."""

    eligible: list[tuple[str, Sequence[float]]] = []
    for node_id, point in daughter_nodes:
        if _distance_um(parent_point[1:], point[1:], scale_um) <= float(max_parent_distance_um):
            eligible.append((str(node_id), point))
    result: list[tuple[str, str, str]] = []
    for (id1, point1), (id2, point2) in combinations(eligible, 2):
        distance = _distance_um(point1[1:], point2[1:], scale_um)
        if float(min_daughter_distance_um) <= distance <= float(max_daughter_distance_um):
            result.append((str(parent_id), id1, id2))
    return result
