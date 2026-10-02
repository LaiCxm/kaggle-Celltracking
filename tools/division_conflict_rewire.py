"""Conservative occupied-daughter division rewiring.

The normal post-processing graph gives each source one continuation edge before
safe division repair.  A true division can therefore be missed when the second
daughter is already occupied by an incorrect continuation from another parent.
This module only permits the atomic, topology-preserving correction:

``q -> daughter2`` becomes ``parent -> daughter2``.

It deliberately does not create nodes, modify unrelated continuations, or use
the sparse ground-truth labels.  Callers receive an audit trail so a complete
official-scoring experiment can distinguish proposed, rejected, and accepted
changes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable


Node = dict[str, object]
Edge = dict[str, object]
Distance = Callable[[Node, Node], float]


def _default_distance(a: Node, b: Node) -> float:
    return math.dist(
        [float(a[key]) for key in ("z", "y", "x")],
        [float(b[key]) for key in ("z", "y", "x")],
    )


def _edge_probability(edge: Edge) -> float:
    """Return a bounded tie-breaker without requiring learned edge scores."""
    value = edge.get("edge_prob")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number)) if math.isfinite(number) else 0.0


def _has_edge_probability(edge: Edge) -> bool:
    value = edge.get("edge_prob")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _add_stat(stats: dict[str, int] | None, key: str, count: int = 1) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + int(count)


def rewire_occupied_daughter_conflicts(
    nodes_by_id: dict[int, Node],
    edges: list[Edge],
    *,
    parent_gate_um: float,
    daughter_gate_um: float,
    divergence_um: float = 2.25,
    max_changes_per_frame: int = 1,
    max_changes_per_dataset: int = 6,
    require_divergence: bool = True,
    min_displacement_gain_um: float = 0.0,
    min_edge_probability_gain: float = 0.0,
    require_edge_probability_present: bool = False,
    stats: dict[str, int] | None = None,
    edge_distance_um: Distance | None = None,
) -> tuple[list[Edge], list[dict[str, object]]]:
    """Turn compact occupied-daughter conflicts into at most one fork per parent.

    A candidate must satisfy all of the following before it can be considered:

    * ``parent -> daughter1`` and ``q -> daughter2`` are existing one-frame
      edges, with one outgoing edge for each source and one incoming edge for
      each daughter;
    * both proposed daughters are at the next time point and within the
      requested parent/daughter geometric gates;
    * the two possible parents already have predecessors, avoiding appearance
      events; and
    * if enabled, the two daughter tracks continue and diverge at the next
      frame, matching the existing safe-division safeguard.

    Accepted changes are exactly one deletion plus one addition.  Greedy
    selection is deterministic, constrained by parent, re-assigned daughter,
    displaced source, frame, and dataset budgets.
    """
    distance = edge_distance_um or _default_distance
    out = [dict(edge) for edge in edges]
    audit: list[dict[str, object]] = []
    if not nodes_by_id or not out or max_changes_per_dataset <= 0:
        return out, audit

    outgoing: dict[int, list[Edge]] = defaultdict(list)
    incoming: dict[int, list[Edge]] = defaultdict(list)
    nodes_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes_by_id.items():
        nodes_by_t[int(node["t"])].append(int(node_id))
    for ids in nodes_by_t.values():
        ids.sort()
    for edge in out:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        outgoing[source_id].append(edge)
        incoming[target_id].append(edge)

    proposals: list[dict[str, object]] = []
    for parent_id in sorted(outgoing):
        parent_edges = outgoing[parent_id]
        if len(parent_edges) != 1 or len(incoming.get(parent_id, [])) != 1:
            continue
        edge_one = parent_edges[0]
        daughter_one_id = int(edge_one["target_id"])
        parent = nodes_by_id.get(parent_id)
        daughter_one = nodes_by_id.get(daughter_one_id)
        if parent is None or daughter_one is None:
            continue
        parent_t = int(parent["t"])
        if int(daughter_one["t"]) != parent_t + 1:
            continue
        parent_to_one = float(distance(parent, daughter_one))
        if parent_to_one > float(parent_gate_um):
            continue

        for daughter_two_id in nodes_by_t.get(parent_t + 1, []):
            if daughter_two_id == daughter_one_id:
                continue
            daughter_two = nodes_by_id[daughter_two_id]
            displaced_edges = incoming.get(daughter_two_id, [])
            if len(displaced_edges) != 1:
                continue
            displaced_edge = displaced_edges[0]
            displaced_parent_id = int(displaced_edge["source_id"])
            if displaced_parent_id == parent_id:
                continue
            displaced_parent = nodes_by_id.get(displaced_parent_id)
            if displaced_parent is None or int(displaced_parent["t"]) != parent_t:
                continue
            if len(outgoing.get(displaced_parent_id, [])) != 1:
                continue
            if len(incoming.get(displaced_parent_id, [])) != 1:
                continue

            parent_to_two = float(distance(parent, daughter_two))
            sister_distance = float(distance(daughter_one, daughter_two))
            if parent_to_two > float(parent_gate_um) or sister_distance > float(daughter_gate_um):
                continue
            displaced_to_two = float(distance(displaced_parent, daughter_two))
            displacement_gain = displaced_to_two - parent_to_two
            if displacement_gain < float(min_displacement_gain_um):
                _add_stat(stats, "division_conflict_rejected_advantage")
                continue
            retained_probability = _edge_probability(edge_one)
            displaced_probability = _edge_probability(displaced_edge)
            if require_edge_probability_present and (
                not _has_edge_probability(edge_one)
                or not _has_edge_probability(displaced_edge)
            ):
                _add_stat(stats, "division_conflict_rejected_missing_edge_confidence")
                continue
            edge_probability_gain = retained_probability - displaced_probability
            if edge_probability_gain < float(min_edge_probability_gain):
                _add_stat(stats, "division_conflict_rejected_edge_confidence")
                continue

            if require_divergence:
                future_one = outgoing.get(daughter_one_id, [])
                future_two = outgoing.get(daughter_two_id, [])
                if len(future_one) != 1 or len(future_two) != 1:
                    _add_stat(stats, "division_conflict_rejected_divergence")
                    continue
                grand_one = nodes_by_id.get(int(future_one[0]["target_id"]))
                grand_two = nodes_by_id.get(int(future_two[0]["target_id"]))
                if (
                    grand_one is None
                    or grand_two is None
                    or int(grand_one["t"]) != parent_t + 2
                    or int(grand_two["t"]) != parent_t + 2
                    or float(distance(grand_one, grand_two)) - sister_distance < float(divergence_um)
                ):
                    _add_stat(stats, "division_conflict_rejected_divergence")
                    continue

            # A compact fork is preferred.  Existing learned probabilities are
            # only a weak deterministic tie-breaker: high confidence for the
            # retained first child and low confidence for the displaced edge.
            score = (
                max(parent_to_one, parent_to_two)
                + 0.20 * sister_distance
                - 0.25 * _edge_probability(edge_one)
                + 0.25 * _edge_probability(displaced_edge)
            )
            proposals.append({
                "parent_id": parent_id,
                "daughter1_id": daughter_one_id,
                "daughter2_id": daughter_two_id,
                "displaced_parent_id": displaced_parent_id,
                "displaced_edge": displaced_edge,
                "parent_t": parent_t,
                "parent_daughter1_um": parent_to_one,
                "parent_daughter2_um": parent_to_two,
                "daughter_distance_um": sister_distance,
                "displaced_parent_daughter2_um": displaced_to_two,
                "displacement_gain_um": displacement_gain,
                "retained_edge_probability": retained_probability,
                "displaced_edge_probability": displaced_probability,
                "edge_probability_gain": edge_probability_gain,
                "proposal_score": float(score),
            })
            _add_stat(stats, "division_conflict_candidates")

    proposals.sort(
        key=lambda row: (
            float(row["proposal_score"]),
            int(row["parent_id"]),
            int(row["daughter2_id"]),
            int(row["displaced_parent_id"]),
        )
    )
    active_pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in out}
    used_parents: set[int] = set()
    used_daughters: set[int] = set()
    used_displaced_parents: set[int] = set()
    per_frame: dict[int, int] = defaultdict(int)

    for proposal in proposals:
        if len(audit) >= int(max_changes_per_dataset):
            _add_stat(stats, "division_conflict_rejected_cap")
            continue
        parent_id = int(proposal["parent_id"])
        daughter_two_id = int(proposal["daughter2_id"])
        displaced_parent_id = int(proposal["displaced_parent_id"])
        parent_t = int(proposal["parent_t"])
        old_pair = (displaced_parent_id, daughter_two_id)
        new_pair = (parent_id, daughter_two_id)
        if (
            parent_id in used_parents
            or daughter_two_id in used_daughters
            or displaced_parent_id in used_displaced_parents
            or per_frame[parent_t] >= int(max_changes_per_frame)
            or old_pair not in active_pairs
            or new_pair in active_pairs
        ):
            _add_stat(stats, "division_conflict_rejected_conflict")
            continue

        out = [
            edge for edge in out
            if (int(edge["source_id"]), int(edge["target_id"])) != old_pair
        ]
        out.append({
            "source_id": parent_id,
            "target_id": daughter_two_id,
            "edge_prob": None,
            "distance_um": float(proposal["parent_daughter2_um"]),
            "division_conflict_rewire": 1,
        })
        active_pairs.discard(old_pair)
        active_pairs.add(new_pair)
        used_parents.add(parent_id)
        used_daughters.add(daughter_two_id)
        used_displaced_parents.add(displaced_parent_id)
        per_frame[parent_t] += 1
        event = {
            **{key: value for key, value in proposal.items() if key != "displaced_edge"},
            "old_source_id": displaced_parent_id,
            "old_target_id": daughter_two_id,
            "new_source_id": parent_id,
            "new_target_id": daughter_two_id,
        }
        audit.append(event)
        _add_stat(stats, "division_conflict_accepted")

    return out, audit
