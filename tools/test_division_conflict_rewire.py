from __future__ import annotations

from tools.division_conflict_rewire import rewire_occupied_daughter_conflicts


def _node(node_id: int, t: int, y: float) -> dict[str, object]:
    return {"node_id": node_id, "t": t, "z": 0.0, "y": y, "x": 0.0}


def _edge(source_id: int, target_id: int, probability: float = 0.5) -> dict[str, object]:
    return {"source_id": source_id, "target_id": target_id, "edge_prob": probability}


def _valid_graph() -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    # parent 1 already owns daughter 2.  Daughter 3 is incorrectly occupied by
    # parent 4, and the daughter tracks clearly diverge at t+2.
    nodes = {
        10: _node(10, -1, 0.0), 11: _node(11, -1, -1.0),
        1: _node(1, 0, 0.0), 4: _node(4, 0, -2.0),
        2: _node(2, 1, 1.0), 3: _node(3, 1, -1.0),
        20: _node(20, 2, 3.0), 30: _node(30, 2, -3.0),
    }
    edges = [
        _edge(10, 1), _edge(11, 4), _edge(1, 2, 0.9), _edge(4, 3, 0.1),
        _edge(2, 20), _edge(3, 30),
    ]
    return nodes, edges


def test_occupied_daughter_is_one_delete_one_add() -> None:
    nodes, edges = _valid_graph()
    stats: dict[str, int] = {}
    result, audit = rewire_occupied_daughter_conflicts(
        nodes, edges, parent_gate_um=3.0, daughter_gate_um=3.0,
        divergence_um=1.0, stats=stats,
    )
    pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in result}
    assert (4, 3) not in pairs
    assert (1, 3) in pairs
    assert len(result) == len(edges)
    assert len(audit) == 1
    assert stats["division_conflict_accepted"] == 1


def test_daughter_gate_rejects_noncompact_fork() -> None:
    nodes, edges = _valid_graph()
    nodes[3] = _node(3, 1, -8.0)
    nodes[30] = _node(30, 2, -10.0)
    result, audit = rewire_occupied_daughter_conflicts(
        nodes, edges, parent_gate_um=10.0, daughter_gate_um=3.0,
        divergence_um=1.0,
    )
    pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in result}
    assert (4, 3) in pairs
    assert (1, 3) not in pairs
    assert audit == []


def test_frame_budget_prevents_multiple_changes() -> None:
    nodes, edges = _valid_graph()
    # Add another otherwise valid occupied daughter in the same transition.
    nodes.update({5: _node(5, 0, 2.0), 6: _node(6, 1, 2.0), 60: _node(60, 2, 5.0), 12: _node(12, -1, 2.0)})
    edges.extend([_edge(12, 5), _edge(5, 6), _edge(6, 60)])
    result, audit = rewire_occupied_daughter_conflicts(
        nodes, edges, parent_gate_um=4.0, daughter_gate_um=4.0,
        divergence_um=1.0, max_changes_per_frame=1,
    )
    assert len(audit) == 1
    assert len(result) == len(edges)


def test_relative_distance_advantage_gate_rejects_weak_rewire() -> None:
    nodes, edges = _valid_graph()
    # The proposed parent is only marginally closer than the displaced parent.
    nodes[4] = _node(4, 0, -0.5)
    result, audit = rewire_occupied_daughter_conflicts(
        nodes, edges, parent_gate_um=3.0, daughter_gate_um=3.0,
        divergence_um=1.0, min_displacement_gain_um=2.0,
    )
    pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in result}
    assert (4, 3) in pairs
    assert (1, 3) not in pairs
    assert audit == []


def test_edge_confidence_advantage_gate_rejects_weak_rewire() -> None:
    nodes, edges = _valid_graph()
    edges[2]["edge_prob"] = 0.55  # retained parent -> daughter1
    edges[3]["edge_prob"] = 0.50  # displaced parent -> daughter2
    result, audit = rewire_occupied_daughter_conflicts(
        nodes, edges, parent_gate_um=3.0, daughter_gate_um=3.0,
        divergence_um=1.0, min_edge_probability_gain=0.10,
    )
    pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in result}
    assert (4, 3) in pairs
    assert (1, 3) not in pairs
    assert audit == []


def test_strict_edge_confidence_gate_rejects_missing_probability() -> None:
    nodes, edges = _valid_graph()
    edges[2]["edge_prob"] = 0.95
    edges[3].pop("edge_prob")
    result, audit = rewire_occupied_daughter_conflicts(
        nodes, edges, parent_gate_um=3.0, daughter_gate_um=3.0,
        divergence_um=1.0, min_edge_probability_gain=0.0,
        require_edge_probability_present=True,
    )
    pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in result}
    assert (4, 3) in pairs
    assert (1, 3) not in pairs
    assert audit == []
