"""CPU tests for the EXP056 edge-protection gate."""

from __future__ import annotations

import numpy as np

from create_exp056_edge_protected import _helper_source


def _load_helper():
    namespace: dict[str, object] = {}
    exec(compile(_helper_source(), "EXP056 embedded helper", "exec"), namespace)
    return namespace["apply_post_ilp_division_proposals"]


def _node(node_id: int, t: int, x: float) -> dict[str, object]:
    return {"node_id": node_id, "t": t, "z": 0.0, "y": 0.0, "x": x}


def _fixture(old_prob: float | None):
    nodes = {
        1: _node(1, 0, 0.0),
        2: _node(2, 1, 5.0),
        3: _node(3, 1, -5.0),
        4: _node(4, 2, 8.0),
        5: _node(5, 2, -8.0),
        6: _node(6, 0, -4.0),
    }
    old = {"source_id": 6, "target_id": 3}
    if old_prob is not None:
        old["edge_prob"] = old_prob
    edges = [
        {"source_id": 1, "target_id": 2, "edge_prob": 0.8},
        {"source_id": 2, "target_id": 4, "edge_prob": 0.8},
        {"source_id": 3, "target_id": 5, "edge_prob": 0.8},
        old,
    ]
    proposals = np.asarray([[1, 2, 3, 0.8, 0.3]], dtype=np.float64)
    return nodes, edges, proposals


def main() -> None:
    apply = _load_helper()

    nodes, edges, proposals = _fixture(old_prob=0.95)
    stats: dict[str, int] = {}
    out = apply(
        nodes, edges, proposals, cosine_max=-0.9, scale_um=(1, 1, 1),
        edge_guard=True, edge_guard_margin=0.10, stats=stats,
    )
    assert (1, 3) not in {(e["source_id"], e["target_id"]) for e in out}
    assert stats["exp056_edge_guard_rejected"] == 1
    assert stats["exp049_added"] == 0

    nodes, edges, proposals = _fixture(old_prob=0.10)
    stats = {}
    out = apply(
        nodes, edges, proposals, cosine_max=-0.9, scale_um=(1, 1, 1),
        edge_guard=True, edge_guard_margin=0.10, stats=stats,
    )
    pairs = {(e["source_id"], e["target_id"]) for e in out}
    assert (6, 3) not in pairs and (1, 3) in pairs
    assert stats["exp056_edge_guard_accepted"] == 1

    nodes, edges, proposals = _fixture(old_prob=None)
    stats = {}
    out = apply(
        nodes, edges, proposals, cosine_max=-0.9, scale_um=(1, 1, 1),
        edge_guard=True, edge_guard_margin=0.10, stats=stats,
    )
    assert (1, 3) in {(e["source_id"], e["target_id"]) for e in out}
    assert stats["exp056_edge_guard_missing_prob"] == 1
    print("EXP056 edge-protection tests passed")


if __name__ == "__main__":
    main()

