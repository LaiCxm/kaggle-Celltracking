"""Unit tests for EXP054 joint division selection."""

from __future__ import annotations

import numpy as np

from exp054_joint_division import apply_exp054_joint_division_proposals


def _node(node_id: int, t: int, x: float) -> dict[str, object]:
    return {"node_id": node_id, "t": t, "z": 0.0, "y": 0.0, "x": float(x)}


def _fixture(source_edge: int | None = 2, existing_fork: bool = False):
    nodes = {
        1: _node(1, 0, 0.0),
        2: _node(2, 1, 4.0),
        3: _node(3, 1, -4.0),
        4: _node(4, 2, 7.0),
        5: _node(5, 2, -7.0),
        6: _node(6, 1, 0.5),
        7: _node(7, 2, 0.75),
    }
    edges = [
        {"source_id": 2, "target_id": 4},
        {"source_id": 3, "target_id": 5},
    ]
    if source_edge is not None:
        edges.insert(0, {"source_id": 1, "target_id": source_edge})
    if existing_fork:
        edges.append({"source_id": 1, "target_id": 6})
    proposals = np.asarray([[1, 2, 3, 0.8, 0.3]], dtype=np.float64)
    return nodes, edges, proposals


def _run(nodes, edges, proposals):
    stats: dict[str, int] = {}
    out = apply_exp054_joint_division_proposals(
        nodes,
        edges,
        proposals,
        cosine_max=-0.9,
        scale_um=(1.0, 1.0, 1.0),
        stats=stats,
    )
    pairs = {(int(edge["source_id"]), int(edge["target_id"])) for edge in out}
    return pairs, stats


def main() -> None:
    pairs, stats = _run(*_fixture(source_edge=2))
    assert (1, 2) in pairs and (1, 3) in pairs
    assert stats["exp054_joint_added"] == 1

    # A wrong single continuation is replaced by the atomic fork.
    pairs, stats = _run(*_fixture(source_edge=6))
    assert (1, 6) not in pairs
    assert (1, 2) in pairs and (1, 3) in pairs
    assert stats["exp054_joint_source_replaced"] == 1

    # A missing continuation can be filled, but an existing fork is protected.
    pairs, _ = _run(*_fixture(source_edge=None))
    assert (1, 2) in pairs and (1, 3) in pairs
    pairs, stats = _run(*_fixture(source_edge=2, existing_fork=True))
    assert (1, 3) not in pairs
    assert stats["exp054_joint_existing_fork_skipped"] == 1

    try:
        _run(*_fixture(source_edge=2)[:2], np.empty((0, 4), dtype=np.float64))
    except ValueError as exc:
        assert "shape (N, 5)" in str(exc)
    else:
        raise AssertionError("malformed proposal did not fail")

    print("EXP054 joint division tests passed")


if __name__ == "__main__":
    main()
