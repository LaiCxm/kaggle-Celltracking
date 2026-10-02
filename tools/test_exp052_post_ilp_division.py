"""Unit tests for EXP052's delayed rank-3 division promotion."""

from __future__ import annotations

import numpy as np

from exp052_post_ilp_division import apply_exp052_top3_division_proposals


def _node(node_id: int, t: int, x: float) -> dict[str, object]:
    return {"node_id": node_id, "t": t, "z": 0.0, "y": 0.0, "x": float(x)}


def _fixture(second_x: float = -5.0) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    nodes = {
        1: _node(1, 0, 0),
        2: _node(2, 1, 5),
        3: _node(3, 1, second_x),
        4: _node(4, 2, 8),
        5: _node(5, 2, second_x + (-3 if second_x < 0 else 3)),
        7: _node(7, 1, -5),
        8: _node(8, 2, -8),
    }
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 2, "target_id": 4},
        {"source_id": 3, "target_id": 5},
        {"source_id": 7, "target_id": 8},
    ]
    top2 = np.asarray([[1, 2, 3, 0.8, 0.30]], dtype=np.float64)
    rank3 = np.asarray([[1, 2, 7, 0.8, 0.25]], dtype=np.float64)
    return nodes, edges, top2, rank3


def _run(nodes, edges, top2, rank3):
    stats: dict[str, int] = {}
    out = apply_exp052_top3_division_proposals(
        nodes,
        edges,
        top2,
        rank3,
        cosine_max=-0.9,
        scale_um=(1, 1, 1),
        stats=stats,
    )
    return {(int(e["source_id"]), int(e["target_id"])) for e in out}, stats


def main() -> None:
    # A valid rank-2 proposal consumes the source; rank-3 is not also added.
    pairs, stats = _run(*_fixture())
    assert (1, 3) in pairs
    assert (1, 7) not in pairs
    assert stats["exp049_added"] == 1
    assert stats["exp052_rank3_added"] == 0

    # If rank-2 fails the sister-distance gate, rank-3 can recover the fork.
    nodes, edges, top2, rank3 = _fixture(second_x=3.0)
    pairs, stats = _run(nodes, edges, top2, rank3)
    assert (1, 3) not in pairs
    assert (1, 7) in pairs
    assert stats["exp049_added"] == 0
    assert stats["exp052_rank3_added"] == 1

    # A malformed proposal violates the five-column compatibility contract.
    nodes, edges, top2, rank3 = _fixture()
    try:
        _run(nodes, edges, top2, rank3[:, :4])
    except ValueError as exc:
        assert "shape (N, 5)" in str(exc)
    else:
        raise AssertionError("rank-3 shape validation did not fire")

    print("EXP052 delayed rank-3 division tests passed")


if __name__ == "__main__":
    main()
