"""Unit tests for EXP049 post-ILP structural promotion."""

from __future__ import annotations

import numpy as np

from exp049_post_ilp_division import apply_post_ilp_division_proposals


def node(node_id, t, x):
    return {"node_id": node_id, "t": t, "z": 0.0, "y": 0.0, "x": float(x)}


def fixture(occupied=False):
    nodes = {
        1: node(1, 0, 0),
        2: node(2, 1, 5),
        3: node(3, 1, -5),
        4: node(4, 2, 8),
        5: node(5, 2, -8),
        6: node(6, 0, -4),
    }
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 2, "target_id": 4},
        {"source_id": 3, "target_id": 5},
    ]
    if occupied:
        edges.append({"source_id": 6, "target_id": 3})
    proposals = np.asarray([[1, 2, 3, 0.8, 0.3]], dtype=np.float64)
    return nodes, edges, proposals


def main():
    nodes, edges, proposals = fixture()
    stats = {}
    out = apply_post_ilp_division_proposals(
        nodes, edges, proposals, cosine_max=-0.9, scale_um=(1, 1, 1), stats=stats
    )
    assert (1, 3) in {(e["source_id"], e["target_id"]) for e in out}
    assert stats["exp049_added"] == 1

    nodes, edges, proposals = fixture(occupied=True)
    stats = {}
    out = apply_post_ilp_division_proposals(
        nodes, edges, proposals, cosine_max=-0.9, scale_um=(1, 1, 1), stats=stats
    )
    pairs = {(e["source_id"], e["target_id"]) for e in out}
    assert (6, 3) not in pairs and (1, 3) in pairs
    assert stats["exp049_rewired"] == 1

    nodes, edges, proposals = fixture()
    out = apply_post_ilp_division_proposals(
        nodes, edges, proposals, cosine_max=-1.01, scale_um=(1, 1, 1)
    )
    assert (1, 3) not in {(e["source_id"], e["target_id"]) for e in out}

    nodes, edges, proposals = fixture()
    edges = [e for e in edges if e["source_id"] != 3]
    stats = {}
    out = apply_post_ilp_division_proposals(
        nodes, edges, proposals, cosine_max=-0.9, scale_um=(1, 1, 1), stats=stats
    )
    assert (1, 3) not in {(e["source_id"], e["target_id"]) for e in out}
    assert stats["exp049_divergence_rejected"] == 1
    print("EXP049 post-ILP division tests passed")


if __name__ == "__main__":
    main()
