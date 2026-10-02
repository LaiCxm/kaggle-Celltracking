from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "audit_preilp_candidates", Path(__file__).with_name("audit_preilp_candidates.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_both_gt_edges_are_detected_and_ranked(tmp_path):
    # Coordinates are voxel coordinates; the first column is frame.
    coords = np.asarray([
        [0, 0, 10, 10],
        [1, 0, 11, 10],
        [1, 0, 10, 12],
        [1, 0, 10, 8],
    ], dtype=np.float32)
    edges = np.asarray([
        [0, 1, 0.91, 0.0],
        [0, 2, 0.72, 0.0],
        [0, 3, 0.95, 0.0],
    ], dtype=np.float32)
    npz = tmp_path / "edge_candidates.npz"
    np.savez_compressed(npz, coords=coords, edges=edges)

    # GT uses the same voxel coordinates and two daughters at t+1.
    gt_nodes = {10: (0, 0, 10, 10), 11: (1, 0, 11, 10), 12: (1, 0, 10, 12)}
    gt_edges = [(10, 11), (10, 12)]
    # Bypass zarr I/O while testing ranking logic.
    original = MODULE.read_gt
    MODULE.read_gt = lambda _path: (gt_nodes, gt_edges)
    try:
        rows, stats = MODULE.audit_npz(npz, Path("unused.geff"))
    finally:
        MODULE.read_gt = original
    assert stats["candidate_edges"] == 3
    assert rows[0]["status"] == "both_edges_in_preilp_pool"
    assert rows[0]["daughter1_rank"] == 2
    assert rows[0]["daughter2_rank"] == 3


def test_missing_second_candidate_is_reported(tmp_path):
    coords = np.asarray([[0, 0, 0, 0], [1, 0, 1, 1], [1, 0, 10, 10]], dtype=np.float32)
    edges = np.asarray([[0, 1, 0.9, 0.0]], dtype=np.float32)
    npz = tmp_path / "edge_candidates.npz"
    np.savez_compressed(npz, coords=coords, edges=edges)
    gt_nodes = {10: (0, 0, 0, 0), 11: (1, 0, 1, 1), 12: (1, 0, 10, 10)}
    gt_edges = [(10, 11), (10, 12)]
    original = MODULE.read_gt
    MODULE.read_gt = lambda _path: (gt_nodes, gt_edges)
    try:
        rows, _ = MODULE.audit_npz(npz, Path("unused.geff"))
    finally:
        MODULE.read_gt = original
    assert rows[0]["status"] == "one_edge_in_preilp_pool"
