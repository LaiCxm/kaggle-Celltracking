from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "audit_prethreshold_candidates", Path(__file__).with_name("audit_prethreshold_candidates.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_raw_topk_distinguishes_source_and_target_budget(tmp_path):
    coords = np.asarray([[0, 0, 0, 0], [1, 0, 1, 1], [1, 0, 2, 2], [1, 0, 3, 3]], dtype=np.float32)
    # The first pair is final, the second is rejected by source budget rank 3,
    # and the third is retained only in the raw manifest with a low score.
    raw = np.asarray([
        [0, 1, 0.90, 1, 1, 0],
        [0, 2, 0.80, 2, 1, 0],
        [0, 3, 0.30, 3, 1, 0],
    ], dtype=np.float32)
    final = np.asarray([[0, 1, 0.90, 1.0]], dtype=np.float32)
    path = tmp_path / "edge_candidates_prethreshold_single_demo.npz"
    np.savez_compressed(path, coords=coords, edges=final, raw_edges=raw)
    got_coords, final_set, records = MODULE._load_manifest(path)
    assert len(got_coords) == 4
    assert (0, 1) in final_set
    assert records[(0, 2)]["source_rank"] == 2
    assert records[(0, 3)]["probability"] == pytest.approx(0.3, abs=1e-6)


def test_audit_reports_final_and_raw_only(tmp_path):
    coords = np.asarray([[0, 0, 0, 0], [1, 0, 1, 1], [1, 0, 2, 2]], dtype=np.float32)
    raw = np.asarray([[0, 1, 0.90, 1, 1, 0], [0, 2, 0.30, 2, 2, 0]], dtype=np.float32)
    final = np.asarray([[0, 1, 0.90, 1.0]], dtype=np.float32)
    path = tmp_path / "edge_candidates_prethreshold_single_demo.npz"
    np.savez_compressed(path, coords=coords, edges=final, raw_edges=raw)
    gt_nodes = {10: (0, 0, 0, 0), 11: (1, 0, 1, 1), 12: (1, 0, 2, 2)}
    gt_edges = [(10, 11), (10, 12)]
    original = MODULE.read_gt
    MODULE.read_gt = lambda _path: (gt_nodes, gt_edges)
    try:
        rows, _ = MODULE.audit_npz(path, Path("unused.geff"), threshold=0.48)
    finally:
        MODULE.read_gt = original
    assert rows[0]["status"] == "final_candidate"
    assert rows[1]["status"] == "prethreshold_topk_only"
    assert rows[1]["threshold_pass"] == 0
