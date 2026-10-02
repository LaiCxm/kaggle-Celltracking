from __future__ import annotations

import numpy as np

from focus_mask_arbitration import FocusMaskProvider
from focus_mask_strategies import (
    apply_safe_division_veto,
    apply_selected_rescue,
    rank_occupied_daughter_conflicts,
)


class ArrayProvider:
    def __init__(self, maps):
        self.maps = maps
        self.error = None

    @property
    def available(self):
        return True

    def get(self, dataset, t, frame_loader):
        return self.maps.get(int(t))


def _cube(arr, label, center, radius=1):
    z, y, x = center
    arr[z - radius:z + radius + 1, y - radius:y + radius + 1, x - radius:x + radius + 1] = label


def _rescue_fixture():
    nodes = {
        1: {"t": 0, "z": 5, "y": 5, "x": 5},
        2: {"t": 0, "z": 5, "y": 13, "x": 5},
        3: {"t": 1, "z": 5, "y": 6, "x": 5},
        4: {"t": 1, "z": 5, "y": 8, "x": 5},
    }
    edges = [
        {"source_id": 1, "target_id": 3, "edge_prob": 0.9},
        {"source_id": 2, "target_id": 4, "edge_prob": 0.9},
    ]
    t0 = np.zeros((12, 20, 12), dtype=np.int32)
    t1 = np.zeros_like(t0)
    _cube(t0, 1, (5, 5, 5), 1)
    t0[5, 13, 5] = 2
    _cube(t1, 1, (5, 6, 5), 1)
    _cube(t1, 2, (5, 8, 5), 1)
    return nodes, edges, ArrayProvider({0: t0, 1: t1})


def test_selected_rescue_enforces_the_conflict_budget():
    nodes, edges, provider = _rescue_fixture()
    ranked = rank_occupied_daughter_conflicts(nodes, edges, max_parent_um=20.0)
    assert ranked
    stats = {}
    out = apply_selected_rescue(
        nodes, edges, dataset="sample", provider=provider,
        frame_loader=lambda t: np.zeros((1, 1, 1)),
        scale_um=(1.0, 1.0, 1.0), radius_um=2.0,
        max_parent_um=20.0, min_division_score=0.1, min_margin=0.01,
        max_conflicts=1, max_frames=2, stats=stats,
    )
    assert stats["focus_selected_conflicts"] == 1
    assert stats["focus_checked"] == 1
    assert stats["focus_accepted"] <= 1
    assert len(out) == len(edges)


def test_focus_weight_can_promote_a_mask_supported_rewire():
    nodes, edges, provider = _rescue_fixture()
    common = dict(
        dataset="sample", provider=provider,
        frame_loader=lambda t: np.zeros((1, 1, 1)),
        scale_um=(1.0, 1.0, 1.0), radius_um=2.0,
        max_parent_um=20.0, min_division_score=0.1, min_margin=0.7,
        max_conflicts=1, max_frames=2,
    )
    conservative = apply_selected_rescue(
        nodes, edges, focus_weight=1.0, stats={}, **common,
    )
    aggressive_stats = {}
    aggressive = apply_selected_rescue(
        nodes, edges, focus_weight=1.5, stats=aggressive_stats, **common,
    )
    conservative_pairs = {(e["source_id"], e["target_id"]) for e in conservative}
    aggressive_pairs = {(e["source_id"], e["target_id"]) for e in aggressive}
    assert conservative_pairs == {(1, 3), (2, 4)}
    assert aggressive_pairs == {(1, 3), (1, 4)}
    assert aggressive_stats["focus_accepted"] == 1


def test_veto_only_removes_only_the_safe_division_edge():
    nodes, edges, provider = _rescue_fixture()
    fork = [
        {"source_id": 1, "target_id": 3, "edge_prob": 0.9},
        {"source_id": 1, "target_id": 4, "edge_prob": None, "safe_division": 1},
    ]
    stats = {}
    out = apply_safe_division_veto(
        nodes, fork, dataset="sample", provider=provider,
        frame_loader=lambda t: np.zeros((1, 1, 1)),
        scale_um=(1.0, 1.0, 1.0), radius_um=2.0,
        min_score=1.01, min_volume_balance=1.01,
        max_divisions=1, max_frames=2, stats=stats,
    )
    assert {(e["source_id"], e["target_id"]) for e in out} == {(1, 3)}
    assert stats["focus_veto_removed"] == 1


def test_veto_missing_masks_is_fail_safe():
    nodes, _, _ = _rescue_fixture()
    fork = [
        {"source_id": 1, "target_id": 3, "edge_prob": 0.9},
        {"source_id": 1, "target_id": 4, "edge_prob": None, "safe_division": 1},
    ]
    provider = ArrayProvider({})
    out = apply_safe_division_veto(
        nodes, fork, dataset="sample", provider=provider,
        frame_loader=lambda t: np.zeros((1, 1, 1)),
        max_divisions=1, max_frames=2,
    )
    assert out == fork
