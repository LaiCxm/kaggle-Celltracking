"""Unit tests for EXP026 division candidate gate sweeps."""

from __future__ import annotations

import unittest

from tools.focus_first.candidates import enumerate_divisions
from tools.focus_first.gate_sweep import (
    enumerate_geometry_divisions,
    select_minimum_budget_full_recall,
    summarize_gate_grid,
)
from tools.focus_first.observations import CenterObservation, build_unified_nodes


def nodes(t: int, points: list[tuple[float, float, float]]):
    centers = [
        CenterObservation(node_id=index, t=t, z=z, y=y, x=x)
        for index, (z, y, x) in enumerate(points)
    ]
    return build_unified_nodes(centers, [], scale_um=(1.0, 1.0, 1.0))


class GateSweepTests(unittest.TestCase):
    def test_geometry_path_matches_full_candidate_geometry(self) -> None:
        parents = nodes(0, [(0.0, 0.0, 0.0)])
        daughters = nodes(1, [(0.0, 3.0, 0.0), (0.0, -4.0, 0.0), (0.0, 8.0, 0.0)])
        full = enumerate_divisions(
            parents,
            daughters,
            max_parent_distance_um=10.0,
            max_daughter_distance_um=10.0,
            scale_um=(1.0, 1.0, 1.0),
        )
        light = enumerate_geometry_divisions(
            parents,
            daughters,
            max_parent_distance_um=10.0,
            max_daughter_distance_um=10.0,
            scale_um=(1.0, 1.0, 1.0),
        )
        full_rows = sorted(
            (row.parent_id, frozenset((row.daughter1_id, row.daughter2_id)), row.parent_distance_um, row.daughter_distance_um)
            for row in full
        )
        light_rows = sorted(
            (row.parent_id, frozenset((row.daughter1_id, row.daughter2_id)), row.parent_distance_um, row.daughter_distance_um)
            for row in light
        )
        self.assertEqual(full_rows, light_rows)

    def test_grid_counts_zero_candidate_events_and_recalls_positive(self) -> None:
        rows = [
            {"event_id": "a", "label": 1, "parent_distance_um": 15.0, "daughter_distance_um": 13.0, "contains_focus_only": True},
            {"event_id": "a", "label": 0, "parent_distance_um": 10.0, "daughter_distance_um": 10.0, "contains_focus_only": False},
        ]
        summaries = summarize_gate_grid(
            rows,
            event_ids=["a", "b"],
            parent_radii_um=[14.0, 16.0],
            daughter_radii_um=[14.0],
        )
        self.assertEqual(summaries[0]["candidate_count"], 1)
        self.assertEqual(summaries[0]["recalled_events"], 0)
        self.assertEqual(summaries[1]["recalled_events"], 1)
        self.assertEqual(summaries[1]["total_events"], 2)

    def test_selection_uses_candidate_budget_then_registered_ties(self) -> None:
        rows = [
            {"parent_radius_um": 20.0, "daughter_radius_um": 18.0, "recalled_events": 4, "total_events": 4, "candidate_count": 80},
            {"parent_radius_um": 18.0, "daughter_radius_um": 20.0, "recalled_events": 4, "total_events": 4, "candidate_count": 70},
            {"parent_radius_um": 16.0, "daughter_radius_um": 16.0, "recalled_events": 3, "total_events": 4, "candidate_count": 40},
        ]
        selected = select_minimum_budget_full_recall(rows)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate_count"], 70)
        self.assertEqual(selected["parent_radius_um"], 18.0)

    def test_selection_returns_none_without_full_recall(self) -> None:
        self.assertIsNone(select_minimum_budget_full_recall([
            {"parent_radius_um": 32.0, "daughter_radius_um": 32.0, "recalled_events": 3, "total_events": 4, "candidate_count": 90},
        ]))


if __name__ == "__main__":
    unittest.main()
