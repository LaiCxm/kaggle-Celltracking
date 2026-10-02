"""Unit tests for the EXP027 structural division selector."""

from __future__ import annotations

import unittest

from tools.focus_first.global_selection import (
    DivisionSelectionCandidate,
    select_nonconflicting_divisions,
    selection_metrics,
)


def candidate(
    name: str,
    parent: str,
    d1: str,
    d2: str,
    cost: float,
    *,
    event: str = "e",
    focus: int = 0,
    label: int | None = None,
) -> DivisionSelectionCandidate:
    return DivisionSelectionCandidate(
        candidate_id=name,
        parent_id=parent,
        daughter1_id=d1,
        daughter2_id=d2,
        event_id=event,
        geometry_cost=cost,
        focus_only_count=focus,
        label=label,
    )


class GlobalSelectionTests(unittest.TestCase):
    def test_parent_and_daughter_conflicts_are_rejected(self) -> None:
        rows = [
            candidate("a", "p1", "d1", "d2", 1.0, label=1),
            candidate("b", "p1", "d3", "d4", 0.5, label=0),
            candidate("c", "p2", "d2", "d5", 0.1, event="f", label=0),
        ]
        selected = select_nonconflicting_divisions(rows)
        self.assertEqual([item.candidate_id for item in selected], ["c", "b"])

    def test_focus_activation_cost_can_overrule_geometry(self) -> None:
        rows = [
            candidate("focus", "p1", "d1", "d2", 1.0, focus=1),
            candidate("pil", "p1", "d3", "d4", 2.0, focus=0),
        ]
        self.assertEqual(select_nonconflicting_divisions(rows)[0].candidate_id, "focus")
        self.assertEqual(
            select_nonconflicting_divisions(rows, focus_activation_cost=2.0)[0].candidate_id,
            "pil",
        )

    def test_ties_are_deterministic_and_metrics_do_not_change_selection(self) -> None:
        rows = [
            candidate("z", "p", "d1", "d2", 1.0, label=0),
            candidate("a", "q", "d3", "d4", 1.0, event="f", label=1),
        ]
        selected = select_nonconflicting_divisions(reversed(rows))
        self.assertEqual([item.candidate_id for item in selected], ["a", "z"])
        metrics = selection_metrics(rows, selected)
        self.assertEqual(metrics["selected_positive_count"], 1)
        self.assertEqual(metrics["selected_negative_count"], 1)
        self.assertEqual(metrics["selection_precision"], 0.5)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_nonconflicting_divisions([], focus_activation_cost=-1.0)
        with self.assertRaises(ValueError):
            select_nonconflicting_divisions([], max_selected=-1)


if __name__ == "__main__":
    unittest.main()
