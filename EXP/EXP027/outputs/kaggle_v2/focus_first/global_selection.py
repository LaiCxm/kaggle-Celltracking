"""Deterministic structural selection for EXP027 division diagnostics.

This is intentionally a small, auditable set-packing selector.  It is not a
replacement for the production ILP: it only selects explicit division triples
under parent/daughter uniqueness and an optional per-event cardinality rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DivisionSelectionCandidate:
    candidate_id: str
    parent_id: str
    daughter1_id: str
    daughter2_id: str
    event_id: str | None
    geometry_cost: float
    focus_only_count: int = 0
    label: int | None = None

    def total_cost(self, focus_activation_cost: float) -> float:
        if focus_activation_cost < 0:
            raise ValueError("focus_activation_cost must be non-negative")
        return float(self.geometry_cost) + float(focus_activation_cost) * int(self.focus_only_count)


def select_nonconflicting_divisions(
    candidates: Iterable[DivisionSelectionCandidate],
    *,
    focus_activation_cost: float = 0.0,
    max_selected: int | None = None,
) -> list[DivisionSelectionCandidate]:
    """Greedily select low-cost division triples under structural constraints.

    Constraints:

    * one selected triple per parent;
    * each daughter node has at most one selected incoming division;
    * at most one triple per ``event_id`` when event ids are supplied.

    Ties are resolved by geometry cost and then candidate id, so input order
    cannot silently change the diagnostic result.  The returned list follows
    selection order and does not mutate the input candidates.
    """

    if focus_activation_cost < 0:
        raise ValueError("focus_activation_cost must be non-negative")
    if max_selected is not None and int(max_selected) < 0:
        raise ValueError("max_selected must be non-negative or None")
    items = list(candidates)
    ranked = sorted(
        items,
        key=lambda item: (
            item.total_cost(focus_activation_cost),
            float(item.geometry_cost),
            int(item.focus_only_count),
            str(item.candidate_id),
        ),
    )
    used_parents: set[str] = set()
    used_daughters: set[str] = set()
    used_events: set[str] = set()
    selected: list[DivisionSelectionCandidate] = []
    for candidate in ranked:
        if candidate.parent_id in used_parents:
            continue
        daughters = {candidate.daughter1_id, candidate.daughter2_id}
        if len(daughters) != 2 or daughters & used_daughters:
            continue
        if candidate.event_id is not None and candidate.event_id in used_events:
            continue
        selected.append(candidate)
        used_parents.add(candidate.parent_id)
        used_daughters.update(daughters)
        if candidate.event_id is not None:
            used_events.add(candidate.event_id)
        if max_selected is not None and len(selected) >= int(max_selected):
            break
    return selected


def selection_metrics(
    candidates: Iterable[DivisionSelectionCandidate],
    selected: Iterable[DivisionSelectionCandidate],
) -> dict[str, int | float]:
    """Summarize label outcomes without using labels during selection."""

    all_candidates = list(candidates)
    chosen = list(selected)
    positive_candidates = sum(int(candidate.label == 1) for candidate in all_candidates)
    selected_positive = sum(int(candidate.label == 1) for candidate in chosen)
    selected_negative = sum(int(candidate.label == 0) for candidate in chosen)
    return {
        "candidate_count": len(all_candidates),
        "positive_candidate_count": positive_candidates,
        "selected_count": len(chosen),
        "selected_positive_count": selected_positive,
        "selected_negative_count": selected_negative,
        "selection_precision": float(selected_positive / len(chosen)) if chosen else 0.0,
    }
