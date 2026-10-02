"""Machine-readable summaries for the EXP021 candidate-layer audit."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Iterable

from .candidates import ContinuationCandidate, DivisionCandidate
from .observations import UnifiedNode


def summarize_frame(
    nodes: Iterable[UnifiedNode],
    continuations: Iterable[ContinuationCandidate],
    divisions: Iterable[DivisionCandidate],
) -> dict[str, object]:
    """Return counts split by source kind, without pretending they are CV."""

    nodes = list(nodes)
    continuations = list(continuations)
    divisions = list(divisions)
    return {
        "node_count": len(nodes),
        "node_kind_counts": dict(Counter(node.kind for node in nodes)),
        "provisional_node_count": sum(bool(node.provisional) for node in nodes),
        "continuation_count": len(continuations),
        "continuation_target_kind_counts": dict(Counter(candidate.target_kind for candidate in continuations)),
        "division_count": len(divisions),
        "division_focus_feature_count": sum(
            candidate.parent_union_overlap is not None for candidate in divisions
        ),
        "division_candidates": [asdict(candidate) for candidate in divisions],
    }

