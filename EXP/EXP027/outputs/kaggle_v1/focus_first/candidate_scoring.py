"""Grouped candidate-ranking utilities for FOCUS3D evidence audits."""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np


def permute_columns_within_groups(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
    group_key: str,
    seed: int,
) -> list[dict[str, object]]:
    """Permute mask-derived columns jointly within video-sized groups.

    Other fields, including centroids, distances, volumes, labels and event
    identity, remain fixed.  Columns are permuted as a block so their internal
    correlations are preserved while their association with candidate truth is
    destroyed.
    """

    result = [dict(row) for row in rows]
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row[group_key]].append(index)
    rng = np.random.default_rng(seed)
    for indices in grouped.values():
        donors = np.asarray(indices, dtype=np.int64)
        rng.shuffle(donors)
        for target_index, donor_index in zip(indices, donors.tolist()):
            for column in columns:
                result[target_index][column] = rows[donor_index].get(column)
    return result


def event_ranking_metrics(
    event_ids: Iterable[Hashable],
    labels: Iterable[int],
    scores: Iterable[float],
) -> dict[str, float | int]:
    """Compute event-level rank metrics for one-positive division candidates."""

    grouped: dict[Hashable, list[tuple[int, float]]] = defaultdict(list)
    for event_id, label, score in zip(event_ids, labels, scores):
        grouped[event_id].append((int(label), float(score)))
    ranks: list[int] = []
    missing_positive = 0
    multiple_positive = 0
    for values in grouped.values():
        positives = sum(label == 1 for label, _ in values)
        if positives == 0:
            missing_positive += 1
            continue
        if positives != 1:
            multiple_positive += 1
            continue
        positive_score = next(score for label, score in values if label == 1)
        strictly_higher = sum(score > positive_score for label, score in values if label != 1)
        tied_negatives = sum(
            np.isclose(score, positive_score, rtol=0.0, atol=1e-12)
            for label, score in values if label != 1
        )
        # Average rank avoids making candidate input order an invisible tie-breaker.
        ranks.append(1.0 + strictly_higher + 0.5 * tied_negatives)
    total_rankable = len(ranks)
    return {
        "events": len(grouped),
        "rankable_events": total_rankable,
        "missing_positive_events": missing_positive,
        "multiple_positive_events": multiple_positive,
        "top1_recall": float(sum(rank <= 1 for rank in ranks) / total_rankable) if ranks else 0.0,
        "top3_recall": float(sum(rank <= 3 for rank in ranks) / total_rankable) if ranks else 0.0,
        "top5_recall": float(sum(rank <= 5 for rank in ranks) / total_rankable) if ranks else 0.0,
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
    }


def leave_one_group_out_splits(groups: Sequence[Hashable]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic movie-level out-of-fold splits."""

    groups_array = np.asarray(groups, dtype=object)
    splits = []
    for group in sorted(set(groups), key=str):
        validation = np.flatnonzero(groups_array == group)
        training = np.flatnonzero(groups_array != group)
        if not len(training) or not len(validation):
            continue
        splits.append((training, validation))
    return splits
