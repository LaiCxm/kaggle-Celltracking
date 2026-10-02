"""Utilities for sparse-GT instance-recall audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class InstanceMatch:
    gt_id: Hashable
    pred_id: Hashable
    distance_um: float


def match_frame_instances(
    gt_items: Iterable[tuple[Hashable, Sequence[float]]],
    pred_items: Iterable[tuple[Hashable, Sequence[float]]],
    *,
    scale_zyx_um: Sequence[float],
    radius_um: float,
) -> list[InstanceMatch]:
    """Return a maximum-cardinality, minimum-distance gated matching for one frame."""
    gt_items = list(gt_items)
    pred_items = list(pred_items)
    if not gt_items or not pred_items:
        return []
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")

    gt = np.asarray([point for _, point in gt_items], dtype=np.float64)
    pred = np.asarray([point for _, point in pred_items], dtype=np.float64)
    scale = np.asarray(tuple(scale_zyx_um), dtype=np.float64)
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape[1] != 3 or pred.shape[1] != 3:
        raise ValueError("points must have shape (n, 3) in z/y/x order")
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError("scale_zyx_um must contain three positive values")

    distances = np.linalg.norm(
        (gt[:, None, :] - pred[None, :, :]) * scale[None, None, :], axis=2
    )
    # A sufficiently large infeasible cost makes Hungarian assignment maximize
    # the number of gated pairs before minimizing their total distance.
    infeasible_cost = (min(len(gt_items), len(pred_items)) + 1) * (radius_um + 1.0)
    costs = np.where(distances <= radius_um, distances, infeasible_cost)
    rows, cols = linear_sum_assignment(costs)

    matches = []
    for row, col in zip(rows, cols):
        distance = float(distances[int(row), int(col)])
        if distance <= radius_um:
            matches.append(
                InstanceMatch(
                    gt_id=gt_items[int(row)][0],
                    pred_id=pred_items[int(col)][0],
                    distance_um=distance,
                )
            )
    return matches


def paired_hit_categories(
    gt_ids: Iterable[Hashable],
    pilkwang_hits: Iterable[Hashable],
    focus3d_hits: Iterable[Hashable],
) -> dict[str, int]:
    """Summarize paired hit/miss outcomes on the same sparse GT nodes."""
    gt = set(gt_ids)
    pil = set(pilkwang_hits) & gt
    focus = set(focus3d_hits) & gt
    return {
        "both_hit": len(pil & focus),
        "pilkwang_only_hit": len(pil - focus),
        "focus3d_only_hit": len(focus - pil),
        "both_miss": len(gt - (pil | focus)),
    }


def bootstrap_mean_difference(
    differences: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 20260917,
) -> dict[str, float]:
    """Equal-unit bootstrap CI for a paired mean difference."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a non-empty finite 1-D sequence")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_gt_zero": float(np.mean(samples > 0.0)),
        "units": int(values.size),
        "iterations": int(iterations),
        "seed": int(seed),
    }
