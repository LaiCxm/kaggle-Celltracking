"""High-recall continuation and division proposal generation.

The candidates are deliberately independent of the currently selected graph.
In particular, a daughter is eligible for a division triple even when another
parent currently owns it in the old ILP output.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Callable, Iterable

import numpy as np

from .observations import FocusInstance, UnifiedNode, DEFAULT_SCALE_UM


@dataclass(frozen=True)
class ContinuationCandidate:
    source_id: str
    target_id: str
    distance_um: float
    focus_overlap: float | None
    focus_volume_ratio: float | None
    source_kind: str
    target_kind: str


@dataclass(frozen=True)
class DivisionCandidate:
    parent_id: str
    daughter1_id: str
    daughter2_id: str
    parent_distance_um: float
    daughter_distance_um: float
    volume_balance: float | None
    parent_union_overlap: float | None
    parent_union_parent_coverage: float | None
    parent_union_daughter_coverage: float | None
    volume_conservation: float | None
    daughter_separation_score: float


def _distance_um(a: UnifiedNode, b: UnifiedNode, scale_um: tuple[float, float, float]) -> float:
    delta = np.asarray(a.point, dtype=np.float64) - np.asarray(b.point, dtype=np.float64)
    scale = np.asarray(scale_um, dtype=np.float64)
    return float(np.sqrt(np.sum((delta * scale) ** 2)))


def _bbox_intersection(
    a: FocusInstance,
    b: FocusInstance,
    shift_a: tuple[int, int, int] = (0, 0, 0),
) -> tuple[slice, slice, slice, slice, slice, slice] | None:
    """Return local slices for two masks after shifting a by integer voxels."""

    az0, ay0, ax0, az1, ay1, ax1 = a.bbox
    bz0, by0, bx0, bz1, by1, bx1 = b.bbox
    sz, sy, sx = shift_a
    az0, az1 = az0 + sz, az1 + sz
    ay0, ay1 = ay0 + sy, ay1 + sy
    ax0, ax1 = ax0 + sx, ax1 + sx
    lo = (max(az0, bz0), max(ay0, by0), max(ax0, bx0))
    hi = (min(az1, bz1), min(ay1, by1), min(ax1, bx1))
    if any(left >= right for left, right in zip(lo, hi)):
        return None
    a_slices = tuple(slice(left - origin, right - origin) for left, right, origin in zip(lo, hi, (az0, ay0, ax0)))
    b_slices = tuple(slice(left - origin, right - origin) for left, right, origin in zip(lo, hi, (bz0, by0, bx0)))
    return (*a_slices, *b_slices)


def mask_overlap(
    source: FocusInstance | None,
    target: FocusInstance | None,
    source_node: UnifiedNode,
    target_node: UnifiedNode,
) -> float | None:
    """Compute a translation-aware cropped-mask IoU without Python voxel sets."""

    if source is None or target is None or source.mask is None or target.mask is None:
        return None
    shift = tuple(int(round(target_node.point[i] - source_node.point[i])) for i in range(3))
    slices = _bbox_intersection(source, target, shift)
    if slices is None:
        return 0.0
    src = source.mask[slices[0], slices[1], slices[2]]
    dst = target.mask[slices[3], slices[4], slices[5]]
    intersection = int(np.count_nonzero(src & dst))
    union = int(np.count_nonzero(src | dst))
    return float(intersection / union) if union else 0.0


def _instance_for_node(
    node: UnifiedNode,
    instances_by_label: dict[object, FocusInstance],
) -> FocusInstance | None:
    """Resolve a frame-local FOCUS label without cross-frame collisions."""

    if node.focus_key is not None and node.focus_key in instances_by_label:
        return instances_by_label[node.focus_key]
    return None


def enumerate_continuations(
    source_nodes: Iterable[UnifiedNode],
    target_nodes: Iterable[UnifiedNode],
    *,
    instances_by_label: dict[object, FocusInstance] | None = None,
    max_distance_um: float = 14.0,
    scale_um: tuple[float, float, float] = DEFAULT_SCALE_UM,
) -> list[ContinuationCandidate]:
    """Enumerate high-recall t -> t+1 candidates from all node proposals."""

    instances_by_label = instances_by_label or {}
    result: list[ContinuationCandidate] = []
    for source in source_nodes:
        for target in target_nodes:
            if int(target.t) != int(source.t) + 1:
                continue
            distance = _distance_um(source, target, scale_um)
            if distance > float(max_distance_um):
                continue
            source_instance = _instance_for_node(source, instances_by_label)
            target_instance = _instance_for_node(target, instances_by_label)
            overlap = mask_overlap(source_instance, target_instance, source, target)
            volume_ratio = None
            if source_instance is not None and target_instance is not None:
                volume_ratio = float(
                    min(source_instance.volume, target_instance.volume)
                    / max(source_instance.volume, target_instance.volume, 1)
                )
            result.append(
                ContinuationCandidate(
                    source_id=source.proposal_id,
                    target_id=target.proposal_id,
                    distance_um=distance,
                    focus_overlap=overlap,
                    focus_volume_ratio=volume_ratio,
                    source_kind=source.kind,
                    target_kind=target.kind,
                )
            )
    return result


def parent_daughter_union_features(
    parent: FocusInstance,
    d1: FocusInstance,
    d2: FocusInstance,
    parent_node: UnifiedNode,
    d1_node: UnifiedNode,
    d2_node: UnifiedNode,
) -> tuple[float, float, float, float] | None:
    """Compare one motion-compensated parent mask with the daughter-mask union.

    The parent is shifted once toward the midpoint of the two daughter
    centroids.  The previous implementation shifted a separate parent copy to
    each daughter and summed the overlaps; that was not a true union comparison
    and could over-count parent support.

    Returns ``(iou, parent_coverage, daughter_coverage, volume_conservation)``.
    """

    if any(instance.mask is None for instance in (parent, d1, d2)):
        return None
    daughter_midpoint = tuple(
        0.5 * (float(d1_node.point[i]) + float(d2_node.point[i])) for i in range(3)
    )
    shift = tuple(
        int(round(daughter_midpoint[i] - float(parent_node.point[i]))) for i in range(3)
    )
    intersections = []
    for daughter in (d1, d2):
        slices = _bbox_intersection(parent, daughter, shift)
        if slices is None:
            intersections.append(0)
            continue
        translated_parent = parent.mask[slices[0], slices[1], slices[2]]
        daughter_mask = daughter.mask[slices[3], slices[4], slices[5]]
        intersections.append(int(np.count_nonzero(translated_parent & daughter_mask)))
    intersection = float(sum(intersections))
    parent_volume = max(float(parent.volume), 1.0)
    daughter_volume = max(float(d1.volume + d2.volume), 1.0)
    intersection = min(intersection, parent_volume, daughter_volume)
    union_volume = max(parent_volume + daughter_volume - intersection, 1.0)
    return (
        float(intersection / union_volume),
        float(intersection / parent_volume),
        float(intersection / daughter_volume),
        float(min(parent_volume, daughter_volume) / max(parent_volume, daughter_volume)),
    )


def _union_overlap(
    parent: FocusInstance,
    d1: FocusInstance,
    d2: FocusInstance,
    parent_node: UnifiedNode,
    d1_node: UnifiedNode,
    d2_node: UnifiedNode,
) -> float | None:
    """Backward-compatible access to the corrected parent/union IoU."""

    features = parent_daughter_union_features(
        parent, d1, d2, parent_node, d1_node, d2_node
    )
    return None if features is None else features[0]


def enumerate_divisions(
    parent_nodes: Iterable[UnifiedNode],
    daughter_nodes: Iterable[UnifiedNode],
    *,
    instances_by_label: dict[object, FocusInstance] | None = None,
    max_parent_distance_um: float = 14.0,
    max_daughter_distance_um: float = 14.0,
    min_daughter_distance_um: float = 0.0,
    scale_um: tuple[float, float, float] = DEFAULT_SCALE_UM,
) -> list[DivisionCandidate]:
    """Enumerate explicit parent/daughter pairs before any graph is selected."""

    instances_by_label = instances_by_label or {}
    result: list[DivisionCandidate] = []
    daughters = list(daughter_nodes)
    for parent in parent_nodes:
        candidates: list[UnifiedNode] = []
        for daughter in daughters:
            if int(daughter.t) != int(parent.t) + 1:
                continue
            if _distance_um(parent, daughter, scale_um) <= float(max_parent_distance_um):
                candidates.append(daughter)
        for d1, d2 in combinations(candidates, 2):
            daughter_distance = _distance_um(d1, d2, scale_um)
            if daughter_distance > float(max_daughter_distance_um):
                continue
            if daughter_distance < float(min_daughter_distance_um):
                continue
            i1 = _instance_for_node(d1, instances_by_label)
            i2 = _instance_for_node(d2, instances_by_label)
            ip = _instance_for_node(parent, instances_by_label)
            balance = None
            if i1 is not None and i2 is not None:
                balance = float(min(i1.volume, i2.volume) / max(i1.volume, i2.volume, 1))
            union_features = (
                parent_daughter_union_features(ip, i1, i2, parent, d1, d2)
                if ip and i1 and i2
                else None
            )
            if union_features is None:
                overlap = parent_coverage = daughter_coverage = conservation = None
            else:
                overlap, parent_coverage, daughter_coverage, conservation = union_features
            separation = float(min(1.0, daughter_distance / max(float(max_daughter_distance_um), 1e-6)))
            result.append(
                DivisionCandidate(
                    parent_id=parent.proposal_id,
                    daughter1_id=d1.proposal_id,
                    daughter2_id=d2.proposal_id,
                    parent_distance_um=max(_distance_um(parent, d1, scale_um), _distance_um(parent, d2, scale_um)),
                    daughter_distance_um=daughter_distance,
                    volume_balance=balance,
                    parent_union_overlap=overlap,
                    parent_union_parent_coverage=parent_coverage,
                    parent_union_daughter_coverage=daughter_coverage,
                    volume_conservation=conservation,
                    daughter_separation_score=separation,
                )
            )
    return result


def candidate_to_dict(candidate: ContinuationCandidate | DivisionCandidate) -> dict[str, object]:
    return asdict(candidate)
