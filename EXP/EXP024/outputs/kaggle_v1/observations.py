"""First-generation unified observation layer for the FOCUS3D-first design.

The module deliberately does not write a graph or make a submission.  It turns
independent center and instance observations into auditable node proposals.
FOCUS-only proposals stay provisional until a later temporal/global stage.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_SCALE_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class CenterObservation:
    """A center detector observation in voxel coordinates."""

    node_id: int
    t: int
    z: float
    y: float
    x: float
    score: float = 0.0
    source: str = "pilkwang"

    @property
    def point(self) -> tuple[float, float, float]:
        return (float(self.z), float(self.y), float(self.x))


@dataclass(frozen=True)
class FocusInstance:
    """Compact per-instance FOCUS3D evidence.

    ``mask`` is a cropped boolean array in local bbox coordinates.  Keeping a
    crop rather than a full volume makes later streaming/caching practical.
    ``bbox`` uses half-open voxel bounds ``(z0, y0, x0, z1, y1, x1)``.
    """

    label: int
    t: int
    centroid: tuple[float, float, float]
    volume: int
    bbox: tuple[int, int, int, int, int, int]
    mask: np.ndarray | None = None
    mean_confidence: float = 1.0
    q10_confidence: float = 1.0
    boundary_fraction: float = 0.0

    @property
    def point(self) -> tuple[float, float, float]:
        return self.centroid

    @property
    def quality(self) -> float:
        """A relative quality prior, not a calibrated probability."""

        volume_term = min(1.0, float(self.volume) / 64.0)
        boundary_term = max(0.0, 1.0 - float(self.boundary_fraction))
        return float(self.mean_confidence * (0.5 + 0.5 * volume_term) * boundary_term)


@dataclass(frozen=True)
class UnifiedNode:
    """A node proposal assembled from one or both observation sources."""

    proposal_id: str
    t: int
    z: float
    y: float
    x: float
    kind: str  # consensus, pilkwang_only, focus_only
    center_ids: tuple[int, ...] = ()
    # ``focus_label`` is diagnostic only; use ``focus_key`` for lookups because
    # labels are unique only within a single frame.
    focus_label: int | None = None
    focus_key: tuple[int, int] | None = None
    center_score: float = 0.0
    focus_quality: float = 0.0
    match_distance_um: float | None = None
    provisional: bool = True

    @property
    def point(self) -> tuple[float, float, float]:
        return (float(self.z), float(self.y), float(self.x))


def _distance_um(
    a: Iterable[float],
    b: Iterable[float],
    scale_um: tuple[float, float, float],
) -> float:
    delta = np.asarray(tuple(a), dtype=np.float64) - np.asarray(tuple(b), dtype=np.float64)
    scale = np.asarray(scale_um, dtype=np.float64)
    return float(np.sqrt(np.sum((delta * scale) ** 2)))


def _validate_confidence(confidence_map: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if confidence_map is None:
        return None
    confidence = np.asarray(confidence_map, dtype=np.float32)
    if confidence.shape != shape:
        raise ValueError(f"confidence_map shape {confidence.shape} != instance_map shape {shape}")
    return confidence


def extract_focus_instances(
    instance_map: np.ndarray,
    *,
    t: int,
    confidence_map: np.ndarray | None = None,
    min_volume: int = 1,
) -> list[FocusInstance]:
    """Extract compact instances and preserve FOCUS confidence evidence."""

    arr = np.asarray(instance_map)
    if arr.ndim != 3:
        raise ValueError(f"instance_map must be 3-D, got {arr.shape}")
    confidence = _validate_confidence(confidence_map, arr.shape)
    result: list[FocusInstance] = []
    shape = np.asarray(arr.shape, dtype=np.int64)
    for raw_label in np.unique(arr):
        label = int(raw_label)
        if label <= 0:
            continue
        coords = np.argwhere(arr == raw_label)
        volume = int(coords.shape[0])
        if volume < int(min_volume):
            continue
        mins = coords.min(axis=0).astype(int)
        maxs = (coords.max(axis=0) + 1).astype(int)
        local = coords - mins
        crop_shape = tuple((maxs - mins).tolist())
        mask = np.zeros(crop_shape, dtype=bool)
        mask[tuple(local.T)] = True
        centroid = tuple(float(v) for v in coords.mean(axis=0))
        if confidence is None:
            mean_confidence = q10_confidence = 1.0
        else:
            values = confidence[arr == raw_label].astype(np.float64)
            mean_confidence = float(np.mean(values))
            q10_confidence = float(np.quantile(values, 0.10))
        touches = np.any((coords == 0) | (coords == (shape - 1)), axis=1)
        boundary_fraction = float(np.mean(touches)) if volume else 0.0
        result.append(
            FocusInstance(
                label=label,
                t=int(t),
                centroid=centroid,
                volume=volume,
                bbox=tuple(int(v) for v in (*mins, *maxs)),
                mask=mask,
                mean_confidence=mean_confidence,
                q10_confidence=q10_confidence,
                boundary_fraction=boundary_fraction,
            )
        )
    return result


def match_centers_to_instances(
    centers: Iterable[CenterObservation],
    instances: Iterable[FocusInstance],
    *,
    scale_um: tuple[float, float, float] = DEFAULT_SCALE_UM,
    radius_um: float = 7.0,
) -> dict[int, tuple[int, float]]:
    """Return one-to-one center-id -> (instance-label, distance) matches."""

    centers = list(centers)
    instances = list(instances)
    if not centers or not instances:
        return {}
    costs = np.asarray(
        [[_distance_um(center.point, instance.point, scale_um) for instance in instances] for center in centers],
        dtype=np.float64,
    )
    rows, cols = linear_sum_assignment(costs)
    matches: dict[int, tuple[int, float]] = {}
    for row, col in zip(rows, cols):
        distance = float(costs[int(row), int(col)])
        if distance <= float(radius_um):
            matches[int(centers[int(row)].node_id)] = (int(instances[int(col)].label), distance)
    return matches


def build_unified_nodes(
    centers: Iterable[CenterObservation],
    instances: Iterable[FocusInstance],
    *,
    scale_um: tuple[float, float, float] = DEFAULT_SCALE_UM,
    match_radius_um: float = 7.0,
    accept_focus_quality: float = 0.0,
) -> list[UnifiedNode]:
    """Build consensus and provisional source-specific node proposals.

    This function does not silently discard FOCUS-only instances.  They are
    explicitly marked provisional so a later temporal/global stage can decide
    whether their node-count penalty is justified.
    """

    centers = list(centers)
    instances = list(instances)
    by_label = {instance.label: instance for instance in instances}
    matches = match_centers_to_instances(
        centers, instances, scale_um=scale_um, radius_um=match_radius_um
    )
    matched_labels = {label for label, _ in matches.values()}
    result: list[UnifiedNode] = []
    for center in centers:
        matched = matches.get(int(center.node_id))
        if matched is None:
            result.append(
                UnifiedNode(
                    proposal_id=f"p:{center.t}:{center.node_id}",
                    t=int(center.t),
                    z=float(center.z),
                    y=float(center.y),
                    x=float(center.x),
                    kind="pilkwang_only",
                    center_ids=(int(center.node_id),),
                    center_score=float(center.score),
                    provisional=False,
                )
            )
            continue
        label, distance = matched
        instance = by_label[label]
        result.append(
            UnifiedNode(
                proposal_id=f"c:{center.t}:{center.node_id}:{label}",
                t=int(center.t),
                z=float(center.z),
                y=float(center.y),
                x=float(center.x),
                kind="consensus",
                center_ids=(int(center.node_id),),
                focus_label=int(label),
                focus_key=(int(instance.t), int(label)),
                center_score=float(center.score),
                focus_quality=float(instance.quality),
                match_distance_um=float(distance),
                provisional=False,
            )
        )
    for instance in instances:
        if instance.label in matched_labels or instance.quality < float(accept_focus_quality):
            continue
        result.append(
            UnifiedNode(
                proposal_id=f"f:{instance.t}:{instance.label}",
                t=int(instance.t),
                z=float(instance.centroid[0]),
                y=float(instance.centroid[1]),
                x=float(instance.centroid[2]),
                kind="focus_only",
                focus_label=int(instance.label),
                focus_key=(int(instance.t), int(instance.label)),
                focus_quality=float(instance.quality),
                provisional=True,
            )
        )
    return result


def node_to_dict(node: UnifiedNode) -> dict[str, object]:
    """JSON-friendly representation for audit manifests."""

    return asdict(node)
