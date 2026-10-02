"""Frame-wise orchestration for the EXP021 V1 candidate layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .candidates import (
    ContinuationCandidate,
    DivisionCandidate,
    enumerate_continuations,
    enumerate_divisions,
)
from .observations import (
    CenterObservation,
    FocusInstance,
    UnifiedNode,
    build_unified_nodes,
    DEFAULT_SCALE_UM,
)


@dataclass(frozen=True)
class FrameCandidateLayer:
    """All source proposals and outgoing event candidates for one frame."""

    t: int
    nodes: tuple[UnifiedNode, ...]
    continuations_to_next: tuple[ContinuationCandidate, ...] = ()
    divisions_to_next: tuple[DivisionCandidate, ...] = ()


def build_frame_candidate_layers(
    centers_by_t: dict[int, Iterable[CenterObservation]],
    instances_by_t: dict[int, Iterable[FocusInstance]],
    *,
    scale_um: tuple[float, float, float] = DEFAULT_SCALE_UM,
    match_radius_um: float = 7.0,
    max_distance_um: float = 14.0,
    max_daughter_distance_um: float = 14.0,
    min_daughter_distance_um: float = 0.0,
    focus_only_quality_floor: float = 0.0,
) -> list[FrameCandidateLayer]:
    """Build candidates from every observation, independent of old graph edges."""

    frame_ids = sorted(set(int(t) for t in centers_by_t) | set(int(t) for t in instances_by_t))
    nodes_by_t: dict[int, list[UnifiedNode]] = {}
    all_instances: dict[tuple[int, int], FocusInstance] = {}
    for t in frame_ids:
        centers = list(centers_by_t.get(t, ()))
        instances = list(instances_by_t.get(t, ()))
        if any(int(center.t) != t for center in centers):
            raise ValueError(f"center observation has wrong frame for t={t}")
        if any(int(instance.t) != t for instance in instances):
            raise ValueError(f"FOCUS instance has wrong frame for t={t}")
        nodes_by_t[t] = build_unified_nodes(
            centers,
            instances,
            scale_um=scale_um,
            match_radius_um=match_radius_um,
            accept_focus_quality=focus_only_quality_floor,
        )
        for instance in instances:
            all_instances[(int(instance.t), int(instance.label))] = instance

    layers: list[FrameCandidateLayer] = []
    for index, t in enumerate(frame_ids):
        nodes = nodes_by_t[t]
        if index + 1 >= len(frame_ids) or frame_ids[index + 1] != t + 1:
            layers.append(FrameCandidateLayer(t=t, nodes=tuple(nodes)))
            continue
        next_nodes = nodes_by_t[t + 1]
        continuations = enumerate_continuations(
            nodes,
            next_nodes,
            instances_by_label=all_instances,
            max_distance_um=max_distance_um,
            scale_um=scale_um,
        )
        divisions = enumerate_divisions(
            nodes,
            next_nodes,
            instances_by_label=all_instances,
            max_parent_distance_um=max_distance_um,
            max_daughter_distance_um=max_daughter_distance_um,
            min_daughter_distance_um=min_daughter_distance_um,
            scale_um=scale_um,
        )
        layers.append(
            FrameCandidateLayer(
                t=t,
                nodes=tuple(nodes),
                continuations_to_next=tuple(continuations),
                divisions_to_next=tuple(divisions),
            )
        )
    return layers

