"""CPU tests for the first FOCUS3D-first observation/candidate layer."""

from __future__ import annotations

import numpy as np

from tools.focus_first.candidates import (
    enumerate_continuations,
    enumerate_divisions,
    parent_daughter_union_features,
)
from tools.focus_first.pipeline import build_frame_candidate_layers
from tools.focus_first.observations import (
    CenterObservation,
    build_unified_nodes,
    extract_focus_instances,
)


def _mask_map() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((8, 12, 12), dtype=np.int32)
    labels[2:4, 3:5, 3:5] = 1
    labels[2:4, 7:9, 3:5] = 2
    confidence = np.zeros_like(labels, dtype=np.float32)
    confidence[labels == 1] = 0.9
    confidence[labels == 2] = 0.8
    return labels, confidence


def test_instance_summary_keeps_confidence_and_bbox():
    labels, confidence = _mask_map()
    instances = extract_focus_instances(labels, t=1, confidence_map=confidence)
    assert [item.label for item in instances] == [1, 2]
    assert instances[0].volume == 8
    assert instances[0].bbox == (2, 3, 3, 4, 5, 5)
    assert abs(instances[0].mean_confidence - 0.9) < 1e-6
    assert instances[0].mask is not None and instances[0].mask.shape == (2, 2, 2)


def test_unified_nodes_keep_focus_only_proposals():
    labels, confidence = _mask_map()
    instances = extract_focus_instances(labels, t=1, confidence_map=confidence)
    centers = [CenterObservation(10, 1, 2.0, 3.5, 3.5, 0.95)]
    nodes = build_unified_nodes(centers, instances, match_radius_um=7.0)
    assert {node.kind for node in nodes} == {"consensus", "focus_only"}
    assert sum(node.kind == "focus_only" for node in nodes) == 1


def test_division_candidates_do_not_require_unoccupied_daughters():
    parent_labels = np.zeros((8, 12, 12), dtype=np.int32)
    parent_labels[2:4, 5:7, 5:7] = 1
    daughter_labels, confidence = _mask_map()
    parent = extract_focus_instances(parent_labels, t=0, confidence_map=confidence)
    daughters = extract_focus_instances(daughter_labels, t=1, confidence_map=confidence)
    pnodes = build_unified_nodes([CenterObservation(1, 0, 2, 5.5, 5.5, 0.9)], parent)
    dnodes = build_unified_nodes(
        [
            CenterObservation(2, 1, 2, 3.5, 3.5, 0.9),
            CenterObservation(3, 1, 2, 7.5, 3.5, 0.9),
        ],
        daughters,
    )
    all_instances = {(item.t, item.label): item for item in [*parent, *daughters]}
    triples = enumerate_divisions(pnodes, dnodes, instances_by_label=all_instances, max_parent_distance_um=14.0)
    assert len(triples) == 1
    assert triples[0].daughter1_id != triples[0].daughter2_id


def test_continuation_candidates_use_union_of_sources():
    labels, confidence = _mask_map()
    instances = extract_focus_instances(labels, t=1, confidence_map=confidence)
    source = build_unified_nodes([CenterObservation(1, 0, 2, 3.5, 3.5, 0.9)], [], match_radius_um=7.0)
    target = build_unified_nodes([], instances)
    candidates = enumerate_continuations(
        source, target, instances_by_label={(i.t, i.label): i for i in instances}
    )
    assert len(candidates) == 2
    assert {candidate.target_kind for candidate in candidates} == {"focus_only"}


def test_pipeline_uses_frame_local_focus_keys():
    labels0, confidence0 = _mask_map()
    labels1, confidence1 = _mask_map()
    i1 = extract_focus_instances(labels1, t=1, confidence_map=confidence1)
    layers = build_frame_candidate_layers(
        {0: [CenterObservation(1, 0, 2, 5.5, 5.5, 0.9)], 1: []},
        {0: [], 1: i1},
    )
    assert len(layers) == 2
    assert len(layers[0].continuations_to_next) == 2
    assert len(layers[0].divisions_to_next) == 1
    assert layers[1].continuations_to_next == ()


def test_parent_daughter_union_uses_one_parent_copy():
    parent_map = np.zeros((6, 10, 8), dtype=np.int32)
    parent_map[2:4, 2:6, 2:4] = 1
    daughter_map = np.zeros_like(parent_map)
    daughter_map[2:4, 2:4, 2:4] = 1
    daughter_map[2:4, 4:6, 2:4] = 2
    parent_instance = extract_focus_instances(parent_map, t=0)[0]
    daughter_instances = extract_focus_instances(daughter_map, t=1)
    parent_node = build_unified_nodes([], [parent_instance])[0]
    daughter_nodes = build_unified_nodes([], daughter_instances)
    features = parent_daughter_union_features(
        parent_instance,
        daughter_instances[0],
        daughter_instances[1],
        parent_node,
        daughter_nodes[0],
        daughter_nodes[1],
    )
    assert features is not None
    assert np.allclose(features, (1.0, 1.0, 1.0, 1.0))


def test_parent_daughter_union_rejects_far_shapes():
    parent_map = np.zeros((6, 20, 20), dtype=np.int32)
    parent_map[2:4, 2:6, 2:4] = 1
    daughter_map = np.zeros_like(parent_map)
    daughter_map[2:4, 13:15, 13:15] = 1
    daughter_map[2:4, 15:17, 13:15] = 2
    parent_instance = extract_focus_instances(parent_map, t=0)[0]
    daughter_instances = extract_focus_instances(daughter_map, t=1)
    # Deliberately keep node motion at zero while the masks are far apart. This
    # isolates whether the feature uses the real global mask geometry.
    parent_node = build_unified_nodes([], [parent_instance])[0]
    daughter_nodes = build_unified_nodes([], daughter_instances)
    daughter_nodes = [
        type(node)(**{**node.__dict__, "z": parent_node.z, "y": parent_node.y, "x": parent_node.x})
        for node in daughter_nodes
    ]
    features = parent_daughter_union_features(
        parent_instance,
        daughter_instances[0],
        daughter_instances[1],
        parent_node,
        daughter_nodes[0],
        daughter_nodes[1],
    )
    assert features is not None
    assert features[:3] == (0.0, 0.0, 0.0)
