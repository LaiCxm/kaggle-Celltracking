"""Unit tests for the label-aware EXP021 division audit primitives."""

from tools.focus_first.division_audit import (
    candidate_pairs_for_parent,
    extract_division_events,
    has_complete_division_candidate,
    match_nodes_by_frame,
)


def test_extract_division_events_is_deterministic():
    nodes = {10: (2, 0, 0, 0), 20: (3, 0, 1, 0), 21: (3, 0, -1, 0)}
    assert extract_division_events(nodes, [(10, 21), (10, 20)]) == [
        extract_division_events(nodes, [(10, 20), (10, 21)])[0]
    ]


def test_extract_division_events_excludes_nonofficial_outdegree():
    nodes = {
        10: (2, 0, 0, 0), 20: (3, 0, 1, 0), 21: (3, 0, -1, 0),
        22: (3, 0, 0, 1),
    }
    assert extract_division_events(nodes, [(10, 20), (10, 21), (10, 22)]) == []


def test_match_nodes_is_one_to_one_and_uses_physical_scale():
    gt = {1: (0, 0, 0, 0), 2: (0, 0, 0, 10)}
    pred = [('a', (0, 0, 0, 0.1)), ('b', (0, 0, 0, 9.9)), ('c', (0, 0, 0, 0.2))]
    matched, distances = match_nodes_by_frame(
        gt, pred, scale_um=(1.0, 1.0, 1.0), radius_um=1.0,
    )
    assert matched == {1: 'a', 2: 'b'}
    assert set(distances) == {1, 2}


def test_complete_candidate_treats_daughters_as_unordered():
    assert has_complete_division_candidate([('p', 'd2', 'd1')], 'p', 'd1', 'd2')
    assert not has_complete_division_candidate([('p', 'd1', 'd1')], 'p', 'd1', 'd2')


def test_candidate_geometry_reports_missing_far_daughter():
    candidates = candidate_pairs_for_parent(
        'p', (0, 0, 0, 0),
        [('d1', (1, 0, 0, 1)), ('d2', (1, 0, 0, 20))],
        scale_um=(1.0, 1.0, 1.0), max_parent_distance_um=25.0,
        max_daughter_distance_um=14.0,
    )
    assert candidates == []
