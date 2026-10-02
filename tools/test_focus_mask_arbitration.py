import numpy as np

from focus_mask_arbitration import (
    division_mask_score,
    extract_instances,
    local_mask_rewire,
    match_nodes_to_instances,
)


def node(i, t, z, y, x):
    return {"node_id": i, "t": t, "z": z, "y": y, "x": x}


def test_extract_and_one_to_one_match():
    arr = np.zeros((4, 8, 8), np.int32)
    arr[1:3, 2:4, 2:4] = 1
    arr[1:3, 5:7, 5:7] = 2
    nodes = {10: node(10, 1, 1.5, 2.5, 2.5), 11: node(11, 1, 1.5, 5.5, 5.5)}
    inst = extract_instances(arr)
    assert set(inst) == {1, 2}
    got = match_nodes_to_instances(nodes, arr, (1.0, 1.0, 1.0), radius_um=2.0)
    assert set(got) == {10, 11}
    assert got[10].label != got[11].label


def test_shape_score_changes_when_contour_is_scrambled():
    parent_arr = np.zeros((5, 12, 12), np.int32)
    parent_arr[2, 4:8, 4:8] = 1
    daughter_arr = np.zeros_like(parent_arr)
    daughter_arr[2, 4:6, 4:6] = 1
    daughter_arr[2, 6:8, 6:8] = 2
    p = extract_instances(parent_arr)[1]
    d1, d2 = extract_instances(daughter_arr)[1], extract_instances(daughter_arr)[2]
    pn = node(1, 0, 2, 6, 6)
    n1 = node(2, 1, 2, 5, 5)
    n2 = node(3, 1, 2, 7, 7)
    good, _ = division_mask_score(pn, n1, n2, p, d1, d2)
    scrambled_arr = np.zeros_like(parent_arr)
    scrambled_arr[0, 0:2, 0:2] = 1
    scrambled_arr[4, 10:12, 10:12] = 2
    s1, s2 = extract_instances(scrambled_arr)[1], extract_instances(scrambled_arr)[2]
    bad, _ = division_mask_score(pn, n1, n2, p, s1, s2)
    assert good > bad


def test_local_rewire_is_exactly_one_delete_one_add():
    nodes = {
        1: node(1, 0, 0, 2, 2), 2: node(2, 1, 0, 1, 1),
        3: node(3, 1, 0, 3, 3), 4: node(4, 0, 0, 8, 8),
    }
    # Parent 1 already owns daughter 2; daughter 3 is incorrectly occupied by 4.
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 4, "target_id": 3},
    ]
    pmap = np.zeros((1, 10, 10), np.int32); pmap[0, 1:4, 1:4] = 1; pmap[0, 7:9, 7:9] = 2
    dmap = np.zeros_like(pmap); dmap[0, 0:2, 0:2] = 1; dmap[0, 2:4, 2:4] = 2
    masks = {0: {1: extract_instances(pmap)[1], 4: extract_instances(pmap)[2]}, 1: {
        2: extract_instances(dmap)[1], 3: extract_instances(dmap)[2],
    },}
    out, stats = local_mask_rewire(nodes, edges, masks, max_parent_um=10, min_division_score=0.0, min_margin=-1.0)
    pairs = {(int(e["source_id"]), int(e["target_id"])) for e in out}
    assert (4, 3) not in pairs and (1, 3) in pairs
    assert len(out) == len(edges)
    assert stats["focus_accepted"] == 1
