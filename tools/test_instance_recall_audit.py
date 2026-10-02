from __future__ import annotations

import math

from instance_recall_audit import (
    bootstrap_mean_difference,
    match_frame_instances,
    paired_hit_categories,
)


def test_gated_assignment_maximizes_match_count_before_distance():
    # Raw minimum-distance assignment picks the zero-distance diagonal first and
    # leaves the second diagonal outside the gate. The gated optimum has 2 hits.
    matches = match_frame_instances(
        [("g0", (0, 0, 0)), ("g1", (0, 3, 0))],
        [("p0", (0, 0, 0)), ("p1", (0, 0, 3))],
        scale_zyx_um=(1, 1, 1),
        radius_um=3.1,
    )
    assert len(matches) == 2
    assert {(m.gt_id, m.pred_id) for m in matches} == {("g0", "p1"), ("g1", "p0")}


def test_physical_scale_controls_gate():
    near = match_frame_instances(
        [(1, (0, 0, 0))], [(2, (1, 0, 0))],
        scale_zyx_um=(1.625, 0.40625, 0.40625), radius_um=2.0,
    )
    far = match_frame_instances(
        [(1, (0, 0, 0))], [(2, (2, 0, 0))],
        scale_zyx_um=(1.625, 0.40625, 0.40625), radius_um=2.0,
    )
    assert len(near) == 1 and math.isclose(near[0].distance_um, 1.625)
    assert far == []


def test_paired_categories_partition_gt():
    result = paired_hit_categories(range(6), [0, 1, 2, 3], [0, 1, 4])
    assert result == {
        "both_hit": 2,
        "pilkwang_only_hit": 2,
        "focus3d_only_hit": 1,
        "both_miss": 1,
    }
    assert sum(result.values()) == 6


def test_bootstrap_is_deterministic_and_reports_effect():
    result = bootstrap_mean_difference([0.1, 0.2, 0.3], iterations=1000, seed=7)
    assert math.isclose(result["mean"], 0.2)
    assert result["ci95_low"] > 0
    assert result["bootstrap_probability_gt_zero"] == 1.0


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
