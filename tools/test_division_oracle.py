from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "EXP" / "EXP019" / "reference" / "tracking_cellmot_075fc5f" / "src"))

from run_exp022_division_oracle import OracleEvent, apply_oracle_events, repair_event_edges
import tracking_cellmot.division_metrics as division_metrics
from tracking_cellmot.metrics import EvaluationResult, per_sample_metrics, summarise


def test_repair_enforces_one_parent_two_daughters_and_single_incoming_edge():
    edges = [(1, 2), (10, 20), (11, 21), (12, 20), (10, 22), (20, 30), (21, 31)]
    result, removed, added = repair_event_edges(edges, parent=10, daughters=(20, 21))
    assert sorted(target for source, target in result if source == 10) == [20, 21]
    assert [source for source, target in result if target == 20] == [10]
    assert [source for source, target in result if target == 21] == [10]
    assert set(removed) == {(11, 21), (12, 20), (10, 22)}
    assert set(added) == {(10, 21)}


def test_repair_only_changes_oracle_neighbourhood():
    edges = [(1, 2), (3, 4), (10, 20), (12, 20), (10, 22), (30, 31)]
    result, removed, added = repair_event_edges(edges, parent=10, daughters=(20, 21))
    changed = set(removed) | set(added)
    assert all(source == 10 or target in {20, 21} for source, target in changed)
    assert {(1, 2), (3, 4), (30, 31)} <= set(result)


def test_daughter_order_does_not_change_result():
    edges = [(10, 22), (12, 20), (11, 21), (1, 2)]
    a = repair_event_edges(edges, 10, (20, 21))
    b = repair_event_edges(edges, 10, (21, 20))
    assert set(a[0]) == set(b[0])
    assert set(a[1]) == set(b[1])
    assert set(a[2]) == set(b[2])


def test_unmapped_event_does_not_modify_graph():
    edges = [(1, 2), (2, 3)]
    event = OracleEvent(
        stem="sample",
        gt_parent=100,
        gt_daughters=(101, 102),
        pred_parent=1,
        pred_daughters=(2, None),
        category="daughter_occupied",
        baseline_true_positive=False,
    )
    result, audit = apply_oracle_events(edges, [event])
    assert result == edges
    assert audit[0]["repairable"] == 0


def test_baseline_input_is_not_mutated():
    edges = [(1, 2), (10, 22), (12, 20)]
    snapshot = list(edges)
    event = OracleEvent(
        stem="sample",
        gt_parent=100,
        gt_daughters=(101, 102),
        pred_parent=10,
        pred_daughters=(20, 21),
        category="divergence_gate",
        baseline_true_positive=False,
    )
    apply_oracle_events(edges, [event])
    assert edges == snapshot


def test_patched_official_metric_and_weighted_summary():
    assert "_weakly_connected_components" not in inspect.getsource(division_metrics)
    a = per_sample_metrics(EvaluationResult(8, 1, 1, 1, 0, 1, 110), n_total=100, node_recall=0.8)
    b = per_sample_metrics(EvaluationResult(3, 1, 0, 0, 1, 1, 40), n_total=50, node_recall=0.6)
    result = summarise([a, b])
    adj_a = (8 / 10) * (1 - 0.1 * 0.1)
    adj_b = (3 / 4) * (1 - 0.1 * -0.2)
    expected_adj = (10 * adj_a + 4 * adj_b) / 14
    assert math.isclose(result["adj_edge_jaccard"], expected_adj, abs_tol=1e-12)
    assert math.isclose(result["division_jaccard"], 1 / 4, abs_tol=1e-12)
    assert math.isclose(result["score"], expected_adj + 0.1 / 4, abs_tol=1e-12)


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
