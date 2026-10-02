from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORER_SRC = ROOT / "EXP" / "EXP019" / "reference" / "tracking_cellmot_075fc5f" / "src"
sys.path.insert(0, str(SCORER_SRC))

import tracking_cellmot.division_metrics as division_metrics
from tracking_cellmot.metrics import EvaluationResult, per_sample_metrics, summarise


def test_patched_division_metric_is_loaded():
    assert "_weakly_connected_components" not in inspect.getsource(division_metrics)


def test_official_weighted_adjusted_score_and_pooled_division():
    a = per_sample_metrics(
        EvaluationResult(8, 1, 1, 1, 0, 1, 110), n_total=100, node_recall=0.8
    )
    b = per_sample_metrics(
        EvaluationResult(3, 1, 0, 0, 1, 1, 40), n_total=50, node_recall=0.6
    )
    result = summarise([a, b])

    adj_a = (8 / 10) * (1 - 0.1 * 0.1)
    adj_b = (3 / 4) * (1 - 0.1 * -0.2)
    expected_adj = (10 * adj_a + 4 * adj_b) / 14
    expected_division = 1 / (1 + 1 + 2)
    assert math.isclose(result["adj_edge_jaccard"], expected_adj, abs_tol=1e-12)
    assert math.isclose(result["division_jaccard"], expected_division, abs_tol=1e-12)
    assert math.isclose(result["score"], expected_adj + 0.1 * expected_division, abs_tol=1e-12)
