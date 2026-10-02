from __future__ import annotations

import math


def sparse_gt_detection_audit(gt_count: int, pred_count: int, matched_pairs: int) -> dict[str, float]:
    """Audit labeled-node recall without treating sparse-GT nonmatches as false positives."""
    if matched_pairs < 0 or matched_pairs > min(gt_count, pred_count):
        raise ValueError("matched_pairs must be a valid one-to-one match count")
    tp = matched_pairs
    unmatched_predictions = pred_count - tp
    fn = gt_count - tp
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "unmatched_predictions": unmatched_predictions,
        "recall": recall,
    }


def test_perfect_one_to_one_detection():
    result = sparse_gt_detection_audit(gt_count=4, pred_count=4, matched_pairs=4)
    assert result == {"tp": 4, "fn": 0, "unmatched_predictions": 0, "recall": 1.0}


def test_extra_predictions_are_not_labeled_false_positives():
    result = sparse_gt_detection_audit(gt_count=4, pred_count=7, matched_pairs=4)
    assert result["tp"] == 4 and result["unmatched_predictions"] == 3 and result["fn"] == 0
    assert "fp" not in result and "precision" not in result and "f1" not in result
    assert result["recall"] == 1.0


def test_missing_labeled_instances_are_false_negatives():
    result = sparse_gt_detection_audit(gt_count=7, pred_count=4, matched_pairs=4)
    assert result["tp"] == 4 and result["unmatched_predictions"] == 0 and result["fn"] == 3
    assert math.isclose(result["recall"], 4 / 7)


def test_true_negative_is_not_defined_for_instance_detection():
    # Background voxels are not a finite set of competing cell instances.
    assert "tn" not in sparse_gt_detection_audit(gt_count=1, pred_count=1, matched_pairs=1)


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
