from __future__ import annotations

import unittest

import pandas as pd

from tools.exp028_candidate_ranking import fixed_cost, run


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for video_index in range(3):
        video = f"v{video_index}"
        for event_index in range(2):
            event = f"{video}:e{event_index}"
            for candidate_index in range(3):
                positive = candidate_index == 0
                rows.append({
                    "video": video, "event_id": event, "mode": "union",
                    "candidate_id": f"{event}:c{candidate_index}",
                    "label": int(positive),
                    "parent_id": f"{video}:p{event_index}",
                    "daughter1_id": f"{event}:d1:{candidate_index}",
                    "daughter2_id": f"{event}:d2:{candidate_index}",
                    "parent_distance_um": 5.0 if positive else 10.0 + candidate_index,
                    "daughter_distance_um": 4.0 if positive else 10.0 + candidate_index,
                    "focus_only_count": 0,
                })
    return pd.DataFrame(rows)


class Exp028Tests(unittest.TestCase):
    def test_fixed_cost_is_deterministic(self) -> None:
        frame = synthetic_frame()
        first = fixed_cost(frame, daughter_weight=0.5, focus_cost=4.0)
        second = fixed_cost(frame, daughter_weight=0.5, focus_cost=4.0)
        self.assertEqual(first.tolist(), second.tolist())

    def test_run_produces_all_arms_and_oof_scores(self) -> None:
        summary, selected, scores, video, manifest = run(synthetic_frame())
        self.assertEqual(set(summary["arm"]), {
            "geometry_1_05_focus0", "geometry_1_05_focus4",
            "geometry_1_00_focus4", "oof_logistic_3feat",
        })
        self.assertEqual(len(scores), 18 * 4)
        self.assertEqual(len(video), 3 * 4)
        self.assertTrue((summary["top1_recall"] >= 0.0).all())
        self.assertEqual(manifest["oof_group"], "video")
        self.assertIn("arm", selected.columns)

    def test_no_submission_or_official_cv(self) -> None:
        from tools import exp028_candidate_ranking as module
        self.assertFalse((module.OUT / "submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
