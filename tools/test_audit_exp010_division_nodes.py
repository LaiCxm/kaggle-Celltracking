import unittest

import numpy as np

from audit_exp010_division_nodes import match_nodes_by_frame, topk_candidate_ids


class AuditHelpersTest(unittest.TestCase):
    def test_matching_is_one_to_one_and_respects_radius(self):
        pred = {
            10: (0, 0.0, 0.0, 0.0),
            11: (0, 0.0, 0.0, 20.0),
        }
        gt = {
            1: (0, 0.0, 0.0, 0.0),
            2: (0, 0.0, 0.0, 100.0),
        }
        gt_to_pred, pred_to_gt, distances = match_nodes_by_frame(pred, gt, max_distance_um=7.0)
        self.assertEqual(gt_to_pred, {1: 10})
        self.assertEqual(pred_to_gt, {10: 1})
        self.assertTrue(np.isclose(distances[1], 0.0))

    def test_topk_candidate_is_next_frame_radius_limited_and_deterministic(self):
        nodes = {
            1: (0, 0.0, 0.0, 0.0),
            2: (1, 0.0, 0.0, 2.0),
            3: (1, 0.0, 0.0, 1.0),
            4: (1, 0.0, 0.0, 100.0),
            5: (2, 0.0, 0.0, 0.0),
        }
        ranked = topk_candidate_ids(nodes[1], [2, 3, 4, 5], nodes, radius_um=1.0, top_k=1)
        self.assertEqual([node_id for node_id, _ in ranked], [3])


if __name__ == "__main__":
    unittest.main()
