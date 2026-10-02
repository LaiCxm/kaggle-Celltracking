"""Run EXP021's synthetic contract checks without requiring pytest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_focus_first import (
    test_continuation_candidates_use_union_of_sources,
    test_division_candidates_do_not_require_unoccupied_daughters,
    test_instance_summary_keeps_confidence_and_bbox,
    test_unified_nodes_keep_focus_only_proposals,
    test_pipeline_uses_frame_local_focus_keys,
)


def main() -> None:
    checks = [
        test_instance_summary_keeps_confidence_and_bbox,
        test_unified_nodes_keep_focus_only_proposals,
        test_division_candidates_do_not_require_unoccupied_daughters,
        test_continuation_candidates_use_union_of_sources,
        test_pipeline_uses_frame_local_focus_keys,
    ]
    for check in checks:
        check()
    print(f"EXP021 selfcheck: PASS ({len(checks)} checks)")


if __name__ == "__main__":
    main()
