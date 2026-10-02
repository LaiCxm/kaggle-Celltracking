"""Static validation for the EXP054 joint-selection notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP054" / "CELL_post_ilp_joint_selection_cv.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"EXP054:cell{index}", "exec")
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "apply_exp054_joint_division_proposals",
        'EXP054_MODES = ["off", "exp049_cos060", "exp054_joint_top2", "exp054_joint_top3"]',
        "EXP054_official_strategy_summary.csv",
        "exp054_joint_source_replaced",
        "no second test pass and no submission.csv",
    ]
    for token in required:
        assert token in joined, token
    assert "submission.csv" in joined
    print("EXP054 notebook validation passed")


if __name__ == "__main__":
    main()
