"""Create the EXP019 v2 notebook with a valid competition submission.

The v1 notebook is intentionally diagnostic and renames the test submission.
This v2 keeps the official four-movie comparison, then applies the selected
conservative arm to the complete test set and leaves /kaggle/working/submission.csv
in place for Kaggle's submission runner.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP019" / "CELL_focus3d_strategy_official_compare.ipynb"
DST = ROOT / "EXP" / "EXP019" / "CELL_focus3d_strategy_official_compare_submit.ipynb"


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(True)


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    comparison = source(cells[10])
    old = """# This is a diagnostic comparison, not a competition submission. Prevent the
# untouched test-base file from being mistaken for the selected EXP019 arm.
_reference_submission = WORKING_DIR / "test_base_reference_do_not_submit.csv"
if SUBMISSION_PATH.exists():
    SUBMISSION_PATH.replace(_reference_submission)
print("EXP019 comparison complete; no submission.csv intentionally produced.")
"""
    new = """# Keep the untouched test output for audit, then create the actual EXP019 v2
# submission with the pre-registered conservative arm.  The validation table is
# still produced above; this final pass is the only test-set output.
_reference_submission = WORKING_DIR / "test_base_reference.csv"
if SUBMISSION_PATH.exists():
    SUBMISSION_PATH.replace(_reference_submission)

EXP019_SELECTED_MODE = "selected_rescue"
FOCUS_STRATEGY_MODE = EXP019_SELECTED_MODE
TEST_DIR = COMP_DIR / "test"
write_test_submission(EXP019_SELECTED_MODE)

if not SUBMISSION_PATH.is_file():
    raise RuntimeError("EXP019 v2 failed to produce submission.csv")
_final_header = SUBMISSION_PATH.open().readline().strip().split(",")
if _final_header != CSV_COLUMNS:
    raise RuntimeError({"submission_header": _final_header, "expected": CSV_COLUMNS})
print(
    "EXP019 v2 complete: wrote submission.csv using mode=",
    EXP019_SELECTED_MODE,
    "rows=",
    sum(1 for _ in SUBMISSION_PATH.open()) - 1,
)
"""
    if old not in comparison:
        raise RuntimeError("v1 diagnostic footer not found")
    set_source(cells[10], comparison.replace(old, new, 1))
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP019 FOCUS3D Strategy Official Compare + Submission"
    )
    DST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
