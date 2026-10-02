"""Create the corrected EXP019 official-CV-only notebook.

EXP019 is a validation experiment.  It must not run a second test-set pass or
emit a competition submission after the five-arm comparison.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP019" / "CELL_focus3d_strategy_official_compare.ipynb"
DST = ROOT / "EXP" / "EXP019" / "CELL_focus3d_strategy_official_cv.ipynb"


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    # The source notebook also contains the ordinary test-set inference pass.
    # EXP019 is CV-only, so skip that pass entirely; validation prediction in
    # cell 7 remains the single prediction run used by the experiment.
    cell4 = "".join(cells[4].get("source", []))
    marker = "start_time = time.time()"
    if marker not in cell4:
        raise RuntimeError("test inference marker not found")
    cell4 = cell4.split(marker, 1)[0] + (
        "predict_seconds = 0.0\n"
        "print(\"EXP019 CV-only: skipped test-set inference; validation prediction runs once below.\")\n"
    )
    cells[4]["source"] = cell4.splitlines(True)
    cell5 = "".join(cells[5].get("source", []))
    if "write_test_submission(\"base\")" not in cell5:
        raise RuntimeError("test submission call not found")
    cell5 = cell5.replace(
        "write_test_submission(\"base\")",
        "print(\"EXP019 CV-only: no test submission is generated.\")",
    )
    cells[5]["source"] = cell5.splitlines(True)
    # The retention guard belongs to production submissions and would reject a
    # CV-only run because no submission.csv is expected.
    cells[6]["source"] = [
        "print(\"EXP019 CV-only: submission retention guard skipped by design.\")\n"
    ]
    # Keep v1's single prediction/cache pass and five official-scoring arms.
    # Make the end-state explicit so this notebook cannot be mistaken for a
    # competition submission kernel.
    source = "".join(cells[10].get("source", []))
    source = source.replace(
        "print(\"EXP019 comparison complete; no submission.csv intentionally produced.\")",
        "print(\"EXP019 official CV complete; no second test pass and no submission.csv.\")",
    )
    cells[10]["source"] = source.splitlines(True)
    manifest = "".join(cells[11].get("source", []))
    manifest += (
        "\nprint(\"EXP019 mode: official-CV-only validation; test submission generation is intentionally absent.\")\n"
    )
    cells[11]["source"] = manifest.splitlines(True)
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP019 FOCUS3D Official CV Only"
    )
    DST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
