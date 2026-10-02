"""Restore the test inference pass in the EXP050 submission notebook.

EXP050 was created from the EXP049 official-CV notebook, whose cell 4
intentionally skips test inference.  The submission notebook must run the
existing test prediction path once before calling ``write_test_submission``.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP050" / "CELL_infer_exp049_cos060_submission.ipynb"
REFERENCE = ROOT / "EXP" / "EXP018" / "CELL_infer_focus_mask_division.ipynb"


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    cell4 = _source(notebook["cells"][4])
    ref4 = _source(reference["cells"][4])
    marker = "predict_seconds = 0.0\nprint(\"EXP019 CV-only: skipped test-set inference; validation prediction runs once below.\")"
    ref_marker = "start_time = time.time()"
    if ref_marker not in ref4:
        raise RuntimeError("EXP018 reference test inference block is missing")

    if cell4.rstrip("\n").endswith(marker):
        # Reuse the proven production inference block. EXP050's preceding code
        # has the same helper names and additionally installs the EXP049
        # proposal capture patch, so the restored block feeds fixed cos060.
        test_pass = ref4[ref4.index(ref_marker):]
        prefix = cell4[: cell4.rfind(marker)]
        cell4 = prefix + test_pass
    elif "pstart_time = time.time()" in cell4:
        # Version 2 retained the final character of the removed CV-only print,
        # producing pstart_time while the footer still read start_time.
        cell4 = cell4.replace("pstart_time = time.time()", ref_marker, 1)
    elif ref_marker not in cell4:
        raise RuntimeError("EXP050 cell 4 has neither CV-only marker nor test inference start")
    notebook["cells"][4]["source"] = cell4.splitlines(True)

    cell5 = _source(notebook["cells"][5]).replace(
        'print("EXP019 CV-only: no test submission is generated.")',
        'print("EXP050: test submission writer is ready; fixed cos060 pass follows in the next cell.")',
    )
    notebook["cells"][5]["source"] = cell5.splitlines(True)

    cell10 = _source(notebook["cells"][10]).replace(
        'submission_rows = sum(1 for _ in SUBMISSION_PATH.open("r", encoding="utf-8")) - 1\n'
        'if submission_rows <= 0:\n'
        '    raise RuntimeError("EXP050 produced an empty submission.csv")\n'
        'if not SUBMISSION_PATH.exists() or SUBMISSION_PATH.name != "submission.csv":\n'
        '    raise RuntimeError("EXP050 did not produce the required submission.csv")',
        'if not SUBMISSION_PATH.is_file():\n'
        '    raise RuntimeError("EXP050 did not produce the required submission.csv")\n'
        'submission_rows = sum(1 for _ in SUBMISSION_PATH.open("r", encoding="utf-8")) - 1\n'
        'if submission_rows <= 0:\n'
        '    raise RuntimeError("EXP050 produced an empty submission.csv")',
    )
    notebook["cells"][10]["source"] = cell10.splitlines(True)

    cell11 = _source(notebook["cells"][11]).replace(
        'print("EXP019 design: same four complete movies and raw predictions across all arms")\n'
        'print("EXP019 arms:", EXP019_MODES)\n'
        'print("EXP019 formal metric: patched official scorer; legacy proxy not used for ranking")\n'
        'print("FOCUS cached inference calls:", getattr(FOCUS_PROVIDER, "calls", 0), "error:", FOCUS_PROVIDER.error)\n\n'
        'print("EXP019 mode: official-CV-only validation; test submission generation is intentionally absent.")',
        'print("EXP050 mode: test-set submission only; fixed EXP049 cos060 selector")\n'
        'print("EXP050 validator: disabled; no validation scoring or threshold scan")\n'
        'print(f"EXP050 test prediction graphs: {len(list((REPO_DIR / \'predictions\').glob(\'*/\' + METHOD + \'/split_0/*.geff\')))}")',
    )
    notebook["cells"][11]["source"] = cell11.splitlines(True)

    metadata = notebook.setdefault("metadata", {})
    kaggle = metadata.setdefault("kaggle", {})
    kaggle["title"] = "EXP050 EXP049 cos060 Test Submission"
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
