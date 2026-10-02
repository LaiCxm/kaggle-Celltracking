from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP051" / "CELL_public_0947_official_cv.ipynb"


def test_notebook_schema_and_compiles() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 12
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell{index}")


def test_fixed_public_candidate_and_official_scorer() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "BIOHUB_MOTION_RELINK_TIGHT_UM\"] = \"5.5\"" in source
    assert "44b6_12dfb391" in source
    assert "44b6_267148e4" in source
    assert "6bba_062c8d37" in source
    assert "6bba_07e24132" in source
    assert "tracking_cellmot_075fc5f" in source
    assert "_official_summarise" in source
    assert "csv_to_geffs" in source
    assert "evaluate_pairs" in source
    assert "_exp051_importlib.spec_from_file_location" in source
    assert "_exp051_importlib.module_from_spec" in source
    assert "_exp051_importlib.util." not in source


def test_no_proxy_sweep_or_submission() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "PP_CANDIDATES" not in source
    assert "score_validator_config" not in source
    assert "write_test_submission(selected_label)" not in source
    assert "write_test_submission(\"" not in source
    assert "FOCUS3D=False" in source
    assert "EXP049 post-ILP division promotion=False" in source
