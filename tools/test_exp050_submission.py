from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP050" / "CELL_infer_exp049_cos060_submission.ipynb"


def source(index: int) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(notebook["cells"][index].get("source", []))


def test_notebook_compiles() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 12
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell{index}")


def test_test_inference_timer_is_defined_before_use() -> None:
    cell4 = source(4)
    assert "pstart_time" not in cell4
    assert cell4.count("start_time = time.time()") == 1
    assert cell4.count("predict_seconds = time.time() - start_time") == 1
    assert cell4.index("start_time = time.time()") < cell4.index(
        "predict_seconds = time.time() - start_time"
    )
    assert "_merge_prediction_shards(worker_count)" in cell4


def test_fixed_cos060_writes_submission() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert 'FOCUS_STRATEGY_MODE = "exp049_cos060"' in all_source
    assert 'write_test_submission("exp049_cos060")' in all_source
    assert "SUBMISSION_PATH.is_file()" in all_source
    assert "skipped test-set inference" not in all_source
