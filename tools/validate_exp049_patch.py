"""Static and dynamic checks for the EXP049 notebook."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from create_exp049_post_ilp_division import RUNTIME_PATCH


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP049" / "CELL_post_ilp_division_proposal_cv.ipynb"
RUNTIME = (
    ROOT / "EXP" / "EXP048" / "outputs" / "kaggle_v2" / "tracking_repo"
    / "scripts" / "predict_unet_transformer.py"
)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"EXP049:cell{index}", "exec")
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "_exp049_division_proposals",
        "EXP049 proposal capture patch applied",
        "apply_post_ilp_division_proposals",
        '"exp049_cos090": -0.90',
        '"exp049_cos075": -0.75',
        '"exp049_cos060": -0.60',
        "EXP049_official_strategy_summary.csv",
    ]
    for token in required:
        assert token in joined, token
    assert 'EXP019_MODES = ["off", "exp049_cos090", "exp049_cos075", "exp049_cos060"]' in joined
    source_cell = "".join(notebook["cells"][5]["source"])
    helper_start = source_cell.index("# EXP049 embedded structural selector.")
    helper_end = source_cell.index("\ndef filter_output_graph(", helper_start)
    helper_block = source_cell[helper_start:helper_end]
    helper_names = set(re.findall(r"^def\s+([A-Za-z_]\w*)", helper_block, flags=re.MULTILINE))
    original_cell = "".join(json.loads(
        (ROOT / "EXP" / "EXP048" / "CELL_deterministic_official_cv_audit.ipynb")
        .read_text(encoding="utf-8")
    )["cells"][5]["source"])
    original_names = set(re.findall(r"^def\s+([A-Za-z_]\w*)", original_cell, flags=re.MULTILINE))
    assert not (helper_names & original_names), helper_names & original_names
    with tempfile.TemporaryDirectory(prefix="exp049-runtime-") as temp_dir:
        runtime = Path(temp_dir) / "predict_unet_transformer.py"
        runtime.write_text(
            RUNTIME.read_text(encoding="utf-8"), encoding="gbk", errors="replace"
        )
        exec(compile(RUNTIME_PATCH, "EXP049-runtime-patch", "exec"), {"_ps": runtime})
        patched = runtime.read_text(encoding="gbk")
        compile(patched, str(runtime), "exec")
        assert "_exp049_division_proposals" in patched
        assert "EXP049 proposals" in patched
    print("EXP049 notebook validation passed")


if __name__ == "__main__":
    main()
