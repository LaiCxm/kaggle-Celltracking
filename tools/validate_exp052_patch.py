"""Static validation for the EXP052 Top-3 official-CV notebook."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from create_exp052_top3_division import top3_runtime_patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP052" / "CELL_post_ilp_division_top3_cv.ipynb"
RUNTIME = (
    ROOT / "EXP" / "EXP048" / "outputs" / "kaggle_v2" / "tracking_repo"
    / "scripts" / "predict_unet_transformer.py"
)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"EXP052:cell{index}", "exec")
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "_exp052_second_proposals",
        "_exp052_rank3_proposals",
        "proposals_rank3=_exp052_rank3",
        "apply_exp052_top3_division_proposals",
        'EXP052_MODES = ["off", "exp049_cos060", "exp052_top3_cos060"]',
        "EXP052_official_strategy_summary.csv",
    ]
    for token in required:
        assert token in joined, token
    assert "n_tgt >= 2" in joined
    assert "if n_tgt >= 3:" in joined
    assert "cosine_max=_exp052_threshold" in joined

    runtime = RUNTIME.read_text(encoding="gbk", errors="replace")
    with tempfile.TemporaryDirectory(prefix="exp052-runtime-") as temp_dir:
        path = Path(temp_dir) / "predict_unet_transformer.py"
        path.write_text(runtime, encoding="gbk", errors="replace")
        patch = top3_runtime_patch()
        exec(compile(patch, "EXP052-runtime-patch", "exec"), {"_ps": path})
        patched = path.read_text(encoding="gbk", errors="replace")
        compile(patched, str(path), "exec")
        assert "_exp052_rank3_proposals" in patched
        assert "proposals_rank3=_exp052_rank3" in patched
        assert "n_tgt >= 2" in patched
        assert "if n_tgt >= 3:" in patched
    print("EXP052 notebook validation passed")


if __name__ == "__main__":
    main()
