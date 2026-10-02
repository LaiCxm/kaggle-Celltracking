"""Compile-check the EXP046 candidate-generation patch."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP046" / "CELL_division_second_edge_rescue_official_cv.ipynb"
RUNTIME = ROOT / "EXP" / "EXP019" / "reference" / "tracking_cellmot_075fc5f" / "scripts" / "predict_unet_transformer.py"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][4]["source"])
    start = cell.index("# EXP046 targeted division second-edge rescue patch")
    end = cell.index("def list_test_stems", start)
    patch = cell[start:end]
    with tempfile.TemporaryDirectory(prefix="exp046-patch-") as temp_dir:
        scripts = Path(temp_dir) / "scripts"
        scripts.mkdir()
        runtime = scripts / "predict_unet_transformer.py"
        runtime.write_bytes(RUNTIME.read_text(encoding="utf-8").encode("gbk", errors="replace"))
        namespace = {"_s": runtime.read_text(encoding="gbk"), "_ps": runtime, "__builtins__": __builtins__}
        exec(compile(patch, "EXP046-rescue-patch", "exec"), namespace)
        patched = runtime.read_text(encoding="gbk")
        compile(patched, str(runtime), "exec")
        assert "_division_rescue_pairs" in patched
        assert "BIOHUB_DIVISION_RESCUE_SECOND_THRESHOLD" in patched
        print("EXP046 dynamic patch validation passed")


if __name__ == "__main__":
    main()
