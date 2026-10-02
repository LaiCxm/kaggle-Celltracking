"""Static validation for the EXP048 deterministic runtime patch."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP048" / "CELL_deterministic_official_cv_audit.ipynb"
RUNTIME = ROOT / "EXP" / "EXP019" / "reference" / "tracking_cellmot_075fc5f" / "scripts" / "predict_unet_transformer.py"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][4]["source"])
    start = cell.index("# EXP048: deterministic runtime guard")
    end_marker = 'print("EXP048 deterministic runtime guard applied", _exp048_seed)\n'
    end = cell.index(end_marker, start) + len(end_marker)
    patch = cell[start:end]
    with tempfile.TemporaryDirectory(prefix="exp048-patch-") as temp_dir:
        scripts = Path(temp_dir) / "scripts"
        scripts.mkdir()
        runtime = scripts / "predict_unet_transformer.py"
        # Match the platform-default codec used by Path.read_text in the patch.
        # The Kaggle runtime is UTF-8; this local Windows validator uses GBK.
        runtime.write_text(RUNTIME.read_text(encoding="utf-8"), encoding="gbk", errors="replace")
        old_seed = os.environ.get("BIOHUB_DETERMINISTIC_SEED")
        os.environ["BIOHUB_DETERMINISTIC_SEED"] = "20260923"
        namespace = {
            "_ps": runtime,
            "os": os,
            "__builtins__": __builtins__,
        }
        try:
            exec(compile(patch, "EXP048-deterministic-patch", "exec"), namespace)
            patched = runtime.read_text(encoding="gbk")
        finally:
            if old_seed is None:
                os.environ.pop("BIOHUB_DETERMINISTIC_SEED", None)
            else:
                os.environ["BIOHUB_DETERMINISTIC_SEED"] = old_seed
        compile(patched, str(runtime), "exec")
        assert "torch.backends.cudnn.deterministic = True" in patched
        assert "torch.backends.cudnn.benchmark = False" in patched
        assert namespace["_s"] == patched
        print("EXP048 deterministic patch validation passed")


if __name__ == "__main__":
    main()
