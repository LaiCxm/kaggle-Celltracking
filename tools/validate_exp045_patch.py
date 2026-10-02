"""Static validation for the EXP045 notebook's dynamic runtime patch."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP045" / "CELL_prethreshold_candidate_audit.ipynb"
RUNTIME = ROOT / "EXP" / "EXP019" / "reference" / "tracking_cellmot_075fc5f" / "scripts" / "predict_unet_transformer.py"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][8]["source"])
    start = cell.index("# Apply eight-view planar detection TTA before graph prediction.")
    end = cell.index("def list_test_stems", start)
    patch = cell[start:end]

    with tempfile.TemporaryDirectory(prefix="exp045-patch-") as temp_dir:
        scripts_dir = Path(temp_dir) / "scripts"
        scripts_dir.mkdir()
        runtime_path = scripts_dir / "predict_unet_transformer.py"
        # The notebook runs on Kaggle Linux, while this local static check is
        # executed on Windows where Path.read_text() uses the active code page.
        # Match that implicit read for the temporary file, then decode the
        # patched result explicitly below.
        runtime_text = RUNTIME.read_text(encoding="utf-8")
        runtime_path.write_bytes(runtime_text.encode("gbk", errors="replace"))
        previous_weight = os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT")
        os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"
        namespace = {
            "REPO_DIR": Path(temp_dir),
            "WORKING_DIR": Path(temp_dir),
            "_ps": runtime_path,
            "_s": runtime_path.read_text(encoding="gbk"),
            "_bidirectional_weight_guard": 0.15,
            "os": os,
            "re": re,
            "Path": Path,
        }
        # The notebook code imports these names in earlier cells; supply them
        # explicitly so this check exercises only the patch-generation logic.
        namespace["__builtins__"] = __builtins__
        try:
            exec(compile(patch, "EXP045-cell8-patch", "exec"), namespace)
            patched = runtime_path.read_text(encoding="gbk")
        finally:
            if previous_weight is None:
                os.environ.pop("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", None)
            else:
                os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = previous_weight
        compile(patched, str(runtime_path), "exec")
        required = ("prethreshold_edges", "_audit_top_k", "raw_edges=_prethreshold_edge_array")
        missing = [token for token in required if token not in patched]
        if missing:
            raise AssertionError(f"EXP045 patch is missing runtime hooks: {missing}")
        print("EXP045 dynamic patch validation passed")
        print(f"patched runtime size: {len(patched)} bytes")


if __name__ == "__main__":
    main()
