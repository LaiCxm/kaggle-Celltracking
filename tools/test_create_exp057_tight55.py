"""Static checks for EXP057 notebook generation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP057" / "CELL_official_cv_cos060_tight55.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    for i, cell in enumerate(code):
        compile("".join(cell.get("source", [])), f"EXP057:cell{i}", "exec")
    joined = "\n".join("".join(c.get("source", [])) for c in code)
    assert 'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"' in joined
    assert 'EXP057_MODES = ["off", "exp049_cos060"]' in joined
    assert "EXP057_official_strategy_summary.csv" in joined
    assert "EXP056_MODES" not in joined
    print("EXP057 notebook validation passed")


if __name__ == "__main__":
    main()
