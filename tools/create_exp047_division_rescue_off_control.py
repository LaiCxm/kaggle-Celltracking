"""Create EXP047: clean off-control for EXP046's second-edge rescue."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP046" / "CELL_division_second_edge_rescue_official_cv.ipynb"
OUTPUT = ROOT / "EXP" / "EXP047" / "CELL_division_second_edge_rescue_off_control.ipynb"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    cell0 = cell0.replace(
        'os.environ["BIOHUB_DIVISION_RESCUE_SECOND_THRESHOLD"] = "0.18"',
        'os.environ["BIOHUB_DIVISION_RESCUE_SECOND_THRESHOLD"] = "1.10"',
    ).replace(
        'os.environ["BIOHUB_DIVISION_RESCUE_TOP1_MIN"] = "0.60"',
        'os.environ["BIOHUB_DIVISION_RESCUE_TOP1_MIN"] = "2.00"',
    )
    notebook["cells"][0]["source"] = cell0.splitlines(True)
    notebook["cells"][0]["source"].append(
        "# EXP047 control: thresholds above probability range disable all rescue pairs.\n"
    )
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP047 Division Second-Edge Rescue Off Control"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
