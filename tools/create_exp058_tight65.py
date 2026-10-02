"""Create EXP058: EXP049 cos060 with motion relink tight radius 6.5 um."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP057" / "CELL_official_cv_cos060_tight55.ipynb"
OUTPUT = ROOT / "EXP" / "EXP058" / "CELL_official_cv_cos060_tight65.ipynb"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cell0 = "".join(cells[0]["source"])
    old = 'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"'
    if cell0.count(old) != 1:
        raise RuntimeError("EXP058 expected one EXP057 motion-relink assignment")
    cell0 = cell0.replace(old, 'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.5"', 1)
    cell0 = cell0.replace("EXP057 requested", "EXP058 requested", 1)
    cells[0]["source"] = cell0.splitlines(True)

    cell1 = "".join(cells[1]["source"])
    if cell1.count("_exp057_motion_relink_raw") != 5:
        raise RuntimeError("EXP058 guard anchor mismatch")
    cell1 = cell1.replace("_exp057_motion_relink_raw", "_exp058_motion_relink_raw")
    cell1 = cell1.replace("5.5", "6.5")
    cell1 = cell1.replace("EXP057 configuration guard", "EXP058 configuration guard")
    cell1 = cell1.replace("EXP057 resolved", "EXP058 resolved")
    cells[1]["source"] = cell1.splitlines(True)

    for index in (10, 11):
        source = "".join(cells[index]["source"])
        source = source.replace("EXP057", "EXP058")
        cells[index]["source"] = source.splitlines(True)

    kaggle = notebook.setdefault("metadata", {}).setdefault("kaggle", {})
    kaggle["title"] = "EXP058 cos060 Motion Relink Tight65 Official CV"
    notebook["metadata"]["title"] = "EXP058 cos060 Motion Relink Tight65 Official CV"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
