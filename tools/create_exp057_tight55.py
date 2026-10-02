"""Create EXP057: EXP049 cos060 with motion relink tight radius 5.5 um."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP056" / "CELL_post_ilp_edge_protected_official_cv.ipynb"
OUTPUT = ROOT / "EXP" / "EXP057" / "CELL_official_cv_cos060_tight55.ipynb"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cell0 = "".join(cells[0]["source"])
    cell0 += (
        "\n# EXP057: single registered motion-relink knob on top of EXP049 cos060.\n"
        'os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n'
        'print("EXP057 requested BIOHUB_MOTION_RELINK_TIGHT_UM=", os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"])\n'
    )
    cells[0]["source"] = cell0.splitlines(True)

    cell1 = "".join(cells[1]["source"])
    guard_anchor = "_drift = {}\n"
    guard_code = (
        "_exp057_motion_relink_raw = _guard_os.environ.get(\"BIOHUB_MOTION_RELINK_TIGHT_UM\")\n"
        "if _exp057_motion_relink_raw is None or not _guard_math.isclose(\n"
        "    float(_exp057_motion_relink_raw), 5.5, rel_tol=0.0, abs_tol=1e-12\n"
        "):\n"
        "    raise RuntimeError(\n"
        "        \"EXP057 configuration guard failed: expected BIOHUB_MOTION_RELINK_TIGHT_UM=5.5, \"\n"
        "        + f\"got {_exp057_motion_relink_raw!r}\"\n"
        "    )\n"
        "print(\"EXP057 resolved BIOHUB_MOTION_RELINK_TIGHT_UM:\", _exp057_motion_relink_raw)\n"
    )
    if cell1.count(guard_anchor) != 1:
        raise RuntimeError("EXP057 configuration guard anchor mismatch")
    cell1 = cell1.replace(guard_anchor, guard_code + guard_anchor, 1)
    cells[1]["source"] = cell1.splitlines(True)

    cell10 = "".join(cells[10]["source"])
    old_modes = 'EXP056_MODES = ["off", "exp049_cos060", "exp056_edge_protected"]'
    new_modes = 'EXP057_MODES = ["off", "exp049_cos060"]'
    if cell10.count(old_modes) != 1:
        raise RuntimeError("EXP057 mode list anchor mismatch")
    cell10 = cell10.replace(old_modes, new_modes, 1)
    cell10 = cell10.replace("for _mode in EXP056_MODES:", "for _mode in EXP057_MODES:", 1)
    cell10 = cell10.replace("EXP056 strategy:", "EXP057 strategy:", 1)
    for old, new in (
        ("EXP056_official_strategy_summary.csv", "EXP057_official_strategy_summary.csv"),
        ("EXP056_official_strategy_per_movie.csv", "EXP057_official_strategy_per_movie.csv"),
        ("EXP056_strategy_stats.csv", "EXP057_strategy_stats.csv"),
    ):
        if cell10.count(old) != 1:
            raise RuntimeError(f"EXP057 output anchor mismatch: {old}")
        cell10 = cell10.replace(old, new, 1)
    cell10 = cell10.replace(
        "# EXP056 is an official-CV comparison, not a competition submission.",
        "# EXP057 is an official-CV comparison, not a competition submission.",
        1,
    )
    cell10 = cell10.replace("EXP056 arm", "EXP057 arm", 1)
    cell10 = cell10.replace(
        "EXP056 official CV complete; no test submission is generated.",
        "EXP057 official CV complete; no test submission is generated.",
        1,
    )
    cells[10]["source"] = cell10.splitlines(True)

    cell11 = "".join(cells[11]["source"]).replace("EXP056", "EXP057")
    cells[11]["source"] = cell11.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP057 cos060 Motion Relink Tight55 Official CV"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
