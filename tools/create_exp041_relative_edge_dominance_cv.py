"""Create EXP041: official-CV relative edge-dominance gating."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP040" / "CELL_official_cv_local_division_conflict.ipynb"
OUT = ROOT / "EXP" / "EXP041" / "CELL_official_cv_relative_edge_dominance.ipynb"
MODULE = ROOT / "tools" / "division_conflict_rewire.py"


CONFIG = '''
DIVISION_CONFLICT_MODE = "base"
DIVISION_CONFLICT_LOGS: list[dict[str, object]] = []
DIVISION_CONFLICT_CONFIGS = {
    "base": {"parent_gate_um": None, "daughter_gate_um": None},
    "dominance_1_14": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 1.0,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
    "dominance_2_14": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 2.0,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
    "dominance_4_14": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 4.0,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
}
print("EXP041 conflict modes:", DIVISION_CONFLICT_CONFIGS)
'''


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 13:
        raise RuntimeError(f"unexpected EXP040 notebook cell count: {len(cells)}")

    cell0 = "".join(cells[0]["source"])
    cell0 = cell0.replace("EXP040", "EXP041")
    cells[0]["source"] = cell0.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    old_call = '            divergence_um=float(_conflict_config["divergence_um"]),\n'
    new_call = old_call + '            min_displacement_gain_um=float(_conflict_config["min_displacement_gain_um"]),\n'
    if old_call not in cell5:
        raise RuntimeError("EXP040 rewire call not found")
    cell5 = cell5.replace(old_call, new_call, 1)
    cells[5]["source"] = cell5.splitlines(True)

    module_source = MODULE.read_text(encoding="utf-8") + CONFIG
    cells[6]["source"] = module_source.splitlines(True)

    runner = "".join(cells[11]["source"])
    runner = runner.replace("EXP040", "EXP041").replace("exp040_", "exp041_")
    runner = runner.replace(
        'EXP041_MODES = ["base", "compact_10_14", "compact_12_14", "compact_14_14"]',
        'EXP041_MODES = ["base", "dominance_1_14", "dominance_2_14", "dominance_4_14"]',
    )
    runner = runner.replace(
        "EXP041 official CV: compact occupied-daughter conflict rewiring",
        "EXP041 official CV: relative edge-dominance occupied-daughter rewiring",
    )
    runner = runner.replace("for _mode in EXP040_MODES:", "for _mode in EXP041_MODES:")
    old_summary = '        "rewire_rejected_divergence": sum(int(row.get("division_conflict_rejected_divergence", 0)) for row in _mode_stats),\n'
    new_summary = old_summary + '        "rewire_rejected_advantage": sum(int(row.get("division_conflict_rejected_advantage", 0)) for row in _mode_stats),\n'
    if old_summary not in runner:
        raise RuntimeError("EXP040 summary insertion point not found")
    runner = runner.replace(old_summary, new_summary, 1)
    cells[11]["source"] = runner.splitlines(True)
    cells[12]["source"] = [
        'print("EXP041 manifest: official patched scorer, CV-only, relative edge-dominance gating.")\n'
    ]
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP041 Official CV Relative Edge Dominance"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
