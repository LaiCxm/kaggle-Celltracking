"""Create EXP042: official-CV relative geometry and edge-confidence gating."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP041" / "CELL_official_cv_relative_edge_dominance.ipynb"
OUT = ROOT / "EXP" / "EXP042" / "CELL_official_cv_edge_confidence_dominance.ipynb"
MODULE = ROOT / "tools" / "division_conflict_rewire.py"


CONFIG = '''
DIVISION_CONFLICT_MODE = "base"
DIVISION_CONFLICT_LOGS: list[dict[str, object]] = []
DIVISION_CONFLICT_CONFIGS = {
    "base": {"parent_gate_um": None, "daughter_gate_um": None},
    "geom1_prob0": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 1.0,
        "min_edge_probability_gain": 0.0,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
    "geom1_prob05": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 1.0,
        "min_edge_probability_gain": 0.05,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
    "geom1_prob10": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 1.0,
        "min_edge_probability_gain": 0.10,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
}
print("EXP042 conflict modes:", DIVISION_CONFLICT_CONFIGS)
'''


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 13:
        raise RuntimeError(f"unexpected EXP041 notebook cell count: {len(cells)}")

    cell0 = "".join(cells[0]["source"]).replace("EXP041", "EXP042")
    cells[0]["source"] = cell0.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    old_call = '            min_displacement_gain_um=float(_conflict_config["min_displacement_gain_um"]),\n'
    new_call = old_call + '            min_edge_probability_gain=float(_conflict_config["min_edge_probability_gain"]),\n'
    if old_call not in cell5:
        raise RuntimeError("EXP041 rewire call not found")
    cells[5]["source"] = cell5.replace(old_call, new_call, 1).splitlines(True)

    cells[6]["source"] = (MODULE.read_text(encoding="utf-8") + CONFIG).splitlines(True)

    runner = "".join(cells[11]["source"])
    runner = runner.replace("EXP041", "EXP042").replace("exp041_", "exp042_")
    runner = runner.replace(
        'EXP042_MODES = ["base", "dominance_1_14", "dominance_2_14", "dominance_4_14"]',
        'EXP042_MODES = ["base", "geom1_prob0", "geom1_prob05", "geom1_prob10"]',
    )
    old_summary = '        "rewire_rejected_advantage": sum(int(row.get("division_conflict_rejected_advantage", 0)) for row in _mode_stats),\n'
    new_summary = old_summary + '        "rewire_rejected_edge_confidence": sum(int(row.get("division_conflict_rejected_edge_confidence", 0)) for row in _mode_stats),\n'
    if old_summary not in runner:
        raise RuntimeError("EXP041 summary insertion point not found")
    runner = runner.replace(old_summary, new_summary, 1)
    runner = runner.replace(
        "EXP042 official CV: relative edge-dominance occupied-daughter rewiring",
        "EXP042 official CV: geometry plus edge-confidence dominance rewiring",
    )
    cells[11]["source"] = runner.splitlines(True)
    cells[12]["source"] = [
        'print("EXP042 manifest: official patched scorer, CV-only, geometry plus edge-confidence gating.")\n'
    ]
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP042 Official CV Edge Confidence Dominance"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
