"""Create EXP043: strict learned-edge provenance audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP042" / "CELL_official_cv_edge_confidence_dominance.ipynb"
OUT = ROOT / "EXP" / "EXP043" / "CELL_official_cv_strict_edge_confidence.ipynb"
MODULE = ROOT / "tools" / "division_conflict_rewire.py"


CONFIG = '''
DIVISION_CONFLICT_MODE = "base"
DIVISION_CONFLICT_LOGS: list[dict[str, object]] = []
DIVISION_CONFLICT_CONFIGS = {
    "base": {"parent_gate_um": None, "daughter_gate_um": None},
    "strict_prob05": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 1.0,
        "min_edge_probability_gain": 0.05,
        "require_edge_probability_present": True,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
    "strict_prob10": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 1.0,
        "min_edge_probability_gain": 0.10,
        "require_edge_probability_present": True,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
    "strict_prob20": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "min_displacement_gain_um": 1.0,
        "min_edge_probability_gain": 0.20,
        "require_edge_probability_present": True,
        "max_changes_per_frame": 1, "max_changes_per_dataset": 6,
        "require_divergence": True,
    },
}
print("EXP043 conflict modes:", DIVISION_CONFLICT_CONFIGS)
'''


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 13:
        raise RuntimeError(f"unexpected EXP042 notebook cell count: {len(cells)}")

    cell0 = "".join(cells[0]["source"]).replace("EXP042", "EXP043")
    cells[0]["source"] = cell0.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    old_call = '            min_edge_probability_gain=float(_conflict_config["min_edge_probability_gain"]),\n'
    new_call = old_call + '            require_edge_probability_present=bool(_conflict_config["require_edge_probability_present"]),\n'
    if old_call not in cell5:
        raise RuntimeError("EXP042 rewire call not found")
    cells[5]["source"] = cell5.replace(old_call, new_call, 1).splitlines(True)

    cells[6]["source"] = (MODULE.read_text(encoding="utf-8") + CONFIG).splitlines(True)

    runner = "".join(cells[11]["source"])
    runner = runner.replace("EXP042", "EXP043").replace("exp042_", "exp043_")
    runner = runner.replace(
        'EXP043_MODES = ["base", "geom1_prob0", "geom1_prob05", "geom1_prob10"]',
        'EXP043_MODES = ["base", "strict_prob05", "strict_prob10", "strict_prob20"]',
    )
    old_summary = '        "rewire_rejected_edge_confidence": sum(int(row.get("division_conflict_rejected_edge_confidence", 0)) for row in _mode_stats),\n'
    new_summary = old_summary + '        "rewire_rejected_missing_edge_confidence": sum(int(row.get("division_conflict_rejected_missing_edge_confidence", 0)) for row in _mode_stats),\n'
    if old_summary not in runner:
        raise RuntimeError("EXP042 summary insertion point not found")
    runner = runner.replace(old_summary, new_summary, 1)
    runner = runner.replace(
        "EXP043 official CV: geometry plus edge-confidence dominance rewiring",
        "EXP043 official CV: strict learned-edge provenance gating",
    )
    cells[11]["source"] = runner.splitlines(True)
    cells[12]["source"] = [
        'print("EXP043 manifest: official patched scorer, CV-only, strict learned-edge provenance.")\n'
    ]
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP043 Official CV Strict Edge Confidence"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
