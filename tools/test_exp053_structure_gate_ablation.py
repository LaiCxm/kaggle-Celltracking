"""Static and unit checks for EXP053 structural-gate ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from create_exp053_structure_gate_ablation import GATE_TABLE, _exp053_integration
from exp052_post_ilp_division import apply_exp052_top3_division_proposals


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP053" / "CELL_post_ilp_division_gate_ablation_cv.ipynb"


def _node(node_id: int, t: int, x: float) -> dict[str, object]:
    return {"node_id": node_id, "t": t, "z": 0.0, "y": 0.0, "x": float(x)}


def _event(primary_x: float, second_x: float, primary_next_x: float, second_next_x: float):
    nodes = {
        1: _node(1, 0, 0.0),
        2: _node(2, 1, primary_x),
        3: _node(3, 1, second_x),
        4: _node(4, 2, primary_next_x),
        5: _node(5, 2, second_next_x),
    }
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 2, "target_id": 4},
        {"source_id": 3, "target_id": 5},
    ]
    proposal = np.asarray([[1, 2, 3, 0.8, 0.3]], dtype=np.float64)
    return nodes, edges, proposal


def _accepts(event, *, sister: float, asymmetry: float, divergence: float) -> bool:
    nodes, edges, proposal = event
    output = apply_exp052_top3_division_proposals(
        nodes,
        edges,
        proposal,
        np.empty((0, 5), dtype=np.float64),
        cosine_max=-0.9,
        scale_um=(1.0, 1.0, 1.0),
        min_sister_um=sister,
        max_sister_um=14.0,
        max_distance_asymmetry=asymmetry,
        min_divergence_gain_um=divergence,
        max_per_frame=1,
        max_total=64,
    )
    return (1, 3) in {(int(row["source_id"]), int(row["target_id"])) for row in output}


def main() -> None:
    expected = {
        "exp053_top3_baseline": (8.0, 0.60, 2.25),
        "exp053_top3_sister075": (7.5, 0.60, 2.25),
        "exp053_top3_asym110": (8.0, 1.10, 2.25),
        "exp053_top3_diverge175": (8.0, 0.60, 1.75),
        "exp053_top3_joint": (7.5, 1.10, 1.75),
    }
    assert set(GATE_TABLE) == set(expected)
    for mode, values in expected.items():
        assert tuple(GATE_TABLE[mode].values()) == values, (mode, GATE_TABLE[mode])

    baseline = expected["exp053_top3_baseline"]
    for mode, values in expected.items():
        changed = sum(a != b for a, b in zip(values, baseline))
        assert changed == (0 if mode.endswith("baseline") else 1 if mode != "exp053_top3_joint" else 3), mode

    # Three synthetic events each violate exactly one baseline gate.  The
    # corresponding single-variable arm must accept it while the baseline
    # remains closed; this tests actual geometry, not only configuration text.
    sister_event = _event(3.875, -3.875, 5.2, -5.2)
    assert not _accepts(sister_event, sister=8.0, asymmetry=0.60, divergence=2.25)
    assert _accepts(sister_event, sister=7.5, asymmetry=0.60, divergence=2.25)

    asym_event = _event(6.0, -2.0, 7.5, -4.0)
    assert not _accepts(asym_event, sister=8.0, asymmetry=0.60, divergence=2.25)
    assert _accepts(asym_event, sister=8.0, asymmetry=1.10, divergence=2.25)

    # Sister distance is exactly 8 and asymmetry is zero.  The next-frame
    # distance is 10, so divergence gain is 2: below the baseline 2.25 but
    # above the relaxed 1.75 gate.
    divergence_event = _event(4.0, -4.0, 5.0, -5.0)
    assert not _accepts(divergence_event, sister=8.0, asymmetry=0.60, divergence=2.25)
    assert _accepts(divergence_event, sister=8.0, asymmetry=0.60, divergence=1.75)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"EXP053:cell{index}", "exec")
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        'EXP053_MODES = ["off", "exp049_cos060", "exp053_top3_baseline", "exp053_top3_sister075", "exp053_top3_asym110", "exp053_top3_diverge175", "exp053_top3_joint"]',
        "proposals_rank3",
        "apply_exp052_top3_division_proposals",
        "EXP053_official_strategy_summary.csv",
        "min_sister=",
        "max_asym=",
        "min_diverge=",
    ]
    for token in required:
        assert token in joined, token
    assert "submission.csv" in joined  # explicit no-submission guard message
    assert "no second test pass and no submission.csv" in joined

    integration = _exp053_integration()
    assert integration.count("min_sister_um=_exp053_min_sister") == 2
    assert integration.count("max_distance_asymmetry=_exp053_max_asym") == 2
    assert integration.count("min_divergence_gain_um=_exp053_min_diverge") == 2
    assert "if FOCUS_STRATEGY_MODE == \"exp049_cos060\"" in integration
    assert "proposals_rank3" in integration
    print("EXP053 structural-gate ablation validation passed")


if __name__ == "__main__":
    main()
