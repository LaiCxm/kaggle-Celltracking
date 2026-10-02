"""Create the EXP054 official-CV notebook for joint fork selection."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP053" / "CELL_post_ilp_division_gate_ablation_cv.ipynb"
OUTPUT = ROOT / "EXP" / "EXP054" / "CELL_post_ilp_joint_selection_cv.ipynb"
HELPER = ROOT / "tools" / "exp054_joint_division.py"


INTEGRATION = r'''
    if FOCUS_STRATEGY_MODE in {
        "exp049_cos060",
        "exp054_joint_top2",
        "exp054_joint_top3",
    } and dataset is not None:
        _exp054_path = Path(os.environ.get(
            "BIOHUB_EXP052_PROPOSAL_DIR", "/kaggle/working/exp052_candidates"
        )) / f"{dataset}.npz"
        if not _exp054_path.exists():
            raise FileNotFoundError(_exp054_path)
        with np.load(_exp054_path) as _exp054_payload:
            _exp054_top2 = np.asarray(_exp054_payload["proposals"], dtype=np.float64)
            _exp054_rank3 = np.asarray(
                _exp054_payload.get("proposals_rank3", np.empty((0, 5))),
                dtype=np.float64,
            )
        _exp054_threshold = -0.60

        def _exp054_accept(_node):
            if not DEEPCENTER_SAFE_DIV_VETO:
                return True
            return deepcenter_accept_repair_point(
                dataset,
                int(_node["t"]),
                node_point(_node),
                deepcenter_bundle,
                repair_frame_cache,
                deepcenter_heatmap_cache,
                stats,
                "safe_div",
                DEEPCENTER_SAFE_DIV_THRESHOLD,
            )

        if FOCUS_STRATEGY_MODE == "exp049_cos060":
            edges = apply_post_ilp_division_proposals(
                nodes_by_id, edges, _exp054_top2,
                cosine_max=_exp054_threshold,
                scale_um=VOXEL_SCALE_UM,
                min_primary_probability=0.60,
                min_second_probability=0.18,
                max_parent_um=7.0,
                min_sister_um=8.0,
                max_sister_um=14.0,
                max_distance_asymmetry=0.60,
                min_divergence_gain_um=SAFE_DIV_DIVERGE_UM,
                max_per_frame=1,
                max_total=64,
                accept_candidate=_exp054_accept,
                stats=stats,
            )
        else:
            _exp054_rows = _exp054_top2
            if FOCUS_STRATEGY_MODE == "exp054_joint_top3":
                _exp054_rows = np.vstack((_exp054_top2, _exp054_rank3))
            edges = apply_exp054_joint_division_proposals(
                nodes_by_id, edges, _exp054_rows,
                cosine_max=_exp054_threshold,
                scale_um=VOXEL_SCALE_UM,
                min_primary_probability=0.60,
                min_second_probability=0.18,
                max_parent_um=7.0,
                min_sister_um=8.0,
                max_sister_um=14.0,
                max_distance_asymmetry=0.60,
                min_divergence_gain_um=SAFE_DIV_DIVERGE_UM,
                max_per_frame=1,
                max_total=64,
                accept_candidate=_exp054_accept,
                stats=stats,
            )
        print(
            f"  [{dataset}] EXP054 mode={FOCUS_STRATEGY_MODE} "
            f"joint_saved={stats.get('exp054_joint_saved_proposals', 0)} "
            f"joint_candidates={stats.get('exp054_joint_structural_candidates', 0)} "
            f"joint_added={stats.get('exp054_joint_added', 0)} "
            f"joint_rewired={stats.get('exp054_joint_rewired', 0)}"
        )
'''


def _embedded_helper() -> str:
    source = HELPER.read_text(encoding="utf-8")
    start = source.index("def _exp054_position_um")
    return "\n# EXP054 embedded joint fork selector.\n" + source[start:] + "\n"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # Carry the deterministic EXP053 capture and frozen-weight runtime into a
    # new notebook, then replace only the post-ILP integration block.
    cell0 = "".join(cells[0]["source"]).replace("EXP053", "EXP054")
    cells[0]["source"] = cell0.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    helper_end = cell5.find("\ndef filter_output_graph(")
    if helper_end < 0:
        raise RuntimeError("EXP054 helper insertion anchor not found")
    cell5 = cell5[:helper_end] + _embedded_helper() + cell5[helper_end:]
    integration_start = cell5.find(
        '\n    if FOCUS_STRATEGY_MODE in {\n        "exp049_cos060",'
    )
    integration_end = cell5.find('\n\n    if FOCUS_STRATEGY_MODE in {"broad_rescue"', integration_start)
    if integration_start < 0 or integration_end < 0:
        raise RuntimeError("EXP054 integration block anchors not found")
    cell5 = cell5[:integration_start] + "\n" + INTEGRATION + cell5[integration_end:]
    cells[5]["source"] = cell5.splitlines(True)

    cell10 = "".join(cells[10]["source"])
    cell10 = _replace_once(
        cell10,
        'EXP053_MODES = ["off", "exp049_cos060", "exp053_top3_baseline", "exp053_top3_sister075", "exp053_top3_asym110", "exp053_top3_diverge175", "exp053_top3_joint"]',
        'EXP054_MODES = ["off", "exp049_cos060", "exp054_joint_top2", "exp054_joint_top3"]',
        "EXP054 mode list",
    )
    cell10 = cell10.replace("for _mode in EXP053_MODES:", "for _mode in EXP054_MODES:")
    cell10 = cell10.replace("EXP053 strategy:", "EXP054 strategy:")
    cell10 = cell10.replace(
        "EXP053 official CV complete; no second test pass and no submission.csv.",
        "EXP054 official CV complete; no second test pass and no submission.csv.",
    )
    for old, new in (
        ("EXP053_official_strategy_summary.csv", "EXP054_official_strategy_summary.csv"),
        ("EXP053_official_strategy_per_movie.csv", "EXP054_official_strategy_per_movie.csv"),
        ("EXP053_strategy_stats.csv", "EXP054_strategy_stats.csv"),
    ):
        cell10 = _replace_once(cell10, old, new, f"EXP054 output {old}")
    anchor = '        "exp052_rank3_cap_rejected": sum(int(row.get("exp052_rank3_cap_rejected", 0)) for row in _focus_stats),\n'
    if anchor in cell10:
        extra = anchor + ''.join(
            f'        "{key}": sum(int(row.get("{key}", 0)) for row in _focus_stats),\n'
            for key in (
                "exp054_joint_saved_proposals",
                "exp054_joint_structural_candidates",
                "exp054_joint_deepcenter_rejected",
                "exp054_joint_divergence_rejected",
                "exp054_joint_existing_fork_skipped",
                "exp054_joint_added",
                "exp054_joint_rewired",
                "exp054_joint_source_replaced",
                "exp054_joint_cap_rejected",
            )
        )
        cell10 = cell10.replace(anchor, extra, 1)
    cells[10]["source"] = cell10.splitlines(True)

    cell11 = "".join(cells[11]["source"]).replace("EXP053", "EXP054")
    cells[11]["source"] = cell11.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP054 ILP Joint Parent Daughter Division Official CV"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
