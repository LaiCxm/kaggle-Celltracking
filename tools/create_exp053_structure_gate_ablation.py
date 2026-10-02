"""Create the EXP053 official-CV structural-gate ablation notebook.

EXP053 reuses EXP052's deterministic Top-2/Top-3 proposal capture and delayed
promotion.  The only algorithmic change between its Top-3 arms is one of the
three post-ILP structural gates: minimum sister distance, maximum distance
asymmetry, or minimum next-frame divergence gain.  The joint arm changes all
three together as a pre-registered diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP052" / "CELL_post_ilp_division_top3_cv.ipynb"
OUTPUT = ROOT / "EXP" / "EXP053" / "CELL_post_ilp_division_gate_ablation_cv.ipynb"


GATE_TABLE = {
    "exp053_top3_baseline": {"min_sister_um": 8.0, "max_distance_asymmetry": 0.60, "min_divergence_gain_um": 2.25},
    "exp053_top3_sister075": {"min_sister_um": 7.5, "max_distance_asymmetry": 0.60, "min_divergence_gain_um": 2.25},
    "exp053_top3_asym110": {"min_sister_um": 8.0, "max_distance_asymmetry": 1.10, "min_divergence_gain_um": 2.25},
    "exp053_top3_diverge175": {"min_sister_um": 8.0, "max_distance_asymmetry": 0.60, "min_divergence_gain_um": 1.75},
    "exp053_top3_joint": {"min_sister_um": 7.5, "max_distance_asymmetry": 1.10, "min_divergence_gain_um": 1.75},
}


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def _exp053_integration() -> str:
    return r'''
    if FOCUS_STRATEGY_MODE in {
        "exp049_cos060",
        "exp053_top3_baseline",
        "exp053_top3_sister075",
        "exp053_top3_asym110",
        "exp053_top3_diverge175",
        "exp053_top3_joint",
    } and dataset is not None:
        _exp053_path = Path(os.environ.get(
            "BIOHUB_EXP052_PROPOSAL_DIR", "/kaggle/working/exp052_candidates"
        )) / f"{dataset}.npz"
        if not _exp053_path.exists():
            raise FileNotFoundError(_exp053_path)
        with np.load(_exp053_path) as _exp053_payload:
            _exp053_rows = np.asarray(_exp053_payload["proposals"], dtype=np.float64)
            _exp053_rank3_rows = np.asarray(
                _exp053_payload.get("proposals_rank3", np.empty((0, 5))),
                dtype=np.float64,
            )
        _exp053_threshold = -0.60
        _exp053_gate_table = {
            "exp053_top3_baseline": (8.0, 0.60, 2.25),
            "exp053_top3_sister075": (7.5, 0.60, 2.25),
            "exp053_top3_asym110": (8.0, 1.10, 2.25),
            "exp053_top3_diverge175": (8.0, 0.60, 1.75),
            "exp053_top3_joint": (7.5, 1.10, 1.75),
        }
        if FOCUS_STRATEGY_MODE == "exp049_cos060":
            _exp053_min_sister, _exp053_max_asym, _exp053_min_diverge = (8.0, 0.60, 2.25)
        else:
            try:
                _exp053_min_sister, _exp053_max_asym, _exp053_min_diverge = _exp053_gate_table[
                    FOCUS_STRATEGY_MODE
                ]
            except KeyError as exc:
                raise RuntimeError(f"Unknown EXP053 gate arm: {FOCUS_STRATEGY_MODE}") from exc

        def _exp053_accept(_node):
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
                nodes_by_id, edges, _exp053_rows,
                cosine_max=_exp053_threshold,
                scale_um=VOXEL_SCALE_UM,
                min_primary_probability=0.60,
                min_second_probability=0.18,
                max_parent_um=7.0,
                min_sister_um=_exp053_min_sister,
                max_sister_um=14.0,
                max_distance_asymmetry=_exp053_max_asym,
                min_divergence_gain_um=_exp053_min_diverge,
                max_per_frame=1,
                max_total=64,
                accept_candidate=_exp053_accept,
                stats=stats,
            )
        else:
            edges = apply_exp052_top3_division_proposals(
                nodes_by_id, edges, _exp053_rows, _exp053_rank3_rows,
                cosine_max=_exp053_threshold,
                scale_um=VOXEL_SCALE_UM,
                min_primary_probability=0.60,
                min_second_probability=0.18,
                max_parent_um=7.0,
                min_sister_um=_exp053_min_sister,
                max_sister_um=14.0,
                max_distance_asymmetry=_exp053_max_asym,
                min_divergence_gain_um=_exp053_min_diverge,
                max_per_frame=1,
                max_total=64,
                accept_candidate=_exp053_accept,
                stats=stats,
            )
        print(
            f"  [{dataset}] EXP053 mode={FOCUS_STRATEGY_MODE} "
            f"min_sister={_exp053_min_sister:.2f} "
            f"max_asym={_exp053_max_asym:.2f} "
            f"min_diverge={_exp053_min_diverge:.2f} "
            f"top2_added={stats.get('exp049_added', 0)} "
            f"rank3_candidates={stats.get('exp052_rank3_structural_candidates', 0)} "
            f"rank3_added={stats.get('exp052_rank3_added', 0)} "
            f"rank3_rewired={stats.get('exp052_rank3_rewired', 0)}"
        )
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # Keep EXP052's capture variable and embedded runtime patch byte-compatible;
    # only the official-CV arm selection and result names change here.
    cell0 = "".join(cells[0]["source"])
    cell0 = _replace_once(
        cell0,
        "# EXP052 captures rank-2 and rank-3 edges for delayed post-ILP testing.\n",
        "# EXP053 captures rank-2 and rank-3 edges for structural-gate ablation.\n",
        "EXP053 cell 0 comment",
    )
    cells[0]["source"] = cell0.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    start = cell5.find(
        '\n    if FOCUS_STRATEGY_MODE in {"exp049_cos060", "exp052_top3_cos060"} and dataset is not None:'
    )
    end = cell5.find('\n\n    if FOCUS_STRATEGY_MODE in {"broad_rescue"', start)
    if start < 0 or end < 0:
        raise RuntimeError("EXP053 integration block anchors not found")
    cell5 = cell5[:start] + "\n" + _exp053_integration() + cell5[end:]
    cells[5]["source"] = cell5.splitlines(True)

    cell10 = "".join(cells[10]["source"])
    cell10 = _replace_once(
        cell10,
        'EXP052_MODES = ["off", "exp049_cos060", "exp052_top3_cos060"]',
        'EXP053_MODES = ["off", "exp049_cos060", "exp053_top3_baseline", "exp053_top3_sister075", "exp053_top3_asym110", "exp053_top3_diverge175", "exp053_top3_joint"]',
        "EXP053 mode list",
    )
    cell10 = _replace_once(
        cell10,
        "for _mode in EXP052_MODES:",
        "for _mode in EXP053_MODES:",
        "EXP053 mode loop",
    )
    for old, new in (
        ("EXP052_official_strategy_summary.csv", "EXP053_official_strategy_summary.csv"),
        ("EXP052_official_strategy_per_movie.csv", "EXP053_official_strategy_per_movie.csv"),
        ("EXP052_strategy_stats.csv", "EXP053_strategy_stats.csv"),
    ):
        cell10 = _replace_once(cell10, old, new, f"EXP053 output {old}")
    cell10 = cell10.replace("EXP019 strategy:", "EXP053 strategy:")
    cell10 = cell10.replace(
        "EXP019 official CV complete; no second test pass and no submission.csv.",
        "EXP053 official CV complete; no second test pass and no submission.csv.",
    )
    cells[10]["source"] = cell10.splitlines(True)

    # The inherited manifest cell only prints resolved configuration.  Rename
    # its legacy EXP019 label as well, otherwise a completed CV run fails at
    # the final print with an undefined variable.
    cell11 = "".join(cells[11]["source"])
    cell11 = _replace_once(
        cell11,
        'print("EXP019 arms:", EXP019_MODES)',
        'print("EXP053 arms:", EXP053_MODES)',
        "EXP053 manifest mode list",
    )
    cell11 = cell11.replace(
        "EXP019 design: same four complete movies and raw predictions across all arms",
        "EXP053 design: same four complete movies and raw predictions across all arms",
    )
    cell11 = cell11.replace(
        "EXP019 formal metric: patched official scorer; legacy proxy not used for ranking",
        "EXP053 formal metric: patched official scorer; legacy proxy not used for ranking",
    )
    cell11 = cell11.replace(
        "EXP019 mode: official-CV-only validation; test submission generation is intentionally absent.",
        "EXP053 mode: official-CV-only validation; test submission generation is intentionally absent.",
    )
    cells[11]["source"] = cell11.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP053 Post-ILP Division Structural-Gate Ablation Official CV"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
