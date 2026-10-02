"""Create the EXP052 official CV notebook from the deterministic EXP049 graph."""

from __future__ import annotations

import json
from pathlib import Path

from create_exp049_post_ilp_division import RUNTIME_PATCH as EXP049_RUNTIME_PATCH


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP049" / "CELL_post_ilp_division_proposal_cv.ipynb"
OUTPUT = ROOT / "EXP" / "EXP052" / "CELL_post_ilp_division_top3_cv.ipynb"
EXP049_HELPER = ROOT / "tools" / "exp049_post_ilp_division.py"
EXP052_HELPER = ROOT / "tools" / "exp052_post_ilp_division.py"


def top3_runtime_patch() -> str:
    """Change EXP049 capture to write two compatible five-column arrays."""
    patch = EXP049_RUNTIME_PATCH.replace("EXP049", "EXP052").replace("exp049", "exp052")
    old_decl = (
        '    "    _exp052_division_proposals: list[tuple[int, int, int, float, float]] = []\\n",\n'
    )
    new_decl = (
        '    "    _exp052_second_proposals: list[tuple[int, int, int, float, float]] = []\\n"\n'
        '    "    _exp052_rank3_proposals: list[tuple[int, int, int, float, float]] = []\\n",\n'
    )
    if patch.count(old_decl) != 1:
        raise RuntimeError("EXP052 declaration anchor mismatch")
    patch = patch.replace(old_decl, new_decl, 1)

    old_capture = (
        '            if os.environ.get("BIOHUB_EXP052_CAPTURE", "0") != "0" and n_tgt >= 2:\n'
        "                for _exp052_i in range(n_src):\n"
        '                    _exp052_order = np.argsort(-probs[_exp052_i], kind="stable")\n'
        "                    _exp052_j1 = int(_exp052_order[0])\n"
        "                    _exp052_j2 = int(_exp052_order[1])\n"
        "                    _exp052_division_proposals.append((\n"
        "                        int(idx_src[_exp052_i]),\n"
        "                        int(idx_tgt[_exp052_j1]),\n"
        "                        int(idx_tgt[_exp052_j2]),\n"
        "                        float(probs[_exp052_i, _exp052_j1]),\n"
        "                        float(probs[_exp052_i, _exp052_j2]),\n"
        "                    ))\n"
    )
    new_capture = (
        '            if os.environ.get("BIOHUB_EXP052_CAPTURE", "0") != "0" and n_tgt >= 2:\n'
        "                for _exp052_i in range(n_src):\n"
        '                    _exp052_order = np.argsort(-probs[_exp052_i], kind="stable")\n'
        "                    _exp052_j1 = int(_exp052_order[0])\n"
        "                    _exp052_j2 = int(_exp052_order[1])\n"
        "                    _exp052_second_proposals.append((\n"
        "                        int(idx_src[_exp052_i]),\n"
        "                        int(idx_tgt[_exp052_j1]),\n"
        "                        int(idx_tgt[_exp052_j2]),\n"
        "                        float(probs[_exp052_i, _exp052_j1]),\n"
        "                        float(probs[_exp052_i, _exp052_j2]),\n"
        "                    ))\n"
        "                    if n_tgt >= 3:\n"
        "                        _exp052_j3 = int(_exp052_order[2])\n"
        "                        _exp052_rank3_proposals.append((\n"
        "                            int(idx_src[_exp052_i]),\n"
        "                            int(idx_tgt[_exp052_j1]),\n"
        "                            int(idx_tgt[_exp052_j3]),\n"
        "                            float(probs[_exp052_i, _exp052_j1]),\n"
        "                            float(probs[_exp052_i, _exp052_j3]),\n"
        "                        ))\n"
    )
    if patch.count(old_capture) != 1:
        raise RuntimeError("EXP052 capture anchor mismatch")
    patch = patch.replace(old_capture, new_capture, 1)

    old_return = (
        '    if os.environ.get("BIOHUB_EXP052_CAPTURE", "0") != "0":\n'
        "        _exp052_dir = Path(os.environ.get(\n"
        '            "BIOHUB_EXP052_PROPOSAL_DIR", "/kaggle/working/exp052_candidates"\n'
        "        ))\n"
        "        _exp052_dir.mkdir(parents=True, exist_ok=True)\n"
        "        _exp052_array = np.asarray(_exp052_division_proposals, dtype=np.float64).reshape((-1, 5))\n"
        "        np.savez_compressed(_exp052_dir / f\"{ds_path.stem}.npz\", proposals=_exp052_array)\n"
        '        print("EXP052 proposals", ds_path.stem, len(_exp052_array), flush=True)\n'
        "\n"
        "    return coords, all_edges\n"
    )
    new_return = (
        '    if os.environ.get("BIOHUB_EXP052_CAPTURE", "0") != "0":\n'
        "        _exp052_dir = Path(os.environ.get(\n"
        '            "BIOHUB_EXP052_PROPOSAL_DIR", "/kaggle/working/exp052_candidates"\n'
        "        ))\n"
        "        _exp052_dir.mkdir(parents=True, exist_ok=True)\n"
        "        _exp052_second = np.asarray(_exp052_second_proposals, dtype=np.float64).reshape((-1, 5))\n"
        "        _exp052_rank3 = np.asarray(_exp052_rank3_proposals, dtype=np.float64).reshape((-1, 5))\n"
        "        np.savez_compressed(\n"
        "            _exp052_dir / f\"{ds_path.stem}.npz\",\n"
        "            proposals=_exp052_second,\n"
        "            proposals_rank3=_exp052_rank3,\n"
        "        )\n"
        "        print(\n"
        '            "EXP052 proposals", ds_path.stem,\n'
        '            "rank2=", len(_exp052_second),\n'
        '            "rank3=", len(_exp052_rank3),\n'
        "            flush=True,\n"
        "        )\n"
        "\n"
        "    return coords, all_edges\n"
    )
    if patch.count(old_return) != 1:
        raise RuntimeError("EXP052 return anchor mismatch")
    return patch.replace(old_return, new_return, 1)


def embedded_helpers() -> str:
    base = EXP049_HELPER.read_text(encoding="utf-8")
    base = base.replace('"""Precision-first post-ILP division proposals for EXP049."""\n\n', "")
    base = base.replace("from __future__ import annotations\n\n", "")
    extra = EXP052_HELPER.read_text(encoding="utf-8")
    extra = extra[extra.index("def apply_exp052_top3_division_proposals") :]
    return "\n# EXP052 embedded selectors (EXP049 Top-2 plus delayed rank-3).\n" + base + "\n" + extra + "\n"


INTEGRATION = r'''
    if FOCUS_STRATEGY_MODE in {"exp049_cos060", "exp052_top3_cos060"} and dataset is not None:
        _exp052_path = Path(os.environ.get(
            "BIOHUB_EXP052_PROPOSAL_DIR", "/kaggle/working/exp052_candidates"
        )) / f"{dataset}.npz"
        if not _exp052_path.exists():
            raise FileNotFoundError(_exp052_path)
        with np.load(_exp052_path) as _exp052_payload:
            _exp052_rows = np.asarray(_exp052_payload["proposals"], dtype=np.float64)
            _exp052_rank3_rows = np.asarray(
                _exp052_payload.get("proposals_rank3", np.empty((0, 5))),
                dtype=np.float64,
            )
        _exp052_threshold = -0.60

        def _exp052_accept(_node):
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
                nodes_by_id, edges, _exp052_rows,
                cosine_max=_exp052_threshold,
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
                accept_candidate=_exp052_accept,
                stats=stats,
            )
        else:
            edges = apply_exp052_top3_division_proposals(
                nodes_by_id, edges, _exp052_rows, _exp052_rank3_rows,
                cosine_max=_exp052_threshold,
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
                accept_candidate=_exp052_accept,
                stats=stats,
            )
        print(
            f"  [{dataset}] EXP052 mode={FOCUS_STRATEGY_MODE} "
            f"top2_added={stats.get('exp049_added', 0)} "
            f"rank3_candidates={stats.get('exp052_rank3_structural_candidates', 0)} "
            f"rank3_added={stats.get('exp052_rank3_added', 0)} "
            f"rank3_rewired={stats.get('exp052_rank3_rewired', 0)}"
        )
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cell0 = "".join(cells[0]["source"])
    old_env = (
        "# EXP049 captures low-score second edges for post-ILP structural testing.\n"
        'os.environ["BIOHUB_EXP049_CAPTURE"] = "1"\n'
        'os.environ["BIOHUB_EXP049_PROPOSAL_DIR"] = "/kaggle/working/exp049_candidates"\n'
    )
    new_env = (
        "# EXP052 captures rank-2 and rank-3 edges for delayed post-ILP testing.\n"
        'os.environ["BIOHUB_EXP052_CAPTURE"] = "1"\n'
        'os.environ["BIOHUB_EXP052_PROPOSAL_DIR"] = "/kaggle/working/exp052_candidates"\n'
    )
    if cell0.count(old_env) != 1:
        raise RuntimeError("EXP052 cell 0 environment anchor mismatch")
    cells[0]["source"] = cell0.replace(old_env, new_env, 1).splitlines(True)

    cell4 = "".join(cells[4]["source"])
    start = cell4.find("# EXP049: save the model's primary and second edge proposals")
    end = cell4.find("\ndef list_test_stems", start)
    if start < 0 or end < 0:
        raise RuntimeError("EXP052 cell 4 runtime patch anchor mismatch")
    cell4 = cell4[:start] + top3_runtime_patch().lstrip("\n") + cell4[end:]
    cells[4]["source"] = cell4.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    helper_start = cell5.find("# EXP049 embedded structural selector.")
    helper_end = cell5.find("\ndef filter_output_graph(", helper_start)
    if helper_start < 0 or helper_end < 0:
        raise RuntimeError("EXP052 helper block anchor mismatch")
    cell5 = cell5[:helper_start] + embedded_helpers() + cell5[helper_end:]
    integration_start = cell5.find('\n    if FOCUS_STRATEGY_MODE.startswith("exp049_cos")')
    integration_end = cell5.find('\n\n    if FOCUS_STRATEGY_MODE in {"broad_rescue"', integration_start)
    if integration_start < 0 or integration_end < 0:
        raise RuntimeError("EXP052 integration block anchor mismatch")
    cell5 = cell5[:integration_start] + "\n" + INTEGRATION + cell5[integration_end:]
    cells[5]["source"] = cell5.splitlines(True)

    cell10 = "".join(cells[10]["source"])
    old_modes = 'EXP019_MODES = ["off", "exp049_cos090", "exp049_cos075", "exp049_cos060"]'
    new_modes = 'EXP052_MODES = ["off", "exp049_cos060", "exp052_top3_cos060"]'
    if cell10.count(old_modes) != 1:
        raise RuntimeError("EXP052 mode list anchor mismatch")
    cell10 = cell10.replace(old_modes, new_modes, 1)
    cell10 = cell10.replace("for _mode in EXP019_MODES:", "for _mode in EXP052_MODES:", 1)
    cell10 = cell10.replace("EXP049_official_strategy_summary.csv", "EXP052_official_strategy_summary.csv")
    cell10 = cell10.replace("EXP049_official_strategy_per_movie.csv", "EXP052_official_strategy_per_movie.csv")
    cell10 = cell10.replace("EXP049_strategy_stats.csv", "EXP052_strategy_stats.csv")
    summary_anchor = '        "exp049_cap_rejected": sum(int(row.get("exp049_cap_rejected", 0)) for row in _focus_stats),\n'
    summary_extra = (
        summary_anchor
        + '        "exp052_rank3_saved_proposals": sum(int(row.get("exp052_rank3_saved_proposals", 0)) for row in _focus_stats),\n'
        + '        "exp052_rank3_frame_blocked": sum(int(row.get("exp052_rank3_frame_blocked", 0)) for row in _focus_stats),\n'
        + '        "exp052_rank3_structural_candidates": sum(int(row.get("exp052_rank3_structural_candidates", 0)) for row in _focus_stats),\n'
        + '        "exp052_rank3_deepcenter_rejected": sum(int(row.get("exp052_rank3_deepcenter_rejected", 0)) for row in _focus_stats),\n'
        + '        "exp052_rank3_divergence_rejected": sum(int(row.get("exp052_rank3_divergence_rejected", 0)) for row in _focus_stats),\n'
        + '        "exp052_rank3_added": sum(int(row.get("exp052_rank3_added", 0)) for row in _focus_stats),\n'
        + '        "exp052_rank3_rewired": sum(int(row.get("exp052_rank3_rewired", 0)) for row in _focus_stats),\n'
        + '        "exp052_rank3_cap_rejected": sum(int(row.get("exp052_rank3_cap_rejected", 0)) for row in _focus_stats),\n'
    )
    if cell10.count(summary_anchor) != 1:
        raise RuntimeError("EXP052 summary anchor mismatch")
    cell10 = cell10.replace(summary_anchor, summary_extra, 1)
    cells[10]["source"] = cell10.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP052 Post-ILP Division Top-3 Official CV"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
