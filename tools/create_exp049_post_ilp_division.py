"""Create EXP049: deterministic post-ILP division proposal sweep."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP048" / "CELL_deterministic_official_cv_audit.ipynb"
OUTPUT = ROOT / "EXP" / "EXP049" / "CELL_post_ilp_division_proposal_cv.ipynb"
HELPER = ROOT / "tools" / "exp049_post_ilp_division.py"


RUNTIME_PATCH = r"""
# EXP049: save the model's primary and second edge proposals without exposing
# the low-probability edge to ILP continuation competition.
_exp049_runtime = _ps.read_text()
_exp049_replacements = []
_exp049_replacements.append((
    "    all_edges: list[tuple[int, int, float, float]] = []\n",
    "    all_edges: list[tuple[int, int, float, float]] = []\n"
    "    _exp049_division_proposals: list[tuple[int, int, int, float, float]] = []\n",
))
_exp049_prob_anchor = '''            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted('''
_exp049_prob_replacement = '''            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            if os.environ.get("BIOHUB_EXP049_CAPTURE", "0") != "0" and n_tgt >= 2:
                for _exp049_i in range(n_src):
                    _exp049_order = np.argsort(-probs[_exp049_i], kind="stable")
                    _exp049_j1 = int(_exp049_order[0])
                    _exp049_j2 = int(_exp049_order[1])
                    _exp049_division_proposals.append((
                        int(idx_src[_exp049_i]),
                        int(idx_tgt[_exp049_j1]),
                        int(idx_tgt[_exp049_j2]),
                        float(probs[_exp049_i, _exp049_j1]),
                        float(probs[_exp049_i, _exp049_j2]),
                    ))

            candidates = sorted('''
_exp049_replacements.append((_exp049_prob_anchor, _exp049_prob_replacement))
_exp049_return_anchor = "    return coords, all_edges\n"
_exp049_return_replacement = '''    if os.environ.get("BIOHUB_EXP049_CAPTURE", "0") != "0":
        _exp049_dir = Path(os.environ.get(
            "BIOHUB_EXP049_PROPOSAL_DIR", "/kaggle/working/exp049_candidates"
        ))
        _exp049_dir.mkdir(parents=True, exist_ok=True)
        _exp049_array = np.asarray(_exp049_division_proposals, dtype=np.float64).reshape((-1, 5))
        np.savez_compressed(_exp049_dir / f"{ds_path.stem}.npz", proposals=_exp049_array)
        print("EXP049 proposals", ds_path.stem, len(_exp049_array), flush=True)

    return coords, all_edges
'''
_exp049_replacements.append((_exp049_return_anchor, _exp049_return_replacement))
for _exp049_old, _exp049_new in _exp049_replacements:
    if _exp049_runtime.count(_exp049_old) != 1:
        raise RuntimeError(
            "EXP049 runtime anchor mismatch: "
            + str(_exp049_runtime.count(_exp049_old))
        )
    _exp049_runtime = _exp049_runtime.replace(_exp049_old, _exp049_new, 1)
compile(_exp049_runtime, str(_ps), "exec")
_ps.write_text(_exp049_runtime)
print("EXP049 proposal capture patch applied")
"""


INTEGRATION = r'''
    if FOCUS_STRATEGY_MODE.startswith("exp049_cos") and dataset is not None:
        _exp049_thresholds = {
            "exp049_cos090": -0.90,
            "exp049_cos075": -0.75,
            "exp049_cos060": -0.60,
        }
        _exp049_path = Path(os.environ.get(
            "BIOHUB_EXP049_PROPOSAL_DIR", "/kaggle/working/exp049_candidates"
        )) / f"{dataset}.npz"
        if not _exp049_path.exists():
            raise FileNotFoundError(_exp049_path)
        with np.load(_exp049_path) as _exp049_payload:
            _exp049_rows = np.asarray(_exp049_payload["proposals"], dtype=np.float64)

        def _exp049_accept(_node):
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

        edges = apply_post_ilp_division_proposals(
            nodes_by_id,
            edges,
            _exp049_rows,
            cosine_max=_exp049_thresholds[FOCUS_STRATEGY_MODE],
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
            accept_candidate=_exp049_accept,
            stats=stats,
        )
        print(
            f"  [{dataset}] EXP049 mode={FOCUS_STRATEGY_MODE} "
            f"structural={stats.get('exp049_structural_candidates', 0)} "
            f"accepted={stats.get('exp049_added', 0)} "
            f"rewired={stats.get('exp049_rewired', 0)}"
        )
'''


def helper_source() -> str:
    source = HELPER.read_text(encoding="utf-8")
    source = source.replace('"""Precision-first post-ILP division proposals for EXP049."""\n\n', "")
    source = source.replace("from __future__ import annotations\n\n", "")
    return "\n# EXP049 embedded structural selector.\n" + source + "\n"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cell0 = "".join(cells[0]["source"])
    cell0 += (
        "\n# EXP049 captures low-score second edges for post-ILP structural testing.\n"
        'os.environ["BIOHUB_EXP049_CAPTURE"] = "1"\n'
        'os.environ["BIOHUB_EXP049_PROPOSAL_DIR"] = "/kaggle/working/exp049_candidates"\n'
    )
    cells[0]["source"] = cell0.splitlines(True)

    cell4 = "".join(cells[4]["source"])
    marker = "\ndef list_test_stems()"
    index = cell4.find(marker)
    if index < 0:
        raise RuntimeError("EXP049 runtime insertion point not found")
    cell4 = cell4[:index] + RUNTIME_PATCH + cell4[index:]
    cells[4]["source"] = cell4.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    function_marker = "\ndef filter_output_graph("
    function_index = cell5.find(function_marker)
    if function_index < 0:
        raise RuntimeError("EXP049 filter function marker not found")
    cell5 = cell5[:function_index] + helper_source() + cell5[function_index:]

    safe_anchor = """    edges = add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
"""
    if cell5.count(safe_anchor) != 1:
        raise RuntimeError("EXP049 safe-division integration anchor mismatch")
    cell5 = cell5.replace(safe_anchor, safe_anchor + INTEGRATION, 1)
    cell5 = cell5.replace(
        'if FOCUS_STRATEGY_MODE != "off" and dataset is not None and edges:',
        'if FOCUS_STRATEGY_MODE in {"broad_rescue", "selected_rescue", "veto_only", "hybrid_selected"} and dataset is not None and edges:',
        1,
    )
    cells[5]["source"] = cell5.splitlines(True)

    cell10 = "".join(cells[10]["source"])
    if 'EXP019_MODES = ["off"]' not in cell10:
        raise RuntimeError("EXP049 mode list anchor not found")
    cell10 = cell10.replace(
        'EXP019_MODES = ["off"]',
        'EXP019_MODES = ["off", "exp049_cos090", "exp049_cos075", "exp049_cos060"]',
        1,
    )
    cell10 = cell10.replace("EXP019_official_strategy_summary.csv", "EXP049_official_strategy_summary.csv")
    cell10 = cell10.replace("EXP019_official_strategy_per_movie.csv", "EXP049_official_strategy_per_movie.csv")
    cell10 = cell10.replace("EXP019_strategy_stats.csv", "EXP049_strategy_stats.csv")
    summary_anchor = '''        "focus_frames_loaded": sum(int(row.get("focus_frames_loaded", 0)) for row in _focus_stats),
    })'''
    summary_replacement = '''        "focus_frames_loaded": sum(int(row.get("focus_frames_loaded", 0)) for row in _focus_stats),
        "exp049_structural_candidates": sum(int(row.get("exp049_structural_candidates", 0)) for row in _focus_stats),
        "exp049_deepcenter_rejected": sum(int(row.get("exp049_deepcenter_rejected", 0)) for row in _focus_stats),
        "exp049_divergence_rejected": sum(int(row.get("exp049_divergence_rejected", 0)) for row in _focus_stats),
        "exp049_added": sum(int(row.get("exp049_added", 0)) for row in _focus_stats),
        "exp049_rewired": sum(int(row.get("exp049_rewired", 0)) for row in _focus_stats),
        "exp049_cap_rejected": sum(int(row.get("exp049_cap_rejected", 0)) for row in _focus_stats),
    })'''
    if cell10.count(summary_anchor) != 1:
        raise RuntimeError("EXP049 summary anchor mismatch")
    cell10 = cell10.replace(summary_anchor, summary_replacement, 1)
    cells[10]["source"] = cell10.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP049 Post-ILP Division Proposal Official CV"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
