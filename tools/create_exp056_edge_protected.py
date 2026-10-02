"""Create EXP056: edge-protected EXP049 cos060 official-CV notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP049" / "CELL_post_ilp_division_proposal_cv.ipynb"
OUTPUT = ROOT / "EXP" / "EXP056" / "CELL_post_ilp_edge_protected_official_cv.ipynb"
HELPER = ROOT / "tools" / "exp049_post_ilp_division.py"


def _helper_source() -> str:
    source = HELPER.read_text(encoding="utf-8")
    source = source.replace('"""Precision-first post-ILP division proposals for EXP049."""\n\n', "")
    source = source.replace("from __future__ import annotations\n\n", "")

    signature = """    accept_candidate: Callable[[dict[str, object]], bool] | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict[str, object]]:
"""
    replacement = """    accept_candidate: Callable[[dict[str, object]], bool] | None = None,
    stats: dict[str, int] | None = None,
    edge_guard: bool = False,
    edge_guard_margin: float = 0.10,
    edge_guard_distance_slack_um: float = 1.5,
) -> list[dict[str, object]]:
"""
    if source.count(signature) != 1:
        raise RuntimeError("EXP056 helper signature anchor mismatch")
    source = source.replace(signature, replacement, 1)

    stats_anchor = """        "exp049_cap_rejected",
    ):
"""
    stats_replacement = """        "exp049_cap_rejected",
        "exp056_edge_guard_checked",
        "exp056_edge_guard_passed",
        "exp056_edge_guard_rejected",
        "exp056_edge_guard_missing_prob",
        "exp056_edge_guard_accepted",
    ):
"""
    if source.count(stats_anchor) != 1:
        raise RuntimeError("EXP056 helper stats anchor mismatch")
    source = source.replace(stats_anchor, stats_replacement, 1)

    guard_anchor = '        stats["exp049_structural_candidates"] += 1\n'
    guard_block = r'''        if edge_guard:
            stats["exp056_edge_guard_checked"] += 1
            _old_incoming = incoming.get(second_id, [])
            if len(_old_incoming) != 1:
                stats["exp056_edge_guard_rejected"] += 1
                continue
            _old_edge = _old_incoming[0]
            _old_prob_raw = _old_edge.get("edge_prob")
            try:
                _old_prob = float(_old_prob_raw)
            except (TypeError, ValueError):
                _old_prob = float("nan")
            if np.isfinite(_old_prob):
                # A fork may replace one incoming edge. Do that only when the
                # proposed second edge is not materially weaker.
                if p2 + float(edge_guard_margin) < _old_prob:
                    stats["exp056_edge_guard_rejected"] += 1
                    continue
            else:
                stats["exp056_edge_guard_missing_prob"] += 1
                _old_distance_raw = _old_edge.get("distance_um")
                try:
                    _old_distance = float(_old_distance_raw)
                except (TypeError, ValueError):
                    _old_distance = float("nan")
                if (
                    np.isfinite(_old_distance)
                    and d2 > _old_distance + float(edge_guard_distance_slack_um)
                ):
                    stats["exp056_edge_guard_rejected"] += 1
                    continue
            stats["exp056_edge_guard_passed"] += 1

'''
    if source.count(guard_anchor) != 1:
        raise RuntimeError("EXP056 helper guard anchor mismatch")
    source = source.replace(guard_anchor, guard_block + guard_anchor, 1)

    selected_anchor = """    stats["exp049_added"] += len(selected)
    return kept
"""
    selected_replacement = """    stats["exp049_added"] += len(selected)
    if edge_guard:
        stats["exp056_edge_guard_accepted"] += len(selected)
    return kept
"""
    if source.count(selected_anchor) != 1:
        raise RuntimeError("EXP056 helper selected anchor mismatch")
    return source.replace(selected_anchor, selected_replacement, 1)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cell0 = "".join(cells[0]["source"])
    cell0 += (
        "\n# EXP056: keep EXP049 cos060 as the 2/0/3 reference and test one\n"
        "# conservative edge-protection gate.\n"
        'os.environ["BIOHUB_EXP056_EDGE_MARGIN"] = "0.10"\n'
        'os.environ["BIOHUB_EXP056_EDGE_DISTANCE_SLACK_UM"] = "1.5"\n'
    )
    cells[0]["source"] = cell0.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    helper_start = cell5.find("# EXP049 embedded structural selector.")
    helper_end = cell5.find("\ndef filter_output_graph(", helper_start)
    if helper_start < 0 or helper_end < 0:
        raise RuntimeError("EXP056 helper replacement anchors not found")
    cell5 = (
        cell5[:helper_start]
        + "# EXP056 embedded edge-protected selector.\n"
        + _helper_source()
        + cell5[helper_end:]
    )

    condition_old = 'if FOCUS_STRATEGY_MODE.startswith("exp049_cos") and dataset is not None:'
    condition_new = '''if FOCUS_STRATEGY_MODE in {
        "exp049_cos090", "exp049_cos075", "exp049_cos060", "exp056_edge_protected"
    } and dataset is not None:'''
    if cell5.count(condition_old) != 1:
        raise RuntimeError("EXP056 integration condition anchor mismatch")
    cell5 = cell5.replace(condition_old, condition_new, 1)

    threshold_anchor = '            "exp049_cos060": -0.60,\n'
    threshold_replacement = threshold_anchor + '            "exp056_edge_protected": -0.60,\n'
    if cell5.count(threshold_anchor) != 1:
        raise RuntimeError("EXP056 threshold anchor mismatch")
    cell5 = cell5.replace(threshold_anchor, threshold_replacement, 1)

    call_anchor = """            accept_candidate=_exp049_accept,
            stats=stats,
"""
    call_replacement = """            accept_candidate=_exp049_accept,
            stats=stats,
            edge_guard=(FOCUS_STRATEGY_MODE == "exp056_edge_protected"),
            edge_guard_margin=float(os.environ.get("BIOHUB_EXP056_EDGE_MARGIN", "0.10")),
            edge_guard_distance_slack_um=float(
                os.environ.get("BIOHUB_EXP056_EDGE_DISTANCE_SLACK_UM", "1.5")
            ),
"""
    if cell5.count(call_anchor) != 1:
        raise RuntimeError("EXP056 helper call anchor mismatch")
    cell5 = cell5.replace(call_anchor, call_replacement, 1)
    cell5 = cell5.replace("EXP049 mode={FOCUS_STRATEGY_MODE}", "EXP056 mode={FOCUS_STRATEGY_MODE}", 1)
    cells[5]["source"] = cell5.splitlines(True)

    cell10 = "".join(cells[10]["source"])
    old_modes = 'EXP019_MODES = ["off", "exp049_cos090", "exp049_cos075", "exp049_cos060"]'
    new_modes = 'EXP056_MODES = ["off", "exp049_cos060", "exp056_edge_protected"]'
    if cell10.count(old_modes) != 1:
        raise RuntimeError("EXP056 mode list anchor mismatch")
    cell10 = cell10.replace(old_modes, new_modes, 1)
    cell10 = cell10.replace("for _mode in EXP019_MODES:", "for _mode in EXP056_MODES:", 1)
    cell10 = cell10.replace("EXP019 strategy:", "EXP056 strategy:", 1)
    for old, new in (
        ("EXP049_official_strategy_summary.csv", "EXP056_official_strategy_summary.csv"),
        ("EXP049_official_strategy_per_movie.csv", "EXP056_official_strategy_per_movie.csv"),
        ("EXP049_strategy_stats.csv", "EXP056_strategy_stats.csv"),
    ):
        if cell10.count(old) != 1:
            raise RuntimeError(f"EXP056 output anchor mismatch: {old}")
        cell10 = cell10.replace(old, new, 1)
    stats_anchor = '''        "exp049_cap_rejected": sum(int(row.get("exp049_cap_rejected", 0)) for row in _focus_stats),
    })'''
    stats_replacement = '''        "exp049_cap_rejected": sum(int(row.get("exp049_cap_rejected", 0)) for row in _focus_stats),
        "exp056_edge_guard_checked": sum(int(row.get("exp056_edge_guard_checked", 0)) for row in _focus_stats),
        "exp056_edge_guard_passed": sum(int(row.get("exp056_edge_guard_passed", 0)) for row in _focus_stats),
        "exp056_edge_guard_rejected": sum(int(row.get("exp056_edge_guard_rejected", 0)) for row in _focus_stats),
        "exp056_edge_guard_missing_prob": sum(int(row.get("exp056_edge_guard_missing_prob", 0)) for row in _focus_stats),
        "exp056_edge_guard_accepted": sum(int(row.get("exp056_edge_guard_accepted", 0)) for row in _focus_stats),
    })'''
    if cell10.count(stats_anchor) != 1:
        raise RuntimeError("EXP056 summary anchor mismatch")
    cell10 = cell10.replace(stats_anchor, stats_replacement, 1)
    cell10 = cell10.replace(
        "# This is a diagnostic comparison, not a competition submission.",
        "# EXP056 is an official-CV comparison, not a competition submission.",
        1,
    )
    cell10 = cell10.replace("EXP019 arm", "EXP056 arm", 1)
    cell10 = cell10.replace(
        "EXP019 official CV complete; no second test pass and no submission.csv.",
        "EXP056 official CV complete; no test submission is generated.",
        1,
    )
    cells[10]["source"] = cell10.splitlines(True)

    cell11 = "".join(cells[11]["source"])
    cell11 = cell11.replace("EXP049", "EXP056", 1)
    cell11 = cell11.replace("EXP019", "EXP056")
    cells[11]["source"] = cell11.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP056 Edge Protected cos060 Official CV"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

