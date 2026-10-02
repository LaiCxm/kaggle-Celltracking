"""Create EXP046: targeted second-edge division recall on official CV."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP019" / "CELL_focus3d_strategy_official_cv.ipynb"
OUTPUT = ROOT / "EXP" / "EXP046" / "CELL_division_second_edge_rescue_official_cv.ipynb"


def rescue_patch() -> str:
    old = """            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )"""
    new = """            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )

            # EXP046: preserve the production threshold for ordinary edges,
            # but expose a narrowly gated second daughter edge for division
            # recall.  This is a candidate-generation diagnostic, not a global
            # threshold change: only the source's second-ranked target is added
            # when its first edge is strong and the physical displacement is
            # still within a division-sized radius.
            _division_rescue_threshold = float(os.environ.get(
                "BIOHUB_DIVISION_RESCUE_SECOND_THRESHOLD", "0.18"
            ))
            _division_rescue_top1_min = float(os.environ.get(
                "BIOHUB_DIVISION_RESCUE_TOP1_MIN", "0.60"
            ))
            _division_rescue_max_um = float(os.environ.get(
                "BIOHUB_DIVISION_RESCUE_MAX_UM", "14.0"
            ))
            _division_rescue_pairs = []
            for _i in range(n_src):
                if n_tgt < 2:
                    break
                _order = np.argsort(-probs[_i], kind="stable")
                _j1, _j2 = int(_order[0]), int(_order[1])
                _p1, _p2 = float(probs[_i, _j1]), float(probs[_i, _j2])
                if _p1 < _division_rescue_top1_min or _p2 < _division_rescue_threshold:
                    continue
                _gi = int(idx_src[_i])
                _gj = int(idx_tgt[_j2])
                _delta_um = (
                    coords_so_far[_gi, 1:].astype(np.float32)
                    - coords_so_far[_gj, 1:].astype(np.float32)
                ) * ds_arr
                _distance_um = float(np.linalg.norm(_delta_um))
                if _distance_um > _division_rescue_max_um:
                    continue
                if any(int(_row[1]) == _i and int(_row[2]) == _j2 for _row in candidates):
                    continue
                _division_rescue_pairs.append((_p2, _i, _j2))

            # Keep these edges behind ordinary model-qualified edges.  They
            # consume the same source/target degree budgets, so the experiment
            # cannot create unconstrained graph branches by itself.
            candidates.extend(_division_rescue_pairs)
            candidates.sort(reverse=True)"""
    return f'''\n# EXP046 targeted division second-edge rescue patch\nimport numpy as _exp046_np\n_exp046_old = {old!r}\n_exp046_new = {new!r}\nif _s.count(_exp046_old) != 1:\n    raise RuntimeError(\"EXP046 candidate-generation anchor mismatch\")\n_s = _s.replace(_exp046_old, _exp046_new, 1)\ncompile(_s, str(_ps), \"exec\")\n_ps.write_text(_s)\nprint(\"EXP046 second-edge division rescue patch applied\")\n'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # Preserve EXP019's complete configuration and guard.  EXP046 adds only
    # the three candidate-rescue knobs after the baseline environment setup.
    cell0 = "".join(cells[0]["source"])
    cell0 += (
        "\n# EXP046: targeted second-edge division recall on official patched CV.\n"
        "# Ordinary threshold and all baseline configuration values remain unchanged.\n"
        'os.environ["BIOHUB_DIVISION_RESCUE_SECOND_THRESHOLD"] = "0.18"\n'
        'os.environ["BIOHUB_DIVISION_RESCUE_TOP1_MIN"] = "0.60"\n'
        'os.environ["BIOHUB_DIVISION_RESCUE_MAX_UM"] = "14.0"\n'
    )
    cells[0]["source"] = cell0.splitlines(True)
    # The source notebook compares several FOCUS3D modes.  EXP046 isolates the
    # candidate-generation change with the unchanged FOCUS-off graph arm.
    cell9 = "".join(cells[10]["source"])
    cell9 = cell9.replace(
        'EXP019_MODES = [\n    "off",\n    "broad_rescue",\n    "selected_rescue",\n    "veto_only",\n    "hybrid_selected",\n]',
        'EXP019_MODES = ["off"]',
    )
    cells[10]["source"] = cell9.splitlines(True)

    cell4 = "".join(cells[4]["source"])
    marker = "\ndef list_test_stems()"
    index = cell4.find(marker)
    if index < 0:
        raise RuntimeError("EXP046 could not find runtime patch insertion point")
    cell4 = cell4[:index] + rescue_patch() + cell4[index:]
    cells[4]["source"] = cell4.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP046 Targeted Division Second-Edge Rescue Official CV"
    )
    cells[0]["metadata"] = {}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
