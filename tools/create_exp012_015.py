from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP010" / "CELL_infer_public_0942.ipynb"
FOCUS_PATCH = (ROOT / "tools" / "exp015_focus_patch.txt").read_text(encoding="utf-8")

VARIANTS = {
    "EXP012": (
        "exp012-safe-div-parent12",
        "Only safe-div parent-daughter radius 9.0 -> 12.0 um; targets the audited 11.02 um daughter.",
        [("BIOHUB_SAFE_DIV_MAX_UM", "12.0")],
        False,
    ),
    "EXP013": (
        "exp013-safe-div-no-divergence",
        "Only disable the safe-div t+2 divergence gate; all distance, mutual-NN, symmetry and DeepCenter gates remain.",
        [("BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE", "0")],
        False,
    ),
    "EXP014": (
        "exp014-safe-div-no-deepcenter-veto",
        "Only disable DeepCenter veto for safe-div proposals; detector, edge model and other gates remain unchanged.",
        [("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "0")],
        False,
    ),
    "EXP015": (
        "exp015-focus3d-candidate-union",
        "Add public Hengck23 FOCUS3D-derived point detections as extra candidate nodes; keep Pilkwang edges and all post-processing.",
        [
            ("BIOHUB_FOCUS3D_UNION", "1"),
            ("BIOHUB_FOCUS3D_THRESHOLD", "0.5"),
            ("BIOHUB_FOCUS3D_DEDUP_UM", "2.0"),
        ],
        True,
    ),
}


def make_notebook(exp: str, slug: str, description: str, env: list[tuple[str, str]], focus: bool) -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = payload["cells"]
    intro = "".join(cells[0].get("source", []))
    intro = intro.replace("# Biohub 0.942 LB, one knob past the public line", f"# {exp} frozen-weight follow-up", 1)
    cells[0]["source"] = (intro + "\n\n" + description + "\n").splitlines(keepends=True)

    config = "".join(cells[4].get("source", []))
    config += f"\n# {exp} overrides\n"
    config += "".join(f"os.environ[{key!r}] = {value!r}\n" for key, value in env)
    cells[4]["source"] = config.splitlines(keepends=True)

    if focus:
        cell8 = "".join(cells[8].get("source", []))
        anchor = "start_time = time.time()"
        if cell8.count(anchor) != 1:
            raise RuntimeError("FOCUS3D insertion anchor not found")
        cell8 = cell8.replace(anchor, FOCUS_PATCH + "\n" + anchor, 1)
        cells[8]["source"] = cell8.splitlines(keepends=True)

    manifest = "".join(cells[13].get("source", []))
    cells[13]["source"] = (manifest + f'\nprint("{exp} variant: {description}")\n').splitlines(keepends=True)
    payload["metadata"]["title"] = f"{exp} {slug}"

    out_dir = ROOT / "EXP" / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"CELL_{slug.replace('-', '_')}.ipynb"
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    for exp, (slug, description, env, focus) in VARIANTS.items():
        make_notebook(exp, slug, description, env, focus)
