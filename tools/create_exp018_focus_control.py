"""Create the clean FOCUS3D-off control from the scored EXP018 v3 notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP018" / "CELL_infer_focus_mask_division.ipynb"
DST = ROOT / "EXP" / "EXP018" / "CELL_infer_focus_mask_control.ipynb"


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(True)


def main() -> None:
    baseline = json.loads(SRC.read_text(encoding="utf-8"))
    control = json.loads(SRC.read_text(encoding="utf-8"))

    cell0 = _cell_source(control["cells"][0])
    marker = 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"\n'
    if marker not in cell0:
        raise RuntimeError("EXP018 v3 validator marker not found")
    if "BIOHUB_FOCUS_MASK_ENABLE" in cell0:
        raise RuntimeError("EXP018 v3 already overrides the FOCUS mask switch")
    replacement = marker + (
        "\n# EXP018-B clean ablation: the only effective configuration change.\n"
        'os.environ["BIOHUB_FOCUS_MASK_ENABLE"] = "0"\n'
    )
    _set_cell_source(control["cells"][0], cell0.replace(marker, replacement, 1))

    changed_cells = [
        index
        for index, (left, right) in enumerate(zip(baseline["cells"], control["cells"]))
        if _cell_source(left) != _cell_source(right)
    ]
    if changed_cells != [0]:
        raise RuntimeError(f"unexpected changed cells: {changed_cells}")

    before = _cell_source(baseline["cells"][0])
    after = _cell_source(control["cells"][0])
    if after.replace(
        "\n# EXP018-B clean ablation: the only effective configuration change.\n"
        'os.environ["BIOHUB_FOCUS_MASK_ENABLE"] = "0"\n',
        "",
        1,
    ) != before:
        raise RuntimeError("control differs from EXP018 v3 by more than the declared switch")

    control.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP018-B clean FOCUS3D-off control"
    )
    DST.write_text(json.dumps(control, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    code = "\n".join(_cell_source(cell) for cell in control["cells"])
    print(DST)
    print("changed_source_cells=0")
    print("focus_mask_enable=0")
    print("code_sha256=" + hashlib.sha256(code.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
