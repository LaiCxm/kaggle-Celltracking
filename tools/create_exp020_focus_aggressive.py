"""Render EXP020: test-only aggressive FOCUS3D mask arbitration."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP017" / "CELL_infer_public_0947_dctta.ipynb"
DST = ROOT / "EXP" / "EXP020" / "CELL_infer_focus3d_aggressive.ipynb"
BASE_MODULE = (ROOT / "tools" / "focus_mask_arbitration.py").read_text(encoding="utf-8")
STRATEGY_MODULE = (ROOT / "tools" / "focus_mask_strategies.py").read_text(encoding="utf-8")


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(True)


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    config = source(cells[0]) + (
        "\n# EXP020: test-only aggressive FOCUS3D evidence; validator stays separate.\n"
        'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"\n'
    )
    set_source(cells[0], config)

    injected = """
# EXP020 self-contained FOCUS3D mask strategy
_focus_base_source = %r
_focus_base_path = WORKING_DIR / "focus_mask_arbitration.py"
_focus_base_path.write_text(_focus_base_source, encoding="utf-8")
_focus_strategy_source = %r
_focus_strategy_path = WORKING_DIR / "focus_mask_strategies.py"
_focus_strategy_path.write_text(_focus_strategy_source, encoding="utf-8")
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))
from focus_mask_arbitration import FocusMaskProvider
from focus_mask_strategies import apply_focus_strategy

# One pre-registered aggressive setting. No threshold sweep is run on test.
FOCUS_STRATEGY_MODE = "selected_rescue"
FOCUS_MATCH_RADIUS_UM = 7.0
FOCUS_PARENT_MAX_UM = 14.0
FOCUS_MIN_DIV_SCORE = 0.20
FOCUS_MIN_MARGIN = 0.02
FOCUS_WEIGHT = 1.75
FOCUS_MAX_CONFLICTS = 64
FOCUS_MAX_FRAMES = 16
FOCUS_PROVIDER = FocusMaskProvider(
    WORKING_DIR / "focus3d_mask_cache",
    scale_um=(1.625, 0.40625, 0.40625),
    max_frames=80,
)
print(
    "EXP020 aggressive FOCUS3D:",
    {"weight": FOCUS_WEIGHT, "margin": FOCUS_MIN_MARGIN,
     "max_conflicts_per_movie": FOCUS_MAX_CONFLICTS,
     "max_frames_per_movie": FOCUS_MAX_FRAMES},
)
""" % (BASE_MODULE, STRATEGY_MODULE)

    pipeline = injected + source(cells[5])
    hook = '''
    if dataset is not None and edges:
        edges = apply_focus_strategy(
            nodes_by_id,
            edges,
            mode=FOCUS_STRATEGY_MODE,
            dataset=dataset,
            provider=FOCUS_PROVIDER,
            frame_loader=lambda _t: read_test_frame(dataset, int(_t), repair_frame_cache),
            scale_um=VOXEL_SCALE_UM,
            radius_um=FOCUS_MATCH_RADIUS_UM,
            max_parent_um=FOCUS_PARENT_MAX_UM,
            min_division_score=FOCUS_MIN_DIV_SCORE,
            min_margin=FOCUS_MIN_MARGIN,
            focus_weight=FOCUS_WEIGHT,
            max_conflicts=FOCUS_MAX_CONFLICTS,
            max_frames=FOCUS_MAX_FRAMES,
            stats=stats,
            edge_distance_um=edge_distance_um,
        )
        print(
            f"  [{dataset}] EXP020 FOCUS: ranked={stats.get('focus_ranked_conflicts', 0)} "
            f"selected={stats.get('focus_selected_conflicts', 0)} "
            f"checked={stats.get('focus_checked', 0)} accepted={stats.get('focus_accepted', 0)} "
            f"frames={stats.get('focus_frames_loaded', 0)}"
        )

'''
    marker = "    _geo_cands = stats['safe_division_geometric_candidates']\n"
    if marker not in pipeline:
        raise RuntimeError("safe-division hook marker not found")
    set_source(cells[5], pipeline.replace(marker, hook + marker, 1))

    manifest = source(cells[11]) + '''
print("EXP020 FOCUS3D role: weighted 3-D contour evidence for occupied-daughter rewiring; no new nodes.")
print("EXP020 parameters:", {
    "focus_weight": FOCUS_WEIGHT,
    "min_margin": FOCUS_MIN_MARGIN,
    "max_conflicts_per_movie": FOCUS_MAX_CONFLICTS,
    "max_frames_per_movie": FOCUS_MAX_FRAMES,
    "focus_inference_calls": getattr(FOCUS_PROVIDER, "calls", 0),
    "focus_error": FOCUS_PROVIDER.error,
})
'''
    set_source(cells[11], manifest)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP020 Aggressive FOCUS3D Evidence"
    )
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
