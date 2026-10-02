"""Render EXP018 from the locked EXP017 notebook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP017" / "CELL_infer_public_0947_dctta.ipynb"
DST = ROOT / "EXP" / "EXP018" / "CELL_infer_focus_mask_division.ipynb"
MODULE = (ROOT / "tools" / "focus_mask_arbitration.py").read_text(encoding="utf-8")


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]
    original_cell0 = cells[0]["source"]
    if isinstance(original_cell0, str):
        original_cell0 = original_cell0.splitlines(True)
    cells[0]["source"] = [
        "# EXP018: EXP017 + original FOCUS3D instance-mask local division arbitration\n",
        "# Frozen Pilkwang dual-seed weights and all EXP017 post-processing remain locked.\n",
        "\n",
    ] + original_cell0 + [
        "\n# EXP018 runtime budget: skip the optional held-out validator during the submission run.\n",
        "os.environ[\"BIOHUB_VALIDATOR_ENABLE\"] = \"0\"\n",
    ]

    import_cell = """
# EXP018 local mask evidence implementation (self-contained copy)
_focus_module_source = %r
_focus_module_path = WORKING_DIR / "focus_mask_arbitration.py"
_focus_module_path.write_text(_focus_module_source, encoding="utf-8")
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))
from focus_mask_arbitration import FocusMaskProvider, apply_focus_mask_arbitration

FOCUS_MASK_ENABLE = os.environ.get("BIOHUB_FOCUS_MASK_ENABLE", "1") != "0"
FOCUS_MASK_MATCH_RADIUS_UM = float(os.environ.get("BIOHUB_FOCUS_MASK_MATCH_RADIUS_UM", "7.0"))
FOCUS_MASK_PARENT_MAX_UM = float(os.environ.get("BIOHUB_FOCUS_MASK_PARENT_MAX_UM", "14.0"))
FOCUS_MASK_MIN_DIV_SCORE = float(os.environ.get("BIOHUB_FOCUS_MASK_MIN_DIV_SCORE", "0.20"))
FOCUS_MASK_MIN_MARGIN = float(os.environ.get("BIOHUB_FOCUS_MASK_MIN_MARGIN", "0.10"))
FOCUS_MASK_MAX_FRAMES = int(os.environ.get("BIOHUB_FOCUS_MASK_MAX_FRAMES", "80"))
FOCUS_MASK_MAX_CONFLICTS = int(os.environ.get("BIOHUB_FOCUS_MASK_MAX_CONFLICTS", "4"))
FOCUS_MASK_MAX_FRAMES_PER_DATASET = int(os.environ.get("BIOHUB_FOCUS_MASK_MAX_FRAMES_PER_DATASET", "2"))
FOCUS_MASK_PROVIDER = FocusMaskProvider(
    WORKING_DIR / "focus3d_mask_cache", scale_um=(1.625, 0.40625, 0.40625), max_frames=FOCUS_MASK_MAX_FRAMES
) if FOCUS_MASK_ENABLE else None
print(f"FOCUS mask arbitration: enabled={FOCUS_MASK_ENABLE} match_radius={FOCUS_MASK_MATCH_RADIUS_UM}um "
      f"parent_max={FOCUS_MASK_PARENT_MAX_UM}um min_score={FOCUS_MASK_MIN_DIV_SCORE} margin={FOCUS_MASK_MIN_MARGIN}")
""" % MODULE
    cell5_source = cells[5]["source"]
    if isinstance(cell5_source, str):
        cell5_source = cell5_source.splitlines(True)
    cells[5]["source"] = import_cell.splitlines(True) + cell5_source

    s = "".join(cells[5]["source"])
    stats_marker = '        "linefit_skipped_nodes": 0,\n'
    if stats_marker in s and '"focus_checked": 0' not in s:
        s = s.replace(stats_marker, stats_marker +
            '        "focus_checked": 0,\n'
            '        "focus_accepted": 0,\n'
            '        "focus_rejected_margin": 0,\n'
            '        "focus_missing_mask": 0,\n'
            '        "focus_frames_loaded": 0,\n'
            '        "focus_nodes_matched": 0,\n'
            '        "focus_runtime_unavailable": 0,\n')
    hook = '''
    if FOCUS_MASK_PROVIDER is not None and dataset is not None and edges:
        edges = apply_focus_mask_arbitration(
            nodes_by_id, edges, dataset=dataset, provider=FOCUS_MASK_PROVIDER,
            frame_loader=lambda _t: read_test_frame(dataset, int(_t), repair_frame_cache),
            scale_um=VOXEL_SCALE_UM, radius_um=FOCUS_MASK_MATCH_RADIUS_UM,
            max_parent_um=FOCUS_MASK_PARENT_MAX_UM,
            min_division_score=FOCUS_MASK_MIN_DIV_SCORE, min_margin=FOCUS_MASK_MIN_MARGIN,
            max_conflicts=FOCUS_MASK_MAX_CONFLICTS,
            max_frames_per_dataset=FOCUS_MASK_MAX_FRAMES_PER_DATASET,
            stats=stats, edge_distance_um=edge_distance_um,
        )
        print(f"  [{dataset}] FOCUS mask local arbitration: checked={stats.get('focus_checked', 0)} "
              f"accepted={stats.get('focus_accepted', 0)} frames={stats.get('focus_frames_loaded', 0)}")

'''
    marker = "    _geo_cands = stats['safe_division_geometric_candidates']\n"
    if marker not in s:
        raise RuntimeError("safe-division insertion marker not found")
    s = s.replace(marker, hook + marker, 1)
    cells[5]["source"] = s.splitlines(True)

    manifest = '''
print(f"FOCUS3D mask provider: enabled={FOCUS_MASK_ENABLE} loaded_calls="
      f"{getattr(FOCUS_MASK_PROVIDER, 'calls', 0) if FOCUS_MASK_PROVIDER else 0} "
      f"error={getattr(FOCUS_MASK_PROVIDER, 'error', None) if FOCUS_MASK_PROVIDER else None}")
print("FOCUS contribution uses instance_map voxel geometry for local occupied-daughter rewiring; no new nodes.")
'''
    cell11_source = cells[11]["source"]
    if isinstance(cell11_source, str):
        cell11_source = cell11_source.splitlines(True)
    cells[11]["source"] = cell11_source + manifest.splitlines(True)

    nb["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
