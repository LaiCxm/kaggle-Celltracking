"""Render EXP019: official-scorer comparison of FOCUS3D mask-use strategies."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP017" / "CELL_infer_public_0947_dctta.ipynb"
DST = ROOT / "EXP" / "EXP019" / "CELL_focus3d_strategy_official_compare.ipynb"
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

    config = source(cells[0]).replace(
        'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"',
        'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "2"',
    )
    config += (
        "\n# EXP019 mechanism comparison: four complete labelled movies.\n"
        'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"\n'
    )
    set_source(cells[0], config)

    injected = """
# EXP019 FOCUS3D strategy modules (self-contained copies)
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

FOCUS_STRATEGY_MODE = "off"
FOCUS_MATCH_RADIUS_UM = 7.0
FOCUS_PARENT_MAX_UM = 14.0
FOCUS_MIN_DIV_SCORE = 0.20
FOCUS_MIN_MARGIN = 0.10
FOCUS_MAX_CONFLICTS = 4
FOCUS_MAX_FRAMES = 4
FOCUS_PROVIDER = FocusMaskProvider(
    WORKING_DIR / "focus3d_mask_cache",
    scale_um=(1.625, 0.40625, 0.40625),
    max_frames=80,
)
print("EXP019 FOCUS strategies ready; production test pass remains FOCUS-off")
""" % (BASE_MODULE, STRATEGY_MODULE)

    pipeline = injected + source(cells[5])
    hook = '''
    if FOCUS_STRATEGY_MODE != "off" and dataset is not None and edges:
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
            max_conflicts=FOCUS_MAX_CONFLICTS,
            max_frames=FOCUS_MAX_FRAMES,
            stats=stats,
            edge_distance_um=edge_distance_um,
        )
        print(
            f"  [{dataset}] FOCUS strategy={FOCUS_STRATEGY_MODE} "
            f"checked={stats.get('focus_checked', 0)} accepted={stats.get('focus_accepted', 0)} "
            f"veto_removed={stats.get('focus_veto_removed', 0)} "
            f"frames={stats.get('focus_frames_loaded', 0)}"
        )

'''
    marker = "    _geo_cands = stats['safe_division_geometric_candidates']\n"
    if marker not in pipeline:
        raise RuntimeError("safe-division hook marker not found")
    pipeline = pipeline.replace(marker, hook + marker, 1)
    set_source(cells[5], pipeline)

    legacy = source(cells[9])
    stop = "validator_sample_rows: list[dict[str, object]] = []\n"
    if stop not in legacy:
        raise RuntimeError("legacy proxy execution marker not found")
    legacy = legacy.split(stop, 1)[0] + (
        "print('EXP019: legacy four-video proxy execution disabled; formal comparison uses patched scorer.')\n"
    )
    set_source(cells[9], legacy)

    comparison = r'''
import copy as _exp019_copy
import csv as _exp019_csv
import glob as _exp019_glob
import importlib.util as _exp019_importlib
import inspect as _exp019_inspect


_metric_hits = _exp019_glob.glob(
    "/kaggle/input/**/tracking_cellmot_075fc5f/src/tracking_cellmot/metrics.py",
    recursive=True,
)
if not _metric_hits:
    raise RuntimeError("EXP019: patched official scorer dataset is not mounted")
_official_metrics_path = Path(sorted(_metric_hits)[0])
_official_src = _official_metrics_path.parents[1]
_official_repo = _official_metrics_path.parents[2]
_official_scripts = _official_repo / "scripts"
for _path in (_official_src, _official_scripts):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tracking_cellmot.division_metrics as _official_division_metrics
from tracking_cellmot.metrics import summarise as _official_summarise

if "_weakly_connected_components" in _exp019_inspect.getsource(_official_division_metrics):
    raise RuntimeError("EXP019 loaded the pre-patch division metric")


def _load_script(name: str, path: Path):
    spec = _exp019_importlib.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = _exp019_importlib.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_csv_module = _load_script("_exp019_csv_to_geffs", _official_scripts / "csv_to_geffs.py")
_eval_module = _load_script("_exp019_evaluate", _official_scripts / "evaluate.py")
print("EXP019 official patched scorer:", _official_metrics_path)


def _write_strategy_csv(mode: str):
    global FOCUS_STRATEGY_MODE, TEST_DIR
    path = WORKING_DIR / f"focus_strategy_{mode}.csv"
    stats_rows = []
    row_id = 0
    old_test_dir = TEST_DIR
    FOCUS_STRATEGY_MODE = mode
    TEST_DIR = TRAIN_DIR
    try:
        with path.open("w", newline="") as handle:
            writer = _exp019_csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for stem in val_stems:
                raw_nodes, raw_edges = VAL_RAW_GRAPHS[stem]
                nodes, edges, stats = filter_output_graph(
                    _exp019_copy.deepcopy(raw_nodes),
                    _exp019_copy.deepcopy(raw_edges),
                    dataset=stem,
                    deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
                )
                for node_id in sorted(nodes):
                    node = nodes[node_id]
                    writer.writerow({
                        "id": row_id, "dataset": stem, "row_type": "node",
                        "node_id": int(node_id), "t": int(node["t"]),
                        "z": max(0, int(round(float(node["z"])))),
                        "y": max(0, int(round(float(node["y"])))),
                        "x": max(0, int(round(float(node["x"])))),
                        "source_id": -1, "target_id": -1,
                    })
                    row_id += 1
                for edge in edges:
                    writer.writerow({
                        "id": row_id, "dataset": stem, "row_type": "edge",
                        "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                        "source_id": int(edge["source_id"]),
                        "target_id": int(edge["target_id"]),
                    })
                    row_id += 1
                stats_rows.append({"mode": mode, "dataset": stem, **stats})
    finally:
        TEST_DIR = old_test_dir
        FOCUS_STRATEGY_MODE = "off"
    return path, stats_rows


EXP019_MODES = [
    "off",
    "broad_rescue",
    "selected_rescue",
    "veto_only",
    "hybrid_selected",
]
_summary_rows = []
_sample_rows = []
_strategy_stats = []

for _mode in EXP019_MODES:
    print("=" * 78)
    print("EXP019 strategy:", _mode)
    _csv_path, _stats = _write_strategy_csv(_mode)
    _strategy_stats.extend(_stats)
    _pred_dir = WORKING_DIR / f"focus_strategy_{_mode}_geffs"
    _csv_module.csv_to_geffs(_csv_path, _pred_dir, overwrite=True)
    _rows, _skipped = _eval_module.evaluate_pairs(_pred_dir, TRAIN_DIR, max_distance=7.0)
    if _skipped or len(_rows) != len(val_stems):
        raise RuntimeError({"mode": _mode, "skipped": _skipped, "rows": len(_rows)})
    _summary = _official_summarise(_rows)
    _focus_stats = [row for row in _stats if row["mode"] == _mode]
    _summary_rows.append({
        "mode": _mode,
        **_summary,
        "focus_checked": sum(int(row.get("focus_checked", 0)) for row in _focus_stats),
        "focus_accepted": sum(int(row.get("focus_accepted", 0)) for row in _focus_stats),
        "focus_veto_removed": sum(int(row.get("focus_veto_removed", 0)) for row in _focus_stats),
        "focus_frames_loaded": sum(int(row.get("focus_frames_loaded", 0)) for row in _focus_stats),
    })
    for _stem, _row in zip(sorted(val_stems), _rows):
        _sample_rows.append({"mode": _mode, "dataset": _stem, **_row})
    print("OFFICIAL SUMMARY", _summary_rows[-1])

_summary_frame = pd.DataFrame(_summary_rows).sort_values("score", ascending=False)
_summary_frame.to_csv(WORKING_DIR / "EXP019_official_strategy_summary.csv", index=False)
pd.DataFrame(_sample_rows).to_csv(WORKING_DIR / "EXP019_official_strategy_per_movie.csv", index=False)
pd.DataFrame(_strategy_stats).to_csv(WORKING_DIR / "EXP019_strategy_stats.csv", index=False)
display(_summary_frame)

# This is a diagnostic comparison, not a competition submission. Prevent the
# untouched test-base file from being mistaken for the selected EXP019 arm.
_reference_submission = WORKING_DIR / "test_base_reference_do_not_submit.csv"
if SUBMISSION_PATH.exists():
    SUBMISSION_PATH.replace(_reference_submission)
print("EXP019 comparison complete; no submission.csv intentionally produced.")
'''
    set_source(cells[10], comparison)

    manifest = source(cells[11]) + r'''
print("EXP019 design: same four complete movies and raw predictions across all arms")
print("EXP019 arms:", EXP019_MODES)
print("EXP019 formal metric: patched official scorer; legacy proxy not used for ranking")
print("FOCUS cached inference calls:", getattr(FOCUS_PROVIDER, "calls", 0), "error:", FOCUS_PROVIDER.error)
'''
    set_source(cells[11], manifest)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP019 FOCUS3D Strategy Official Comparison"
    )
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DST)


if __name__ == "__main__":
    main()
