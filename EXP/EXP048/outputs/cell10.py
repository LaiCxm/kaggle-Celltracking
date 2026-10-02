
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


EXP019_MODES = ["off"]
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
print("EXP019 official CV complete; no second test pass and no submission.csv.")
