"""Create EXP040: official-CV validation of compact occupied-daughter rewiring."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP019" / "CELL_focus3d_strategy_official_cv.ipynb"
OUT = ROOT / "EXP" / "EXP040" / "CELL_official_cv_local_division_conflict.ipynb"
MODULE = ROOT / "tools" / "division_conflict_rewire.py"


RUNNER = r'''# EXP040: full-graph occupied-daughter division conflict comparison.
# This cell loads the patched official scorer directly; no proxy metric is used.
import csv as _exp040_csv
import glob as _exp040_glob
import importlib.util as _exp040_importlib
import inspect as _exp040_inspect
import copy as _exp040_copy

_metric_hits = _exp040_glob.glob(
    "/kaggle/input/**/tracking_cellmot_075fc5f/src/tracking_cellmot/metrics.py",
    recursive=True,
)
if not _metric_hits:
    raise RuntimeError("EXP040: patched official scorer dataset is not mounted")
_official_metrics_path = Path(sorted(_metric_hits)[0])
_official_src = _official_metrics_path.parents[1]
_official_repo = _official_metrics_path.parents[2]
_official_scripts = _official_repo / "scripts"
for _path in (_official_src, _official_scripts):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tracking_cellmot.division_metrics as _official_division_metrics
from tracking_cellmot.metrics import summarise as _official_summarise

if "_weakly_connected_components" in _exp040_inspect.getsource(_official_division_metrics):
    raise RuntimeError("EXP040 loaded the pre-patch division metric")


def _load_script(name: str, path: Path):
    spec = _exp040_importlib.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = _exp040_importlib.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_csv_module = _load_script("_exp040_csv_to_geffs", _official_scripts / "csv_to_geffs.py")
_eval_module = _load_script("_exp040_evaluate", _official_scripts / "evaluate.py")
print("EXP040 official patched scorer:", _official_metrics_path)
print("EXP040 held-out movies:", val_stems)


def _write_exp040_csv(mode: str):
    global DIVISION_CONFLICT_MODE, FOCUS_STRATEGY_MODE, TEST_DIR
    path = WORKING_DIR / f"exp040_{mode}.csv"
    stats_rows = []
    logs = []
    row_id = 0
    old_test_dir = TEST_DIR
    DIVISION_CONFLICT_MODE = mode
    FOCUS_STRATEGY_MODE = "off"
    DIVISION_CONFLICT_LOGS.clear()
    TEST_DIR = TRAIN_DIR
    try:
        with path.open("w", newline="") as handle:
            writer = _exp040_csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for stem in val_stems:
                raw_nodes, raw_edges = VAL_RAW_GRAPHS[stem]
                nodes, edges, stats = filter_output_graph(
                    _exp040_copy.deepcopy(raw_nodes),
                    _exp040_copy.deepcopy(raw_edges),
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
        logs = [dict(row) for row in DIVISION_CONFLICT_LOGS]
    finally:
        TEST_DIR = old_test_dir
        DIVISION_CONFLICT_MODE = "base"
        FOCUS_STRATEGY_MODE = "off"
    return path, stats_rows, logs


EXP040_MODES = ["base", "compact_10_14", "compact_12_14", "compact_14_14"]
_summary_rows = []
_sample_rows = []
_strategy_stats = []
_rewire_logs = []

for _mode in EXP040_MODES:
    print("=" * 78)
    print("EXP040 mode:", _mode, DIVISION_CONFLICT_CONFIGS[_mode])
    _csv_path, _stats, _logs = _write_exp040_csv(_mode)
    _strategy_stats.extend(_stats)
    _rewire_logs.extend(_logs)
    _pred_dir = WORKING_DIR / f"exp040_{_mode}_geffs"
    _csv_module.csv_to_geffs(_csv_path, _pred_dir, overwrite=True)
    _rows, _skipped = _eval_module.evaluate_pairs(_pred_dir, TRAIN_DIR, max_distance=7.0)
    if _skipped or len(_rows) != len(val_stems):
        raise RuntimeError({"mode": _mode, "skipped": _skipped, "rows": len(_rows)})
    _summary = _official_summarise(_rows)
    _mode_stats = [row for row in _stats if row["mode"] == _mode]
    _summary_rows.append({
        "mode": _mode,
        **_summary,
        "rewire_candidates": sum(int(row.get("division_conflict_candidates", 0)) for row in _mode_stats),
        "rewire_accepted": sum(int(row.get("division_conflict_accepted", 0)) for row in _mode_stats),
        "rewire_rejected_divergence": sum(int(row.get("division_conflict_rejected_divergence", 0)) for row in _mode_stats),
        "rewire_rejected_conflict": sum(int(row.get("division_conflict_rejected_conflict", 0)) for row in _mode_stats),
        "rewire_rejected_cap": sum(int(row.get("division_conflict_rejected_cap", 0)) for row in _mode_stats),
    })
    for _stem, _row in zip(sorted(val_stems), _rows):
        _sample_rows.append({"mode": _mode, "dataset": _stem, **_row})
    print("OFFICIAL SUMMARY", _summary_rows[-1])

_out = WORKING_DIR / "exp040_official_cv"
_out.mkdir(parents=True, exist_ok=True)
_summary_frame = pd.DataFrame(_summary_rows).sort_values("score", ascending=False)
_summary_frame.to_csv(_out / "official_summary.csv", index=False)
pd.DataFrame(_sample_rows).to_csv(_out / "official_per_movie.csv", index=False)
pd.DataFrame(_strategy_stats).to_csv(_out / "rewire_stats.csv", index=False)
pd.DataFrame(_rewire_logs).to_csv(_out / "rewire_log.csv", index=False)
(_out / "manifest.json").write_text(json.dumps({
    "experiment": "EXP040",
    "official_patched_scorer": str(_official_metrics_path),
    "submission_generated": False,
    "modes": DIVISION_CONFLICT_CONFIGS,
    "held_out_movies": val_stems,
    "method": "post-safe-division local occupied-daughter one-delete-one-add rewire",
}, indent=2), encoding="utf-8")
report = [
    "# EXP040 official CV: compact occupied-daughter conflict rewiring", "",
    "This is a four-complete-movie official patched-scorer validation; no test submission is generated.", "",
    "```text", _summary_frame.to_string(index=False), "```", "",
    "The three non-base modes are pre-registered geometry gates. Each accepted operation deletes exactly q->daughter2 and inserts parent->daughter2; candidate and accepted counts are in rewire_stats.csv, with every accepted operation in rewire_log.csv.",
]
(_out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
display(_summary_frame)
if SUBMISSION_PATH.exists():
    raise RuntimeError("EXP040 is CV-only and must not emit submission.csv")
print("EXP040 official CV complete; no test-set pass and no submission.csv.")
'''


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 12:
        raise RuntimeError(f"unexpected EXP019 CV notebook cell count: {len(cells)}")

    cell0 = "".join(cells[0]["source"])
    cell0 = cell0.replace(
        "Research edition.",
        "Research edition.\n\nEXP040 validates only local occupied-daughter division rewiring with the official patched scorer.",
    )
    cells[0]["source"] = cell0.splitlines(True)

    cell5 = "".join(cells[5]["source"])
    marker = '    if FOCUS_STRATEGY_MODE != "off" and dataset is not None and edges:\n'
    if marker not in cell5:
        raise RuntimeError("EXP019 post-process insertion point not found")
    insertion = '''    if DIVISION_CONFLICT_MODE != "base" and dataset is not None and edges:
        _conflict_config = DIVISION_CONFLICT_CONFIGS[DIVISION_CONFLICT_MODE]
        edges, _conflict_events = rewire_occupied_daughter_conflicts(
            nodes_by_id, edges,
            parent_gate_um=float(_conflict_config["parent_gate_um"]),
            daughter_gate_um=float(_conflict_config["daughter_gate_um"]),
            divergence_um=float(_conflict_config["divergence_um"]),
            max_changes_per_frame=int(_conflict_config["max_changes_per_frame"]),
            max_changes_per_dataset=int(_conflict_config["max_changes_per_dataset"]),
            require_divergence=bool(_conflict_config["require_divergence"]),
            stats=stats, edge_distance_um=edge_distance_um,
        )
        DIVISION_CONFLICT_LOGS.extend([
            {"mode": DIVISION_CONFLICT_MODE, "dataset": dataset, **_event}
            for _event in _conflict_events
        ])
        print(f"  [{dataset}] compact division conflict mode={DIVISION_CONFLICT_MODE} "
              f"candidates={stats.get('division_conflict_candidates', 0)} "
              f"accepted={stats.get('division_conflict_accepted', 0)}")

'''
    cells[5]["source"] = cell5.replace(marker, insertion + marker, 1).splitlines(True)

    module_source = MODULE.read_text(encoding="utf-8")
    module_source += '''

DIVISION_CONFLICT_MODE = "base"
DIVISION_CONFLICT_LOGS: list[dict[str, object]] = []
DIVISION_CONFLICT_CONFIGS = {
    "base": {"parent_gate_um": None, "daughter_gate_um": None},
    "compact_10_14": {
        "parent_gate_um": 10.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "max_changes_per_frame": 1,
        "max_changes_per_dataset": 6, "require_divergence": True,
    },
    "compact_12_14": {
        "parent_gate_um": 12.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "max_changes_per_frame": 1,
        "max_changes_per_dataset": 6, "require_divergence": True,
    },
    "compact_14_14": {
        "parent_gate_um": 14.0, "daughter_gate_um": 14.0,
        "divergence_um": 2.25, "max_changes_per_frame": 1,
        "max_changes_per_dataset": 6, "require_divergence": True,
    },
}
print("EXP040 conflict modes:", DIVISION_CONFLICT_CONFIGS)
'''
    module_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": module_source.splitlines(True),
        "id": "compact-division-conflict-module",
    }
    cells.insert(6, module_cell)
    # Original cell 10 becomes cell 11 after the module insertion.
    cells[11]["source"] = RUNNER.splitlines(True)
    cells[12]["source"] = [
        'print("EXP040 manifest: official patched scorer, CV-only, compact occupied-daughter conflict arms.")\n'
    ]
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP040 Official CV Local Division Conflict"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
