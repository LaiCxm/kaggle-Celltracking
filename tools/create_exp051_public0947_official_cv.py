"""Create an official-CV-only audit notebook from the locked public 0.947 candidate.

The source notebook is EXP017's exact sjlee101/biohub-lf-dctta reproduction.
Only the execution target and scorer are changed: test inference/submission and
the notebook's legacy proxy sweep are removed, while four complete labelled
movies are scored by the patched official scorer.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "EXP" / "EXP017" / "CELL_infer_public_0947_dctta.ipynb"
DST = ROOT / "EXP" / "EXP051" / "CELL_public_0947_official_cv.ipynb"

VAL_STEMS = [
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
]


def text(cell: dict) -> str:
    return "".join(cell.get("source", []))


OFFICIAL_CV_CELL = r'''
# EXP051 official patched-scorer audit of the public 0.947 candidate.
import csv as _exp051_csv
import glob as _exp051_glob
import importlib.util as _exp051_importlib
import inspect as _exp051_inspect

_metric_hits = _exp051_glob.glob(
    "/kaggle/input/**/tracking_cellmot_075fc5f/src/tracking_cellmot/metrics.py",
    recursive=True,
)
if not _metric_hits:
    raise RuntimeError("EXP051: patched official scorer dataset is not mounted")
_official_metrics_path = Path(sorted(_metric_hits)[0])
_official_src = _official_metrics_path.parents[1]
_official_repo = _official_metrics_path.parents[2]
_official_scripts = _official_repo / "scripts"
for _path in (_official_src, _official_scripts):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tracking_cellmot.division_metrics as _official_division_metrics
from tracking_cellmot.metrics import summarise as _official_summarise

if "_weakly_connected_components" in _exp051_inspect.getsource(_official_division_metrics):
    raise RuntimeError("EXP051 loaded the pre-patch division metric")


def _load_script(name: str, path: Path):
    _spec = _exp051_importlib.spec_from_file_location(name, path)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"EXP051 cannot load {path}")
    _module = _exp051_importlib.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    return _module


_csv_module = _load_script("_exp051_csv_to_geffs", _official_scripts / "csv_to_geffs.py")
_eval_module = _load_script("_exp051_evaluate", _official_scripts / "evaluate.py")
print("EXP051 official patched scorer:", _official_metrics_path)

_csv_columns = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]
_cv_csv = WORKING_DIR / "EXP051_public0947_official_cv.csv"
_cv_geff_dir = WORKING_DIR / "EXP051_public0947_official_cv_geffs"
_cv_stats = []
_row_id = 0
_old_test_dir = TEST_DIR
TEST_DIR = TRAIN_DIR
try:
    with _cv_csv.open("w", newline="") as _handle:
        _writer = _exp051_csv.DictWriter(_handle, fieldnames=_csv_columns)
        _writer.writeheader()
        for _stem in val_stems:
            _pred_path = next((REPO_DIR / "predictions").rglob(f"{_stem}.geff"), None)
            if _pred_path is None:
                raise RuntimeError(f"EXP051 missing prediction graph for {_stem}")
            _graph = graph_from_geff(_pred_path)
            _nodes = {}
            for _row in _graph.node_attrs().iter_rows(named=True):
                _node_id = int(_row["node_id"])
                _nodes[_node_id] = {
                    "node_id": _node_id,
                    "t": int(_row["t"]),
                    "z": float(_row["z"]),
                    "y": float(_row["y"]),
                    "x": float(_row["x"]),
                }
            _edges = []
            for _row in _graph.edge_attrs().iter_rows(named=True):
                _edge_prob = _row.get("edge_prob") if hasattr(_row, "get") else None
                _edges.append({
                    "source_id": int(_row["source_id"]),
                    "target_id": int(_row["target_id"]),
                    "edge_prob": None if _edge_prob is None else float(_edge_prob),
                })
            _raw_nodes = len(_nodes)
            _nodes, _edges, _filter_stats = filter_output_graph(
                _nodes,
                _edges,
                dataset=_stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
            if not _nodes:
                raise RuntimeError(f"EXP051 post-processing removed every node from {_stem}")
            for _node_id in sorted(_nodes):
                _node = _nodes[_node_id]
                _writer.writerow({
                    "id": _row_id,
                    "dataset": _stem,
                    "row_type": "node",
                    "node_id": int(_node_id),
                    "t": int(_node["t"]),
                    "z": max(0, int(round(float(_node["z"])))),
                    "y": max(0, int(round(float(_node["y"])))),
                    "x": max(0, int(round(float(_node["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                })
                _row_id += 1
            for _edge in _edges:
                _writer.writerow({
                    "id": _row_id,
                    "dataset": _stem,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(_edge["source_id"]),
                    "target_id": int(_edge["target_id"]),
                })
                _row_id += 1
            _cv_stats.append({
                "dataset": _stem,
                "raw_nodes": _raw_nodes,
                "nodes": len(_nodes),
                "edges": len(_edges),
                **_filter_stats,
            })
finally:
    TEST_DIR = _old_test_dir

_csv_module.csv_to_geffs(_cv_csv, _cv_geff_dir, overwrite=True)
_official_rows, _skipped = _eval_module.evaluate_pairs(
    _cv_geff_dir,
    TRAIN_DIR,
    max_distance=7.0,
)
if _skipped or len(_official_rows) != len(val_stems):
    raise RuntimeError({"skipped": _skipped, "rows": len(_official_rows)})
_summary = _official_summarise(_official_rows)
_summary["experiment"] = "EXP051 public sjlee101/biohub-lf-dctta"
_summary["source_notebook"] = "EXP017/CELL_infer_public_0947_dctta.ipynb"
_summary["fixed_postprocess"] = {"MOTION_RELINK_TIGHT_UM": 5.5}
_summary["validation_movies"] = list(val_stems)
_summary["official_scorer"] = str(_official_metrics_path)
_summary["prediction_rows"] = _row_id
_summary["filter_stats"] = _cv_stats
_cv_stats_path = WORKING_DIR / "EXP051_public0947_official_cv_per_movie.csv"
pd.DataFrame(_official_rows).to_csv(_cv_stats_path, index=False)
(WORKING_DIR / "EXP051_public0947_official_cv_summary.json").write_text(
    json.dumps(_summary, indent=2, sort_keys=True, default=str) + "\n"
)
print("EXP051 OFFICIAL SUMMARY", json.dumps(_summary, indent=2, sort_keys=True, default=str))
print("EXP051 complete: exact public 0.947 candidate, no EXP049 logic, no FOCUS3D, no submission.csv")
'''


def main() -> None:
    notebook = json.loads(SRC.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 12:
        raise RuntimeError(f"unexpected source cell count: {len(cells)}")

    # Lock the published candidate's selected post-process arm. The source
    # notebook defaults to 6.0 and scans tight55 only when its legacy validator
    # is enabled; the public run selected tight55=5.5.
    cell0 = text(cells[0])
    cell0 += (
        "\n# EXP051: official CV only; lock the public run's selected tight55 arm.\n"
        "os.environ[\"BIOHUB_MOTION_RELINK_TIGHT_UM\"] = \"5.5\"\n"
    )
    cells[0]["source"] = cell0.splitlines(True)

    # Retain all setup and shard helpers but remove the test-set inference pass.
    cell4 = text(cells[4])
    run_start = cell4.index("start_time = time.time()")
    cell4 = cell4[:run_start] + (
        "predict_seconds = 0.0\n"
        "print(\"EXP051 CV-only: skipped hidden test inference; only fixed validation movies run below.\")\n"
    )
    cells[4]["source"] = cell4.splitlines(True)

    cell5 = text(cells[5])
    _submission_call = '\nwrite_test_submission("base")\n'
    if cell5.count(_submission_call) != 1:
        raise RuntimeError("EXP051 expected exactly one original test submission call")
    cell5 = cell5.replace(_submission_call, "\n", 1)
    cells[5]["source"] = cell5.splitlines(True)

    # The original cell assumes a test submission exists. It is irrelevant for
    # a CV audit and would incorrectly turn the validation notebook into a code
    # competition submission.
    cells[6]["source"] = [
        "print(\"EXP051 CV-only: submission retention guard skipped by design.\")\n"
    ]

    # Replace the broad prefix-based holdout selection with the fixed four
    # complete movies used by EXP019/EXP049, containing five GT divisions.
    cell7 = text(cells[7])
    selection_start = cell7.index("val_stems: list[str] = []")
    selection_end = cell7.index("def _merge_validator_shards", selection_start)
    fixed_selection = """val_stems: list[str] = %r
if not TRAIN_DIR.exists():
    raise RuntimeError(f\"EXP051: TRAIN_DIR not found at {TRAIN_DIR}\")
_missing_val = [
    _stem for _stem in val_stems
    if not (TRAIN_DIR / f\"{_stem}.zarr\").exists()
    or not (TRAIN_DIR / f\"{_stem}.geff\").exists()
]
if _missing_val:
    raise RuntimeError(f\"EXP051: fixed validation movies missing: {_missing_val}\")
print(f\"EXP051: fixed official-CV movies ({len(val_stems)}): {val_stems}\")
print(\"EXP051: these four movies contain five GT division events; no movie selection or parameter scan is performed.\")


""" % VAL_STEMS
    cell7 = cell7[:selection_start] + fixed_selection + cell7[selection_end:]
    cells[7]["source"] = cell7.splitlines(True)

    # Do not execute the source notebook's hand-written metric or proxy sweep.
    cells[8]["source"] = OFFICIAL_CV_CELL.splitlines(True)
    cells[9]["source"] = [
        "print(\"EXP051: legacy hand-written metric and post-process sweep disabled; official scorer ran in cell 8.\")\n"
    ]
    cells[10]["source"] = [
        "print(\"EXP051: no selected-arm rewrite and no submission generation.\")\n"
    ]
    cells[11]["source"] = [
        "print(\"EXP051 manifest: exact sjlee101/biohub-lf-dctta public candidate\")\n",
        "print(\"EXP051 fixed MOTION_RELINK_TIGHT_UM=5.5; official patched scorer; four complete validation movies\")\n",
        "print(\"EXP051 FOCUS3D=False; EXP049 post-ILP division promotion=False; submission=False\")\n",
    ]

    metadata = notebook.setdefault("metadata", {})
    metadata["title"] = "EXP051 Official CV Public 0.947 Candidate"
    kaggle = metadata.setdefault("kaggle", {})
    kaggle["title"] = "EXP051 Official CV Public 0.947 Candidate"
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        ast.parse(text(cell), filename=f"{DST}:cell{index}")
    print(DST)
    print(f"cells={len(cells)} fixed_validation_movies={len(VAL_STEMS)}")


if __name__ == "__main__":
    main()
