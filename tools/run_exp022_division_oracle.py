from __future__ import annotations

"""Measure division edge-selection headroom on the EXP019 four-movie panel.

This is a validation-only oracle diagnostic. Ground-truth labels are used to
identify the prediction nodes that the patched official scorer matches to each
division endpoint. The script never creates ``submission.csv`` and must not be
used on competition test data.
"""

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
import tracksdata as td
from geff import GeffMetadata


ROOT = Path(__file__).resolve().parents[1]
SCORER_ROOT = ROOT / "EXP" / "EXP019" / "reference" / "tracking_cellmot_075fc5f"
SCORER_SRC = SCORER_ROOT / "src"
if str(SCORER_SRC) not in sys.path:
    sys.path.insert(0, str(SCORER_SRC))

from tracking_cellmot.division_metrics import match_divisions, score_divisions
from tracking_cellmot.io import DEFAULT_SCALE
from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise


BASE_CSV = ROOT / "EXP" / "EXP019" / "outputs" / "kaggle_v1" / "focus_strategy_off.csv"
GT_ROOT = ROOT / "tmp_data" / "gt_geff"
OUTPUT_ROOT = ROOT / "EXP" / "EXP022" / "outputs" / "local_official_cv"
STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
EXPECTED_BASELINE_SCORE = 0.9379884448485519
MAX_DISTANCE_UM = 7.0


# These categories come from the EXP010 stage audit. The event identity is the
# GT parent id, which is stable across prediction graph remapping.
EVENT_CATEGORY = {
    172000000050: "daughter_occupied",
    16000000001: "daughter_occupied",
    90001276: "already_true_positive",
    36000309: "daughter_occupied",
    66000550: "divergence_gate",
}


@dataclass(frozen=True)
class NodeRecord:
    node_id: int
    t: int
    z: float
    y: float
    x: float


@dataclass(frozen=True)
class OracleEvent:
    stem: str
    gt_parent: int
    gt_daughters: tuple[int, int]
    pred_parent: int | None
    pred_daughters: tuple[int | None, int | None]
    category: str
    baseline_true_positive: bool

    @property
    def repairable(self) -> bool:
        endpoints = (self.pred_parent, *self.pred_daughters)
        return all(value is not None for value in endpoints) and len(set(endpoints)) == 3


def repair_event_edges(
    edges: Iterable[tuple[int, int]],
    parent: int,
    daughters: Iterable[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Force one parent-to-two-daughters fork while preserving unrelated edges."""
    daughters_tuple = tuple(sorted(int(value) for value in daughters))
    if len(daughters_tuple) != 2 or len(set(daughters_tuple)) != 2:
        raise ValueError("A division must contain two distinct daughters")
    if parent in daughters_tuple:
        raise ValueError("The parent cannot also be a daughter")

    original = [(int(source), int(target)) for source, target in edges]
    daughter_set = set(daughters_tuple)
    removed = [
        edge
        for edge in original
        if (edge[1] in daughter_set and edge[0] != parent)
        or (edge[0] == parent and edge[1] not in daughter_set)
    ]
    removed_set = set(removed)
    result = [edge for edge in original if edge not in removed_set]
    present = set(result)
    added = [edge for edge in ((parent, daughters_tuple[0]), (parent, daughters_tuple[1])) if edge not in present]
    result.extend(added)
    return result, removed, added


def apply_oracle_events(
    edges: Iterable[tuple[int, int]],
    events: Iterable[OracleEvent],
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Apply repairable events without mutating the supplied baseline edge list."""
    result = [(int(source), int(target)) for source, target in edges]
    audit_rows: list[dict] = []
    for event in events:
        if not event.repairable:
            audit_rows.append({
                "stem": event.stem,
                "gt_parent": event.gt_parent,
                "category": event.category,
                "repairable": 0,
                "removed_edges": "[]",
                "added_edges": "[]",
            })
            continue
        assert event.pred_parent is not None
        daughters = tuple(int(value) for value in event.pred_daughters if value is not None)
        result, removed, added = repair_event_edges(result, event.pred_parent, daughters)
        audit_rows.append({
            "stem": event.stem,
            "gt_parent": event.gt_parent,
            "category": event.category,
            "repairable": 1,
            "removed_edges": json.dumps(removed),
            "added_edges": json.dumps(added),
        })
    return result, audit_rows


def _load_graph(path: Path) -> td.graph.BaseGraph:
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def _build_prediction_graph(
    nodes: list[NodeRecord],
    edges: list[tuple[int, int]],
) -> tuple[td.graph.InMemoryGraph, dict[int, int], dict[int, int]]:
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    assigned = graph.bulk_add_nodes([
        {"t": node.t, "z": node.z, "y": node.y, "x": node.x}
        for node in nodes
    ])
    raw_to_internal = dict(zip((node.node_id for node in nodes), assigned, strict=True))
    internal_to_raw = {internal: raw for raw, internal in raw_to_internal.items()}
    graph.bulk_add_edges([
        {"source_id": raw_to_internal[source], "target_id": raw_to_internal[target]}
        for source, target in edges
    ])
    return graph, raw_to_internal, internal_to_raw


def _load_base_csv() -> tuple[dict[str, list[NodeRecord]], dict[str, list[tuple[int, int]]]]:
    frame = pl.read_csv(
        BASE_CSV,
        columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"],
    )
    nodes_by_stem: dict[str, list[NodeRecord]] = {}
    edges_by_stem: dict[str, list[tuple[int, int]]] = {}
    for stem in STEMS:
        sample = frame.filter(pl.col("dataset") == stem)
        node_rows = sample.filter(pl.col("row_type") == "node")
        edge_rows = sample.filter(pl.col("row_type") == "edge")
        nodes_by_stem[stem] = [
            NodeRecord(
                node_id=int(row["node_id"]),
                t=int(row["t"]),
                z=float(row["z"]),
                y=float(row["y"]),
                x=float(row["x"]),
            )
            for row in node_rows.iter_rows(named=True)
        ]
        edges_by_stem[stem] = [
            (int(source), int(target))
            for source, target in zip(
                edge_rows["source_id"].to_list(),
                edge_rows["target_id"].to_list(),
                strict=True,
            )
        ]
    return nodes_by_stem, edges_by_stem


def _exact_divisions(gt_graph: td.graph.BaseGraph) -> list[tuple[int, tuple[int, int]]]:
    node_times = dict(
        zip(
            gt_graph.node_attrs()[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            gt_graph.node_attrs()[td.DEFAULT_ATTR_KEYS.T].to_list(),
            strict=True,
        )
    )
    events: list[tuple[int, tuple[int, int]]] = []
    for parent in sorted(gt_graph.node_ids()):
        daughters = sorted(gt_graph.successors(parent))
        if len(daughters) != 2:
            continue
        if all(int(node_times[daughter]) == int(node_times[parent]) + 1 for daughter in daughters):
            events.append((int(parent), (int(daughters[0]), int(daughters[1]))))
    return events


def _discover_events(
    stem: str,
    nodes: list[NodeRecord],
    edges: list[tuple[int, int]],
    gt_graph: td.graph.BaseGraph,
) -> list[OracleEvent]:
    pred_graph, _, internal_to_raw = _build_prediction_graph(nodes, edges)
    matched_by_event = match_divisions(
        pred_graph,
        gt_graph,
        scale=DEFAULT_SCALE,
        max_distance=MAX_DISTANCE_UM,
    )
    baseline_scores = score_divisions(
        pred_graph,
        gt_graph,
        scale=DEFAULT_SCALE,
        max_distance=MAX_DISTANCE_UM,
    ).scores

    events: list[OracleEvent] = []
    for gt_parent, gt_daughters in _exact_divisions(gt_graph):
        matched_graph = matched_by_event[gt_parent]
        matched_attrs = matched_graph.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        )
        gt_to_raw: dict[int, int] = {}
        for row in matched_attrs.iter_rows(named=True):
            gt_id = row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
            if gt_id is None or int(gt_id) == -1:
                continue
            gt_to_raw[int(gt_id)] = internal_to_raw[int(row[td.DEFAULT_ATTR_KEYS.NODE_ID])]
        events.append(OracleEvent(
            stem=stem,
            gt_parent=gt_parent,
            gt_daughters=gt_daughters,
            pred_parent=gt_to_raw.get(gt_parent),
            pred_daughters=(gt_to_raw.get(gt_daughters[0]), gt_to_raw.get(gt_daughters[1])),
            category=EVENT_CATEGORY.get(gt_parent, "unclassified"),
            baseline_true_positive=bool(baseline_scores.get(gt_parent, 0)),
        ))
    return events


def _estimated_node_count(gt_path: Path) -> float:
    metadata = GeffMetadata.read(gt_path)
    value = (metadata.extra or {}).get("estimated_number_of_nodes")
    if value is None:
        raise RuntimeError(f"estimated_number_of_nodes missing from {gt_path}")
    return float(value)


def _score_sample(
    stem: str,
    nodes: list[NodeRecord],
    edges: list[tuple[int, int]],
    gt_graph: td.graph.BaseGraph,
) -> tuple[dict, dict[int, int]]:
    pred_graph, _, _ = _build_prediction_graph(nodes, edges)
    result = evaluate(
        pred_graph,
        gt_graph,
        scale=DEFAULT_SCALE,
        max_distance=MAX_DISTANCE_UM,
    )
    recall = node_recall(pred_graph, gt_graph)
    metrics = per_sample_metrics(
        result,
        n_total=_estimated_node_count(GT_ROOT / f"{stem}.geff"),
        node_recall=recall,
    )
    event_scores = score_divisions(
        pred_graph,
        gt_graph,
        scale=DEFAULT_SCALE,
        max_distance=MAX_DISTANCE_UM,
    ).scores
    return metrics, {int(key): int(value) for key, value in event_scores.items()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_edges(edges: Iterable[tuple[int, int]]) -> str:
    return json.dumps(list(edges), ensure_ascii=False)


def main() -> None:
    if not BASE_CSV.exists():
        raise FileNotFoundError(BASE_CSV)
    missing_gt = [stem for stem in STEMS if not (GT_ROOT / f"{stem}.geff").exists()]
    if missing_gt:
        raise FileNotFoundError(f"Missing GT GEFF files: {missing_gt}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    nodes_by_stem, baseline_edges = _load_base_csv()
    baseline_snapshot = {stem: tuple(edges) for stem, edges in baseline_edges.items()}
    gt_graphs = {stem: _load_graph(GT_ROOT / f"{stem}.geff") for stem in STEMS}

    events: list[OracleEvent] = []
    for stem in STEMS:
        events.extend(_discover_events(
            stem,
            nodes_by_stem[stem],
            baseline_edges[stem],
            gt_graphs[stem],
        ))
    if len(events) != 5:
        raise RuntimeError(f"Expected 5 exact divisions on the fixed panel, found {len(events)}")

    mode_categories: dict[str, set[str] | None] = {
        "baseline": set(),
        "occupied_only": {"daughter_occupied"},
        "divergence_only": {"divergence_gate"},
        "all_repairable": None,
    }
    summary_rows: list[dict] = []
    per_movie_rows: list[dict] = []
    event_mode_rows: list[dict] = []
    edge_change_rows: list[dict] = []
    baseline_metrics_by_stem: dict[str, dict] = {}

    for mode, categories in mode_categories.items():
        sample_rows: list[dict] = []
        for stem in STEMS:
            selected = [
                event
                for event in events
                if event.stem == stem
                and mode != "baseline"
                and (categories is None or event.category in categories)
            ]
            mode_edges, changes = apply_oracle_events(baseline_edges[stem], selected)
            for row in changes:
                edge_change_rows.append({"mode": mode, **row})
            metrics, event_scores = _score_sample(
                stem,
                nodes_by_stem[stem],
                mode_edges,
                gt_graphs[stem],
            )
            if mode == "baseline":
                baseline_metrics_by_stem[stem] = dict(metrics)
            sample_rows.append(metrics)
            per_movie_rows.append({"mode": mode, "dataset": stem, **metrics})
            for event in [value for value in events if value.stem == stem]:
                event_mode_rows.append({
                    "mode": mode,
                    "dataset": stem,
                    "gt_parent": event.gt_parent,
                    "category": event.category,
                    "repairable": int(event.repairable),
                    "division_true_positive": int(event_scores.get(event.gt_parent, 0)),
                })
        summary_rows.append({"mode": mode, **summarise(sample_rows)})

    baseline_summary = next(row for row in summary_rows if row["mode"] == "baseline")
    if not math.isclose(
        float(baseline_summary["score"]),
        EXPECTED_BASELINE_SCORE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "Official scorer baseline mismatch: "
            f"expected {EXPECTED_BASELINE_SCORE}, got {baseline_summary['score']}"
        )

    # Measure each event independently on its own movie. This identifies which
    # local repair changes edge and division counts instead of attributing the
    # aggregate result to all events at once.
    single_event_rows: list[dict] = []
    for event in events:
        changed_edges, changes = apply_oracle_events(baseline_edges[event.stem], [event])
        metrics, event_scores = _score_sample(
            event.stem,
            nodes_by_stem[event.stem],
            changed_edges,
            gt_graphs[event.stem],
        )
        baseline = baseline_metrics_by_stem[event.stem]
        change = changes[0]
        single_event_rows.append({
            "dataset": event.stem,
            "gt_parent": event.gt_parent,
            "category": event.category,
            "repairable": int(event.repairable),
            "pred_parent": "" if event.pred_parent is None else event.pred_parent,
            "pred_daughters": _format_edges(
                [] if not event.repairable else [(int(event.pred_daughters[0]), int(event.pred_daughters[1]))]
            ),
            "baseline_tp": int(event.baseline_true_positive),
            "after_tp": int(event_scores.get(event.gt_parent, 0)),
            "removed_edges": change["removed_edges"],
            "added_edges": change["added_edges"],
            "edge_tp_delta": int(metrics["edge_tp"] - baseline["edge_tp"]),
            "edge_fp_delta": int(metrics["edge_fp"] - baseline["edge_fp"]),
            "edge_fn_delta": int(metrics["edge_fn"] - baseline["edge_fn"]),
            "division_tp_delta": int(metrics["division_tp"] - baseline["division_tp"]),
            "division_fp_delta": int(metrics["division_fp"] - baseline["division_fp"]),
            "division_fn_delta": int(metrics["division_fn"] - baseline["division_fn"]),
            "movie_adj_edge_delta": float(metrics["adj_edge_jaccard"] - baseline["adj_edge_jaccard"]),
        })

    if {stem: tuple(edges) for stem, edges in baseline_edges.items()} != baseline_snapshot:
        raise AssertionError("The baseline edge graph was mutated in place")

    for row in summary_rows:
        row["score_delta_vs_baseline"] = float(row["score"] - baseline_summary["score"])
    _write_csv(OUTPUT_ROOT / "official_summary.csv", summary_rows)
    _write_csv(OUTPUT_ROOT / "official_per_movie.csv", per_movie_rows)
    _write_csv(OUTPUT_ROOT / "event_mapping_and_single_effect.csv", single_event_rows)
    _write_csv(OUTPUT_ROOT / "event_status_by_mode.csv", event_mode_rows)
    _write_csv(OUTPUT_ROOT / "edge_changes_by_mode.csv", edge_change_rows)

    all_row = next(row for row in summary_rows if row["mode"] == "all_repairable")
    repairable_count = sum(event.repairable for event in events)
    category_labels = {
        "daughter_occupied": "子节点被错误延续边占用",
        "already_true_positive": "基础图已命中",
        "divergence_gate": "后续发散距离门拒绝",
        "unclassified": "未分类",
    }
    mapping_lines = []
    for event in events:
        daughters = "/".join("-" if value is None else str(value) for value in event.pred_daughters)
        mapping_lines.append(
            f"| {event.stem} | {event.gt_parent} | {category_labels[event.category]} | "
            f"{event.pred_parent or '-'} | {daughters} | "
            f"{'是' if event.repairable else '否'} | {int(event.baseline_true_positive)} |"
        )
    summary_lines = []
    for row in summary_rows:
        summary_lines.append(
            f"| {row['mode']} | {row['score']:.6f} | {row['score_delta_vs_baseline']:+.6f} | "
            f"{row['adj_edge_jaccard']:.6f} | {row['division_jaccard']:.6f} | "
            f"{row['division_tp']}/{row['division_fp']}/{row['division_fn']} |"
        )
    single_lines = []
    for row in single_event_rows:
        single_lines.append(
            f"| {row['dataset']} | {row['gt_parent']} | {row['repairable']} | "
            f"{row['baseline_tp']}→{row['after_tp']} | "
            f"{row['edge_tp_delta']:+d}/{row['edge_fp_delta']:+d}/{row['edge_fn_delta']:+d} | "
            f"{row['division_tp_delta']:+d}/{row['division_fp_delta']:+d}/{row['division_fn_delta']:+d} |"
        )

    report = [
        "# EXP022 分裂结构理想上限诊断",
        "",
        "本实验以 EXP019 的 FOCUS3D 关闭组最终图为唯一基础图，在同一 4 个完整验证视频上做配对比较。",
        "真实标签只用于验证集中的理想修复：删除两个子细胞的错误入边、删除父细胞指向其他节点的出边，并补上正确的两条分叉边。",
        "正式结果完全使用 patched official scorer；不生成 `submission.csv`，不得用于测试集推理。",
        "",
        "## 官方匹配后的事件可修复性",
        "",
        f"5 个真实分裂中，当前最终节点集可直接构造 {repairable_count} 个；另有 {5 - repairable_count} 个缺少 7 微米内可评分端点。",
        "这说明‘检测器原始候选里存在’不等于‘最终提交节点图里仍保留且能被官方评分器匹配’。",
        "",
        "| 视频 | GT 父节点 | EXP010 阻断类别 | 预测父节点 | 两个预测子节点 | 可修复 | 基础图已命中 |",
        "|---|---:|---|---:|---|---|---:|",
        *mapping_lines,
        "",
        "## 官方评分汇总",
        "",
        "| 处理组 | 官方 CV | 相对基础图 | 调整后普通边 | 分裂 Jaccard | 分裂 TP/FP/FN |",
        "|---|---:|---:|---:|---:|---:|",
        *summary_lines,
        "",
        "## 单事件修复的计数变化",
        "",
        "普通边和分裂列均按 `TP/FP/FN` 的变化量报告。",
        "",
        "| 视频 | GT 父节点 | 可修复 | 分裂命中 | 普通边 ΔTP/ΔFP/ΔFN | 分裂 ΔTP/ΔFP/ΔFN |",
        "|---|---:|---:|---:|---:|---:|",
        *single_lines,
        "",
        "## 结论",
        "",
        f"- 只修复当前最终节点集里可构造的事件，官方 CV 从 {baseline_summary['score']:.6f} 变为 {all_row['score']:.6f}（{all_row['score_delta_vs_baseline']:+.6f}）。",
        f"- 分裂计数从 {baseline_summary['division_tp']}/{baseline_summary['division_fp']}/{baseline_summary['division_fn']} 变为 {all_row['division_tp']}/{all_row['division_fp']}/{all_row['division_fn']}。",
        "- 总分略高于 1 不是计算错误：官方总分是调整后普通边 Jaccard 再加 `0.1 × 分裂 Jaccard`，公式没有把两者之和截断到 1。",
        "- 这个结果是有标签理想上限，不是可部署算法收益；下一步只能开发不读取标签的冲突仲裁规则，并继续用完整电影官方评分验证。",
        "- 缺少端点的事件无法靠重排现有边救回；若要覆盖它，必须检查最终节点筛选为何删除了原始候选，而不是继续放宽分叉边阈值。",
        "",
        "机器可读结果见同目录的 5 份 CSV。",
    ]
    (OUTPUT_ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_score": baseline_summary["score"],
        "all_repairable_score": all_row["score"],
        "delta": all_row["score_delta_vs_baseline"],
        "baseline_division": [
            baseline_summary["division_tp"], baseline_summary["division_fp"], baseline_summary["division_fn"]
        ],
        "all_repairable_division": [
            all_row["division_tp"], all_row["division_fp"], all_row["division_fn"]
        ],
        "repairable_events": repairable_count,
        "output": str(OUTPUT_ROOT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
