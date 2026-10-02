from __future__ import annotations

"""Audit EXP010 division node recall and a reproducible candidate graph.

This is diagnostic code only.  The prediction GEFF files contain the final
selected edges, not the private pre-ILP edge pool, so the candidate graph in
this report is an independent nearest-parent reconstruction.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
GT_ROOT = ROOT / "tmp_data" / "gt_geff"
PRED_ROOT = (
    ROOT
    / "EXP"
    / "EXP010"
    / "outputs"
    / "tracking_repo"
    / "predictions"
    / "unknown"
    / "unet_transformer_val"
    / "split_0"
)
OUT_ROOT = ROOT / "EXP" / "EXP010" / "outputs" / "division_node_candidate_audit"
STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
SCALE = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
MATCH_RADIUS_UM = 7.0
CANDIDATE_RADIUS_UM = 9.9
CANDIDATE_TOP_K = 10


def read_geff(path: Path) -> tuple[dict[int, tuple[int, float, float, float]], list[tuple[int, int]], float | None]:
    group = zarr.open_group(path, mode="r")
    ids = np.asarray(group["nodes/ids"][:], dtype=np.int64)
    t = np.asarray(group["nodes/props/t/values"][:], dtype=np.int64)
    z = np.asarray(group["nodes/props/z/values"][:], dtype=np.float64)
    y = np.asarray(group["nodes/props/y/values"][:], dtype=np.float64)
    x = np.asarray(group["nodes/props/x/values"][:], dtype=np.float64)
    nodes = {
        int(node_id): (int(tt), float(zz), float(yy), float(xx))
        for node_id, tt, zz, yy, xx in zip(ids, t, z, y, x)
    }
    edges_arr = np.asarray(group["edges/ids"][:], dtype=np.int64)
    edges = [(int(source), int(target)) for source, target in edges_arr]
    extra = group.attrs.get("geff", {}).get("extra", {})
    estimated = extra.get("estimated_number_of_nodes")
    return nodes, edges, None if estimated is None else float(estimated)


def physical_positions(nodes: dict[int, tuple[int, float, float, float]], ids: list[int]) -> np.ndarray:
    return np.asarray([nodes[node_id][1:] for node_id in ids], dtype=np.float64) * SCALE


def match_nodes_by_frame(
    pred_nodes: dict[int, tuple[int, float, float, float]],
    gt_nodes: dict[int, tuple[int, float, float, float]],
    max_distance_um: float = MATCH_RADIUS_UM,
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """One-to-one minimum-distance matching, independently for each frame."""
    pred_by_t: dict[int, list[int]] = defaultdict(list)
    gt_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, value in pred_nodes.items():
        pred_by_t[value[0]].append(node_id)
    for node_id, value in gt_nodes.items():
        gt_by_t[value[0]].append(node_id)

    gt_to_pred: dict[int, int] = {}
    pred_to_gt: dict[int, int] = {}
    distances: dict[int, float] = {}
    for frame in sorted(set(pred_by_t) | set(gt_by_t)):
        pred_ids = sorted(pred_by_t.get(frame, []))
        gt_ids = sorted(gt_by_t.get(frame, []))
        if not pred_ids or not gt_ids:
            continue
        pred_pos = physical_positions(pred_nodes, pred_ids)
        gt_pos = physical_positions(gt_nodes, gt_ids)
        cost = np.linalg.norm(gt_pos[:, None, :] - pred_pos[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            distance = float(cost[row, col])
            if distance <= max_distance_um:
                gt_id = gt_ids[int(row)]
                pred_id = pred_ids[int(col)]
                gt_to_pred[gt_id] = pred_id
                pred_to_gt[pred_id] = gt_id
                distances[gt_id] = distance
    return gt_to_pred, pred_to_gt, distances


def division_events(edges: list[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[source].append(target)
    return [
        (source, sorted(targets))
        for source, targets in sorted(outgoing.items())
        if len(targets) >= 2
    ]


def topk_candidate_ids(
    parent: tuple[int, float, float, float],
    child_ids: list[int],
    nodes: dict[int, tuple[int, float, float, float]],
    radius_um: float = CANDIDATE_RADIUS_UM,
    top_k: int = CANDIDATE_TOP_K,
) -> list[tuple[int, float]]:
    """Return nearest next-frame children inside radius, with deterministic ties."""
    parent_pos = np.asarray(parent[1:], dtype=np.float64) * SCALE
    ranked: list[tuple[int, float]] = []
    for child_id in child_ids:
        child = nodes[child_id]
        if child[0] != parent[0] + 1:
            continue
        distance = float(np.linalg.norm(parent_pos - np.asarray(child[1:], dtype=np.float64) * SCALE))
        if distance <= radius_um:
            ranked.append((child_id, distance))
    ranked.sort(key=lambda item: (item[1], item[0]))
    return ranked[:top_k]


def _node_row(video: str, role: str, gt_id: int, gt_value, pred_id, distance):
    return {
        "video": video,
        "role": role,
        "gt_node": gt_id,
        "gt_t": gt_value[0],
        "gt_z": gt_value[1],
        "gt_y": gt_value[2],
        "gt_x": gt_value[3],
        "pred_node": "" if pred_id is None else pred_id,
        "pred_t": "" if pred_id is None else gt_value[0],
        "match_distance_um": "" if distance is None else round(distance, 6),
        "detected": int(pred_id is not None),
    }


def audit_video(video: str) -> tuple[list[dict], list[dict], dict]:
    gt_nodes, gt_edges, estimated = read_geff(GT_ROOT / f"{video}.geff")
    pred_nodes, pred_edges, _ = read_geff(PRED_ROOT / f"{video}.geff")
    gt_to_pred, _, distances = match_nodes_by_frame(pred_nodes, gt_nodes)
    events = division_events(gt_edges)
    pred_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, value in pred_nodes.items():
        pred_by_t[value[0]].append(node_id)
    pred_edge_set = set(pred_edges)
    incoming = defaultdict(list)
    for source, target in pred_edges:
        incoming[target].append(source)

    node_rows = []
    candidate_rows = []
    for event_index, (gt_parent, gt_children) in enumerate(events, 1):
        children = gt_children[:2]
        mapped = [gt_to_pred.get(gt_parent), *(gt_to_pred.get(child) for child in children)]
        parent_id, child1_id, child2_id = mapped
        for role, gt_id in (("parent", gt_parent), ("daughter1", children[0]), ("daughter2", children[1])):
            node_rows.append(_node_row(video, role, gt_id, gt_nodes[gt_id], gt_to_pred.get(gt_id), distances.get(gt_id)))

        all_nodes_detected = all(node_id is not None for node_id in mapped)
        ranked: list[tuple[int, float]] = []
        if parent_id is not None:
            ranked = topk_candidate_ids(pred_nodes[parent_id], pred_by_t[pred_nodes[parent_id][0] + 1], pred_nodes)
        candidate_ids = {node_id for node_id, _ in ranked}
        radius_ids: set[int] = set()
        if parent_id is not None:
            radius_ids = {node_id for node_id, _ in topk_candidate_ids(
                pred_nodes[parent_id], pred_by_t[pred_nodes[parent_id][0] + 1], pred_nodes,
                radius_um=CANDIDATE_RADIUS_UM,
                top_k=len(pred_by_t[pred_nodes[parent_id][0] + 1]),
            )}
        d1_radius = child1_id is not None and child1_id in radius_ids
        d2_radius = child2_id is not None and child2_id in radius_ids
        d1_topk = child1_id is not None and child1_id in candidate_ids
        d2_topk = child2_id is not None and child2_id in candidate_ids
        if not all_nodes_detected:
            category = "node_missing"
        elif not (d1_topk and d2_topk):
            category = "candidate_not_generated"
        else:
            category = "candidate_generated_but_not_final"

        final_d1 = parent_id is not None and child1_id is not None and (parent_id, child1_id) in pred_edge_set
        final_d2 = parent_id is not None and child2_id is not None and (parent_id, child2_id) in pred_edge_set
        if final_d1 and final_d2:
            category = "stored_graph_fork"
        elif all_nodes_detected and d1_topk and d2_topk:
            category = "candidate_generated_but_not_final"

        rank_by_id = {node_id: rank + 1 for rank, (node_id, _) in enumerate(ranked)}
        candidate_rows.append({
            "video": video,
            "event": event_index,
            "gt_parent": gt_parent,
            "gt_child1": children[0],
            "gt_child2": children[1],
            "pred_parent": "" if parent_id is None else parent_id,
            "pred_child1": "" if child1_id is None else child1_id,
            "pred_child2": "" if child2_id is None else child2_id,
            "parent_detected": int(parent_id is not None),
            "child1_detected": int(child1_id is not None),
            "child2_detected": int(child2_id is not None),
            "candidate_radius_um": CANDIDATE_RADIUS_UM,
            "candidate_top_k": CANDIDATE_TOP_K,
            "candidate_count": len(ranked),
            "child1_in_radius": int(d1_radius),
            "child2_in_radius": int(d2_radius),
            "child1_in_topk": int(d1_topk),
            "child2_in_topk": int(d2_topk),
            "child1_rank": rank_by_id.get(child1_id, ""),
            "child2_rank": rank_by_id.get(child2_id, ""),
            "stored_edge_child1": int(final_d1),
            "stored_edge_child2": int(final_d2),
            "daughter1_incoming_sources": json.dumps(incoming.get(child1_id, [])),
            "daughter2_incoming_sources": json.dumps(incoming.get(child2_id, [])),
            "status": category,
            "candidate_preview": json.dumps(ranked[:10]),
        })

    summary = {
        "video": video,
        "gt_nodes": len(gt_nodes),
        "pred_nodes": len(pred_nodes),
        "gt_edges": len(gt_edges),
        "pred_edges": len(pred_edges),
        "estimated_true_nodes": estimated,
        "gt_divisions": len(events),
        "parent_detected": sum(row["parent_detected"] for row in candidate_rows),
        "daughter1_detected": sum(row["child1_detected"] for row in candidate_rows),
        "daughter2_detected": sum(row["child2_detected"] for row in candidate_rows),
        "candidate_pair_topk": sum(row["child1_in_topk"] and row["child2_in_topk"] for row in candidate_rows),
        "stored_graph_forks": sum(row["stored_edge_child1"] and row["stored_edge_child2"] for row in candidate_rows),
    }
    return node_rows, candidate_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_node_rows: list[dict] = []
    all_candidate_rows: list[dict] = []
    summaries = []
    for video in STEMS:
        node_rows, candidate_rows, summary = audit_video(video)
        all_node_rows.extend(node_rows)
        all_candidate_rows.extend(candidate_rows)
        summaries.append(summary)

    # The existing notebook replay is the authoritative diagnostic for the
    # safe-div post-processing stages.  Join it here without pretending that
    # its rows are the private pre-ILP candidate pool.
    safe_div_map = {}
    stage_path = ROOT / "EXP" / "EXP010" / "outputs" / "division_stage_audit" / "division_candidate_audit.csv"
    if stage_path.exists():
        with stage_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                safe_div_map[(row["video"], int(row["event"]))] = row
    for row in all_candidate_rows:
        replay = safe_div_map.get((row["video"], int(row["event"])), {})
        row["safe_div_gate_status"] = replay.get("gate_status", "")
        row["safe_div_selected"] = replay.get("selected_edge", "")

    write_csv(OUT_ROOT / "node_audit.csv", all_node_rows)
    write_csv(OUT_ROOT / "candidate_audit.csv", all_candidate_rows)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    total = {key: sum(row[key] for row in summaries) for key in (
        "gt_divisions", "parent_detected", "daughter1_detected", "daughter2_detected",
        "candidate_pair_topk", "stored_graph_forks",
    )}
    counts = defaultdict(int)
    for row in all_candidate_rows:
        counts[row["status"]] += 1
    report = [
        "# EXP010 节点与 division 候选审计",
        "",
        "审计对象：EXP010 验证集中的 4 个带标签视频。GT 通过官方比赛 GEFF 下载；预测使用 EXP010 `unet_transformer_val` GEFF（safe-div 重放前）。",
        "节点匹配使用逐帧一对一最小距离匹配，物理尺度为 `(1.625, 0.40625, 0.40625)` 微米/体素，半径为 7 微米。",
        "",
        "## 重要边界",
        "",
        "预测 GEFF 只保存该文件写入时已保留的边，不保存模型内部 ILP 之前的完整候选边池，也不是 safe-div 重放后的最终图。因此本报告的 9.9 微米、Top-10 候选图是独立重构，用于回答候选召回是否可能成立；它不是内部私有候选池的逐边复刻。",
        "",
        "## 汇总",
        "",
        f"- GT division：{total['gt_divisions']}；parent 检出：{total['parent_detected']}；daughter1 检出：{total['daughter1_detected']}；daughter2 检出：{total['daughter2_detected']}。",
        f"- 独立 Top-{CANDIDATE_TOP_K} 候选图同时覆盖两个 daughter：{total['candidate_pair_topk']} / {total['gt_divisions']}。",
        f"- 预测 GEFF 中保存的边形成完整 fork：{total['stored_graph_forks']} / {total['gt_divisions']}。这不是 safe-div 之后的最终图。",
        "",
        "按事件分类：",
        f"- 节点未检出：{counts['node_missing']}；",
        f"- 节点都检出但独立候选 Top-{CANDIDATE_TOP_K} 未覆盖：{counts['candidate_not_generated']}；",
        f"- 候选覆盖但最终边未形成 fork：{counts['candidate_generated_but_not_final']}；",
        f"- 预测 GEFF 中已经保存为 fork：{counts['stored_graph_fork']}。",
        "",
        "## 逐事件",
        "",
        "| 视频 | 事件 | 预测 parent | 两个 daughter 是否检出 | Top-10 候选覆盖 | GEFF 中两条边 | safe-div 重放 | 结论 |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for row in all_candidate_rows:
        detected = f"{row['child1_detected']}/{row['child2_detected']}"
        covered = f"{row['child1_in_topk']}/{row['child2_in_topk']}"
        stored = f"{row['stored_edge_child1']}/{row['stored_edge_child2']}"
        safe = row["safe_div_gate_status"] or "未找到重放记录"
        if row["safe_div_selected"]:
            safe = safe + " / selected=" + row["safe_div_selected"]
        report.append(f"| {row['video']} | {row['event']} | {row['pred_parent'] or '-'} | {detected} | {covered} | {stored} | {safe} | {row['status']} |")
    report.extend([
        "",
        "详细节点对应关系见 `node_audit.csv`；逐 division 候选排名、GEFF 中保存的边、daughter 入边和 safe-div 重放状态见 `candidate_audit.csv`；机器可读汇总见 `summary.json`。",
    ])
    (OUT_ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"total": total, "status_counts": dict(counts), "output": str(OUT_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
