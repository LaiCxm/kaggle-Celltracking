"""Audit raw pre-threshold edge scores against GT division edges.

EXP045 consumes the compact top-k-per-source/target manifests emitted before
the production threshold and degree budgets.  It never changes the graph.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


def match_nodes(pred_coords: np.ndarray, gt_nodes: dict[int, tuple[int, float, float, float]], radius_um: float = 7.0):
    from scipy.optimize import linear_sum_assignment

    pred_by_t: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(pred_coords):
        pred_by_t[int(row[0])].append(index)
    gt_by_t: dict[int, list[int]] = defaultdict(list)
    for node_id, row in gt_nodes.items():
        gt_by_t[int(row[0])].append(node_id)
    gt_to_pred: dict[int, int] = {}
    distances: dict[int, float] = {}
    for t in sorted(set(pred_by_t) | set(gt_by_t)):
        pred_ids = pred_by_t.get(t, [])
        gt_ids = gt_by_t.get(t, [])
        if not pred_ids or not gt_ids:
            continue
        pred_pos = pred_coords[pred_ids, 1:4] * SCALE_UM
        gt_pos = np.asarray([gt_nodes[node_id][1:] for node_id in gt_ids], dtype=np.float64) * SCALE_UM
        cost = np.linalg.norm(gt_pos[:, None, :] - pred_pos[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(np.where(cost <= radius_um, cost, 1e9))
        for row, col in zip(rows, cols):
            if float(cost[row, col]) <= radius_um:
                gt_to_pred[gt_ids[int(row)]] = int(pred_ids[int(col)])
                distances[gt_ids[int(row)]] = float(cost[row, col])
    return gt_to_pred, distances


def read_gt(path: Path):
    import zarr

    group = zarr.open_group(str(path), mode="r")
    ids = np.asarray(group["nodes/ids"][:], dtype=np.int64)
    t = np.asarray(group["nodes/props/t/values"][:], dtype=np.int64)
    z = np.asarray(group["nodes/props/z/values"][:], dtype=np.float64)
    y = np.asarray(group["nodes/props/y/values"][:], dtype=np.float64)
    x = np.asarray(group["nodes/props/x/values"][:], dtype=np.float64)
    nodes = {int(i): (int(tt), float(zz), float(yy), float(xx)) for i, tt, zz, yy, xx in zip(ids, t, z, y, x)}
    edges = [tuple(map(int, row)) for row in np.asarray(group["edges/ids"][:], dtype=np.int64)]
    return nodes, edges


def division_events(nodes, edges):
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[int(source)].append(int(target))
    return [
        (source, sorted(targets))
        for source, targets in sorted(outgoing.items())
        if len(targets) == 2 and all(int(nodes[target][0]) == int(nodes[source][0]) + 1 for target in targets)
    ]


def _load_manifest(path: Path):
    payload = np.load(path)
    coords = np.asarray(payload["coords"], dtype=np.float64)
    final_edges = np.asarray(payload["edges"], dtype=np.float64).reshape((-1, 4))
    raw = np.asarray(payload["raw_edges"], dtype=np.float64).reshape((-1, 6))
    # source, target, probability, source rank (0 if absent), target rank (0 if absent), frame
    records: dict[tuple[int, int], dict[str, float]] = {}
    for source, target, prob, source_rank, target_rank, frame in raw:
        key = (int(source), int(target))
        record = records.setdefault(key, {
            "probability": float(prob),
            "source_rank": 0.0,
            "target_rank": 0.0,
            "frame": float(frame),
        })
        record["probability"] = max(record["probability"], float(prob))
        record["source_rank"] = max(record["source_rank"], float(source_rank))
        record["target_rank"] = max(record["target_rank"], float(target_rank))
    final_set = {(int(row[0]), int(row[1])) for row in final_edges}
    return coords, final_set, records


def audit_npz(npz_path: Path, gt_path: Path, threshold: float = 0.48, max_children: int = 2, max_parents: int = 1, radius_um: float = 7.0):
    coords, final_set, raw_records = _load_manifest(npz_path)
    gt_nodes, gt_edges = read_gt(gt_path)
    mapping, distances = match_nodes(coords, gt_nodes, radius_um=radius_um)
    rows = []
    for event_index, (parent, daughters) in enumerate(division_events(gt_nodes, gt_edges), 1):
        pred_parent = mapping.get(parent)
        pred_daughters = [mapping.get(daughter) for daughter in daughters]
        event_rows = []
        for daughter_gt, pred_daughter in zip(daughters, pred_daughters):
            key = (pred_parent, pred_daughter) if pred_parent is not None and pred_daughter is not None else None
            record = raw_records.get(key, {}) if key is not None else {}
            probability = record.get("probability", "")
            source_rank = int(record["source_rank"]) if record.get("source_rank", 0) else ""
            target_rank = int(record["target_rank"]) if record.get("target_rank", 0) else ""
            final_present = int(key in final_set) if key is not None else 0
            threshold_pass = int(float(probability) > threshold) if probability != "" else ""
            source_budget_rejected = int(bool(threshold_pass and source_rank and source_rank > max_children)) if threshold_pass != "" else ""
            target_budget_rejected = int(bool(threshold_pass and target_rank and target_rank > max_parents)) if threshold_pass != "" else ""
            if final_present:
                status = "final_candidate"
            elif source_budget_rejected:
                status = "source_budget_rejected"
            elif target_budget_rejected:
                status = "target_budget_rejected"
            elif record:
                status = "prethreshold_topk_only"
            else:
                status = "not_in_saved_topk"
            event_rows.append({
                "daughter_gt": daughter_gt,
                "daughter_pred_index": "" if pred_daughter is None else pred_daughter,
                "daughter_match_um": distances.get(daughter_gt, ""),
                "raw_probability": probability,
                "source_rank": source_rank,
                "target_rank": target_rank,
                "threshold_pass": threshold_pass,
                "source_budget_rejected": source_budget_rejected,
                "target_budget_rejected": target_budget_rejected,
                "final_candidate": final_present,
                "status": status,
            })
        for child_index, item in enumerate(event_rows, 1):
            rows.append({
                "video": gt_path.stem,
                "event": event_index,
                "parent_gt": parent,
                "daughter1_gt": daughters[0],
                "daughter2_gt": daughters[1],
                "parent_pred_index": "" if pred_parent is None else pred_parent,
                "daughter_index": child_index,
                **item,
            })
    return rows, {"coords": int(len(coords)), "final_edges": int(len(final_set)), "raw_topk_edges": int(len(raw_records)), "gt_divisions": len(rows) // 2}


def run(pred_dir: Path, gt_dir: Path, out_dir: Path, stems: list[str], threshold: float = 0.48, max_children: int = 2, max_parents: int = 1, radius_um: float = 7.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    manifests = []
    for stem in stems:
        candidates = sorted(pred_dir.glob(f"edge_candidates_prethreshold_*_{stem}.npz"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one pre-threshold manifest for {stem}, found {candidates}")
        rows, stats = audit_npz(candidates[0], gt_dir / f"{stem}.geff", threshold, max_children, max_parents, radius_um)
        all_rows.extend(rows)
        manifests.append({"video": stem, "path": str(candidates[0]), **stats})
    csv_path = out_dir / "prethreshold_candidate_audit.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    counts = defaultdict(int)
    for row in all_rows:
        counts[row["status"]] += 1
    summary = {
        "threshold": threshold,
        "max_children": max_children,
        "max_parents": max_parents,
        "radius_um": radius_um,
        "gt_divisions": len(all_rows) // 2,
        "gt_edges": len(all_rows),
        "status_counts": dict(counts),
        "manifests": manifests,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# EXP045 pre-threshold candidate edge audit", "",
        "This is a label-based diagnostic. It does not alter the graph or claim official CV/LB.",
        f"Threshold={threshold}; source budget={max_children}; target budget={max_parents}; node-match radius={radius_um} um.",
        "",
        f"GT division events: {len(all_rows) // 2}; GT parent→daughter edges: {len(all_rows)}.",
        "",
        "| status | count |", "|---|---:|",
    ]
    report.extend(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    report += ["", "| video | event | daughter | raw probability | source rank | target rank | threshold | source budget | target budget | final | status |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in all_rows:
        report.append(
            f"| {row['video']} | {row['event']} | {row['daughter_index']} | {row['raw_probability']} | "
            f"{row['source_rank']} | {row['target_rank']} | {row['threshold_pass']} | "
            f"{row['source_budget_rejected']} | {row['target_budget_rejected']} | {row['final_candidate']} | {row['status']} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.48)
    parser.add_argument("--max-children", type=int, default=2)
    parser.add_argument("--max-parents", type=int, default=1)
    parser.add_argument("--radius-um", type=float, default=7.0)
    args = parser.parse_args()
    print(json.dumps(run(args.pred_dir, args.gt_dir, args.out, args.stems, args.threshold, args.max_children, args.max_parents, args.radius_um), indent=2))


if __name__ == "__main__":
    main()
