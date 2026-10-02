"""Audit the detector-to-ILP candidate edge pool without changing predictions.

The production predictor writes candidate edges immediately before graph/ILP
selection. This module matches those coordinates to sparse GT nodes and
reports whether each GT division's two parent->daughter edges were present in
that pool, including their per-parent probability ranks.
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


def audit_npz(npz_path: Path, gt_path: Path, radius_um: float = 7.0):
    payload = np.load(npz_path)
    coords = np.asarray(payload["coords"], dtype=np.float64)
    edges = np.asarray(payload["edges"], dtype=np.float64).reshape((-1, 4))
    gt_nodes, gt_edges = read_gt(gt_path)
    mapping, distances = match_nodes(coords, gt_nodes, radius_um=radius_um)
    by_source: dict[int, list[tuple[int, float]]] = defaultdict(list)
    by_target: dict[int, list[int]] = defaultdict(list)
    for row in edges:
        source, target, prob = int(row[0]), int(row[1]), float(row[2])
        if 0 <= source < len(coords) and 0 <= target < len(coords):
            by_source[source].append((target, prob))
            by_target[target].append(source)
    for source in by_source:
        by_source[source].sort(key=lambda item: (-item[1], item[0]))

    rows = []
    for event_index, (parent, daughters) in enumerate(division_events(gt_nodes, gt_edges), 1):
        pred_parent = mapping.get(parent)
        pred_daughters = [mapping.get(daughter) for daughter in daughters]
        detected = pred_parent is not None and all(item is not None for item in pred_daughters)
        ranked = by_source.get(pred_parent, []) if pred_parent is not None else []
        ranks = {target: rank + 1 for rank, (target, _prob) in enumerate(ranked)}
        daughter_present = [item in ranks if item is not None else False for item in pred_daughters]
        if not detected:
            status = "node_missing"
        elif all(daughter_present):
            status = "both_edges_in_preilp_pool"
        elif any(daughter_present):
            status = "one_edge_in_preilp_pool"
        else:
            status = "gt_edge_absent_from_preilp_pool"
        row = {
            "video": gt_path.stem,
            "event": event_index,
            "parent_gt": parent,
            "daughter1_gt": daughters[0],
            "daughter2_gt": daughters[1],
            "parent_pred_index": "" if pred_parent is None else pred_parent,
            "daughter1_pred_index": "" if pred_daughters[0] is None else pred_daughters[0],
            "daughter2_pred_index": "" if pred_daughters[1] is None else pred_daughters[1],
            "parent_match_um": distances.get(parent, ""),
            "daughter1_match_um": distances.get(daughters[0], ""),
            "daughter2_match_um": distances.get(daughters[1], ""),
            "candidate_edges_from_parent": len(ranked),
            "daughter1_present": int(daughter_present[0]),
            "daughter2_present": int(daughter_present[1]),
            "daughter1_rank": ranks.get(pred_daughters[0], "") if pred_daughters[0] is not None else "",
            "daughter2_rank": ranks.get(pred_daughters[1], "") if pred_daughters[1] is not None else "",
            "daughter1_probability": next((prob for target, prob in ranked if target == pred_daughters[0]), "") if pred_daughters[0] is not None else "",
            "daughter2_probability": next((prob for target, prob in ranked if target == pred_daughters[1]), "") if pred_daughters[1] is not None else "",
            "daughter1_competing_incoming": len(by_target.get(pred_daughters[0], [])) if pred_daughters[0] is not None else 0,
            "daughter2_competing_incoming": len(by_target.get(pred_daughters[1], [])) if pred_daughters[1] is not None else 0,
            "status": status,
        }
        rows.append(row)
    return rows, {"coords": int(len(coords)), "candidate_edges": int(len(edges)), "gt_divisions": len(rows)}


def run(pred_dir: Path, gt_dir: Path, out_dir: Path, stems: list[str], radius_um: float = 7.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    manifests = []
    for stem in stems:
        candidates = sorted(pred_dir.glob(f"edge_candidates_preilp_*_{stem}.npz"))
        if not candidates:
            raise FileNotFoundError(f"no pre-ILP candidate manifest for {stem} in {pred_dir}")
        # A validation shard owns each stem. Refuse ambiguous duplicate files.
        if len(candidates) != 1:
            raise RuntimeError(f"expected one candidate manifest for {stem}, found {candidates}")
        rows, stats = audit_npz(candidates[0], gt_dir / f"{stem}.geff", radius_um=radius_um)
        all_rows.extend(rows)
        manifests.append({"video": stem, "path": str(candidates[0]), **stats})
    csv_path = out_dir / "preilp_candidate_audit.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    counts = defaultdict(int)
    for row in all_rows:
        counts[row["status"]] += 1
    summary = {"radius_um": radius_um, "events": len(all_rows), "status_counts": dict(counts), "manifests": manifests}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# EXP044 pre-ILP candidate edge audit",
        "",
        "This is a label-based diagnostic on complete training videos. It does not alter the graph or claim official CV/LB.",
        "Candidate edges are the thresholded, per-source/per-target capped edges emitted immediately before graph construction and ILP.",
        "",
        f"Events audited: {len(all_rows)}; node-match radius: {radius_um} um.",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    report.extend(f"| {key} | {value} |" for key, value in sorted(counts.items()))
    report += ["", "| video | event | status | daughter 1 rank/prob | daughter 2 rank/prob | incoming competitors |"]
    report += ["|---|---:|---|---|---|---|"]
    for row in all_rows:
        report.append(
            f"| {row['video']} | {row['event']} | {row['status']} | "
            f"{row['daughter1_rank']}/{row['daughter1_probability']} | "
            f"{row['daughter2_rank']}/{row['daughter2_probability']} | "
            f"{row['daughter1_competing_incoming']}/{row['daughter2_competing_incoming']} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--radius-um", type=float, default=7.0)
    args = parser.parse_args()
    print(json.dumps(run(args.pred_dir, args.gt_dir, args.out, args.stems, args.radius_um), indent=2))


if __name__ == "__main__":
    main()
