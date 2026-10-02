from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "EXP" / "EXP010" / "CELL_infer_public_0942.ipynb"


AUDIT_SOURCE = r'''# ============================================================
# CELL 13 -- DIVISION STAGE AUDIT (held-out TRAIN only)
# Replays the already-generated validation graphs through each output
# stage and records official-style TP/FN plus GT candidate gate failures.
# This cell is diagnostic only; it does not alter submission.csv.
# ============================================================

import copy as _audit_copy
from collections import Counter as _audit_Counter

_AUDIT_STAGES = [
    "raw_ilp",
    "edge_filter",
    "motion_relink",
    "single_parent",
    "single_child",
    "gap_close",
    "gap2",
    "safe_div_geometry",
    "safe_div_post_veto",
    "safe_div_selected",
    "division_geometry_filter",
    "prune_isolated",
    "short_track",
    "final",
]
_audit_stage_rows = []
_audit_event_rows = []
_audit_candidate_rows = []


def _audit_stage_snapshot(stem, stage, nodes_by_id, edges, gt_nodes_plain, gt_edges_plain):
    pred_nodes_plain = nodes_by_id_to_plain(nodes_by_id)
    pred_edges_plain = [
        (int(e["source_id"]), int(e["target_id"])) if isinstance(e, dict)
        else (int(e[0]), int(e[1]))
        for e in edges
    ]
    p2g, g2p = match_nodes_bipartite(
        pred_nodes_plain, gt_nodes_plain, max_dist=VALIDATOR_MATCH_RADIUS_UM
    )
    div_tp, div_fp, div_fn = compute_division_confusion(
        pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, p2g, g2p
    )
    out = {}
    for source_id, target_id in pred_edges_plain:
        out.setdefault(source_id, set()).add(target_id)
    fork_count = sum(1 for targets in out.values() if len(targets) >= 2)
    _audit_stage_rows.append({
        "video": stem,
        "stage": stage,
        "nodes": len(pred_nodes_plain),
        "edges": len(pred_edges_plain),
        "forks": fork_count,
        "tp": div_tp,
        "fp_official": div_fp,
        "fn": div_fn,
    })

    gt_out = {}
    gt_in = {}
    for gs, gt in gt_edges_plain:
        gt_out.setdefault(gs, set()).add(gt)
        gt_in[gt] = gs
    components = weakly_connected_components(list(pred_nodes_plain), pred_edges_plain)
    fork_components = {
        components[n]
        for n, targets in out.items()
        if len(targets) >= 2 and n in components
    }
    for event_index, gsrc in enumerate(sorted(s for s, targets in gt_out.items() if len(targets) >= 2), 1):
        children = sorted(gt_out[gsrc])[:2]
        anchor_candidates = [gsrc]
        if gsrc in gt_in:
            anchor_candidates.append(gt_in[gsrc])
        anchor_pred = [g2p[a] for a in anchor_candidates if a in g2p]
        daughter_hits = []
        for child in children:
            lineage = {child}
            stack = [child]
            while stack:
                current = stack.pop()
                for nxt in gt_out.get(current, ()):
                    if nxt not in lineage:
                        lineage.add(nxt)
                        stack.append(nxt)
            hits = {
                components[p]
                for gid in lineage
                if (p := g2p.get(gid)) is not None and p in components
            }
            daughter_hits.append(hits)
        anchor_components = {components[p] for p in anchor_pred if p in components}
        if not anchor_pred:
            reason = "parent_anchor_missing"
        elif not daughter_hits[0] or not daughter_hits[1]:
            reason = "daughter_lineage_missing"
        elif not anchor_components:
            reason = "parent_component_missing"
        elif not any(
            comp in daughter_hits[0] and comp in daughter_hits[1] and comp in fork_components
            for comp in anchor_components
        ):
            common = bool(anchor_components & daughter_hits[0] & daughter_hits[1])
            reason = "no_fork_in_common_component" if common else "daughter_lineages_separate"
        else:
            reason = "TP"
        _audit_event_rows.append({
            "video": stem,
            "stage": stage,
            "event": event_index,
            "gt_parent": gsrc,
            "gt_child1": children[0],
            "gt_child2": children[1],
            "parent_mapped": int(bool(anchor_pred)),
            "daughter1_lineage_mapped": int(bool(daughter_hits[0])),
            "daughter2_lineage_mapped": int(bool(daughter_hits[1])),
            "same_component": int(bool(anchor_components & daughter_hits[0] & daughter_hits[1])),
            "has_fork": int(bool(anchor_components & fork_components)),
            "status": reason,
        })
    return p2g, g2p


def _audit_gt_events(gt_nodes_plain, gt_edges_plain):
    gt_out = {}
    for gs, gt in gt_edges_plain:
        gt_out.setdefault(gs, set()).add(gt)
    return [
        (gsrc, sorted(targets)[:2])
        for gsrc, targets in sorted(gt_out.items())
        if len(targets) >= 2
    ]


def _audit_candidate_gate(stem, base_nodes, base_edges, gt_nodes_plain, gt_edges_plain,
                          frame_cache, deepcenter_cache, deepcenter_bundle, selected_edges):
    """Evaluate only predicted triples that can correspond to one GT division.
    A missing mapped node/orphan is reported separately from geometric gates.
    """
    base_plain = nodes_by_id_to_plain(base_nodes)
    base_edge_pairs = [(int(e["source_id"]), int(e["target_id"])) for e in base_edges]
    _, g2p = match_nodes_bipartite(base_plain, gt_nodes_plain, max_dist=VALIDATOR_MATCH_RADIUS_UM)
    out_by_source = {}
    incoming = set()
    for e in base_edges:
        out_by_source.setdefault(int(e["source_id"]), []).append(e)
        incoming.add(int(e["target_id"]))
    ids_by_t = {}
    for nid, node in base_nodes.items():
        ids_by_t.setdefault(int(node["t"]), []).append(nid)
    selected_pairs = {(int(e["source_id"]), int(e["target_id"])) for e in selected_edges}
    gate_stats = {
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
    }
    events = _audit_gt_events(gt_nodes_plain, gt_edges_plain)
    for event_index, (gsrc, children) in enumerate(events, 1):
        parent_id = g2p.get(gsrc)
        child_ids = [g2p.get(children[0]), g2p.get(children[1])]
        status = "not_in_predicted_nodes"
        source_id = None
        existing_id = None
        candidate_id = None
        if parent_id is not None and all(cid is not None for cid in child_ids):
            source_options = [sid for sid in ids_by_t.get(int(base_nodes[parent_id]["t"]), [])
                              if sid == parent_id]
            if len(source_options) != 1:
                status = "parent_not_in_one_child_source_set"
            else:
                source_id = source_options[0]
                source_edges = out_by_source.get(source_id, [])
                if len(source_edges) != 1:
                    status = "parent_not_exactly_one_existing_child"
                else:
                    existing_id = int(source_edges[0]["target_id"])
                    existing_gt = next((i for i, cid in enumerate(child_ids) if cid == existing_id), None)
                    if existing_gt is None:
                        status = "existing_child_is_not_gt_daughter"
                    else:
                        other_id = child_ids[1 - existing_gt]
                        candidate_id = other_id
                        if candidate_id in incoming:
                            status = "gt_daughter_has_incoming_edge"
                        elif candidate_id not in ids_by_t.get(int(base_nodes[parent_id]["t"]) + 1, []):
                            status = "gt_daughter_not_in_next_frame"
                        else:
                            source = base_nodes[source_id]
                            existing = base_nodes[existing_id]
                            candidate = base_nodes[candidate_id]
                            parent_dist = edge_distance_um(source, candidate)
                            sister_dist = edge_distance_um(existing, candidate)
                            existing_dist = edge_distance_um(source, existing)
                            status = "candidate_survived_gates"
                            if existing_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                                status = "existing_child_distance_gate"
                            elif parent_dist > SAFE_DIV_MAX_UM:
                                status = "parent_distance_gate"
                            elif sister_dist > SAFE_DIV_SISTER_MAX_UM:
                                status = "sister_distance_gate"
                            else:
                                candidate_ids = [nid for nid in ids_by_t.get(int(base_nodes[parent_id]["t"]) + 1, [])
                                                 if nid not in incoming]
                                if SAFE_DIV_REQUIRE_MUTUAL_NN:
                                    positions = np.stack([_position_um(base_nodes[nid]) for nid in candidate_ids])
                                    _, nn_idx = cKDTree(positions).query(_position_um(existing))
                                    if candidate_id != candidate_ids[int(nn_idx)]:
                                        status = "mutual_nn_gate"
                                if status == "candidate_survived_gates" and SAFE_DIV_REQUIRE_DIVERGENCE:
                                    c1_succ = out_by_source.get(existing_id, [])
                                    q_succ = out_by_source.get(candidate_id, [])
                                    if len(c1_succ) != 1 or len(q_succ) != 1:
                                        status = "divergence_successor_gate"
                                    else:
                                        c1g = base_nodes.get(int(c1_succ[0]["target_id"]))
                                        qg = base_nodes.get(int(q_succ[0]["target_id"]))
                                        if c1g is None or qg is None or int(c1g["t"]) != int(source["t"]) + 2 or int(qg["t"]) != int(source["t"]) + 2:
                                            status = "divergence_grandchild_gate"
                                        elif edge_distance_um(c1g, qg) - sister_dist < SAFE_DIV_DIVERGE_UM:
                                            status = "divergence_distance_gate"
                                if status == "candidate_survived_gates" and DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                                    stem, int(candidate["t"]), node_point(candidate), deepcenter_bundle,
                                    frame_cache, deepcenter_cache, gate_stats, "safe_div", DEEPCENTER_SAFE_DIV_THRESHOLD,
                                ):
                                    status = "deepcenter_gate"
                                elif status == "candidate_survived_gates" and SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:
                                    denom = max((existing_dist + parent_dist) / 2.0, 1e-6)
                                    if abs(existing_dist - parent_dist) / denom > SAFE_DIV_SISTER_SYMMETRY_TAU:
                                        status = "symmetry_gate"
                                if status == "candidate_survived_gates" and (source_id, candidate_id) not in selected_pairs:
                                    status = "greedy_selection_cap_or_conflict"
        _audit_candidate_rows.append({
            "video": stem,
            "event": event_index,
            "gt_parent": gsrc,
            "gt_child1": children[0],
            "gt_child2": children[1],
            "pred_parent": source_id,
            "pred_existing_child": existing_id,
            "pred_candidate_child": candidate_id,
            "gate_status": status,
            "selected_edge": int((source_id, candidate_id) in selected_pairs if source_id is not None and candidate_id is not None else 0),
        })


def _audit_apply_division_geometry_filter(nodes_by_id, edges, stats):
    if not OUTPUT_DIVISION_GEOMETRY_FILTER or not edges:
        return edges
    by_source = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(edge)
    filtered = []
    for source_id, source_edges in by_source.items():
        if len(source_edges) <= 1:
            filtered.extend(source_edges)
            continue
        ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
        source = nodes_by_id[source_id]
        top1, top2 = ranked[0], ranked[1]
        d1, d2 = float(top1["distance_um"]), float(top2["distance_um"])
        sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
        valid = (max(d1, d2) <= DIV_PARENT_MAX_UM and sister <= DIV_SISTER_MAX_UM
                 and int(nodes_by_id[int(top1["target_id"])] ["t"]) == int(source["t"]) + 1
                 and int(nodes_by_id[int(top2["target_id"])] ["t"]) == int(source["t"]) + 1)
        if valid:
            filtered.extend([top1, top2])
        elif DIV_DROP_TO_SINGLE_IF_BAD:
            filtered.append(top1)
        else:
            filtered.extend(ranked)
    return filtered


def _audit_replay_video(stem):
    pred_path = next((REPO_DIR / "predictions").rglob(f"{stem}.geff"), None)
    gt_path = TRAIN_DIR / f"{stem}.geff"
    if pred_path is None or not gt_path.exists():
        print("DIVISION AUDIT: missing graph", stem)
        return
    pred_graph = graph_from_geff(pred_path)
    gt_graph = graph_from_geff(gt_path)
    gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
    nodes = {}
    for row in pred_graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes[nid] = {"node_id": nid, "t": int(row["t"]), "z": float(row["z"]),
                      "y": float(row["y"]), "x": float(row["x"])}
    raw_edges = []
    for row in pred_graph.edge_attrs().iter_rows(named=True):
        prob = row.get("edge_prob") if hasattr(row, "get") else None
        raw_edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                          "edge_prob": None if prob is None else float(prob)})
    _audit_stage_snapshot(stem, "raw_ilp", nodes, raw_edges, gt_nodes_plain, gt_edges_plain)
    edges = []
    for edge in raw_edges:
        source, target = nodes.get(int(edge["source_id"])), nodes.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            continue
        distance = edge_distance_um(source, target)
        edge["distance_um"] = distance
        if OUTPUT_EDGE_MAX_UM > 0 and distance > OUTPUT_EDGE_MAX_UM:
            continue
        edges.append(edge)
    _audit_stage_snapshot(stem, "edge_filter", nodes, edges, gt_nodes_plain, gt_edges_plain)
    stats = {k: 0 for k in ["motion_relink_edges", "motion_relink_tight_edges", "motion_relink_relaxed_edges", "motion_relink_frames", "motion_relink_skipped_large_frame", "dropped_multi_parent_edges", "dropped_multi_child_edges", "gap_candidates", "gap_pairs_selected", "gap_reused_existing", "gap_inserted_synthetic", "gap_added_nodes", "gap_added_edges", "gap_skipped_node_cap", "gap_density_nodes_scored", "gap_density_candidates_expanded", "gap_density_candidates_restricted", "gap_density_selected_outside_base", "gap_density_step_delta_milli_sum", "gap_refined_synthetic", "gap_refine_failed", "gap_refine_rejected_shift", "gap2_candidates", "gap2_pairs_selected", "gap2_added_nodes", "gap2_added_edges", "gap2_skipped_cap", "safe_division_candidates", "safe_division_geometric_candidates", "safe_divisions_added", "safe_division_skipped_cap", "safe_division_mutual_nn_rejected", "safe_division_divergence_rejected", "safe_division_symmetry_rejected", "deepcenter_gap_checked", "deepcenter_gap_bypassed_strong_motion", "deepcenter_gap_bypassed_observed_node", "deepcenter_gap_accepted", "deepcenter_gap_rejected", "deepcenter_gap_missing", "deepcenter_safe_div_checked", "deepcenter_safe_div_accepted", "deepcenter_safe_div_rejected", "deepcenter_safe_div_missing", "short_track_components_removed", "short_track_nodes_removed", "short_track_edges_removed", "short_track_filter_skipped_all", "short_track_rescue_triggered", "short_track_rescue_components", "short_track_rescue_nodes", "short_track_rescue_budget", "linefit_smoothed_nodes", "linefit_skipped_nodes"]}
    if OUTPUT_MOTION_RELINK:
        learned = {}
        for edge in edges:
            if edge.get("edge_prob") is not None:
                learned[(int(edge["source_id"]), int(edge["target_id"]))] = float(edge["edge_prob"])
        relinked = motion_relink_edges(nodes, stats, learned)
        if relinked:
            edges = relinked
    _audit_stage_snapshot(stem, "motion_relink", nodes, edges, gt_nodes_plain, gt_edges_plain)
    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            if target_id not in best or edge_sort_key(edge) > edge_sort_key(best[target_id]):
                best[target_id] = edge
        keep_ids = {id(edge) for edge in best.values()}
        edges = [edge for edge in edges if id(edge) in keep_ids]
    _audit_stage_snapshot(stem, "single_parent", nodes, edges, gt_nodes_plain, gt_edges_plain)
    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            if source_id not in best or edge_sort_key(edge) > edge_sort_key(best[source_id]):
                best[source_id] = edge
        keep_ids = {id(edge) for edge in best.values()}
        edges = [edge for edge in edges if id(edge) in keep_ids]
    _audit_stage_snapshot(stem, "single_child", nodes, edges, gt_nodes_plain, gt_edges_plain)
    old_test_dir = TEST_DIR
    globals()["TEST_DIR"] = TRAIN_DIR
    try:
        frame_cache, dc_cache = {}, {}
        nodes, edges = close_single_frame_gaps(nodes, edges, stats, dataset=stem,
                                               deepcenter_bundle=DEEPCENTER_VETO_DETECTOR,
                                               frame_cache=frame_cache, deepcenter_cache=dc_cache)
        _audit_stage_snapshot(stem, "gap_close", nodes, edges, gt_nodes_plain, gt_edges_plain)
        nodes, edges = recover_strict_gap2(nodes, edges, stats, dataset=stem)
        _audit_stage_snapshot(stem, "gap2", nodes, edges, gt_nodes_plain, gt_edges_plain)
        base_nodes = _audit_copy.deepcopy(nodes)
        base_edges = _audit_copy.deepcopy(edges)
        edges = add_safe_divisions_postlink(nodes, edges, stats, dataset=stem,
                                            deepcenter_bundle=DEEPCENTER_VETO_DETECTOR,
                                            frame_cache=frame_cache, deepcenter_cache=dc_cache)
        _audit_candidate_gate(stem, base_nodes, base_edges, gt_nodes_plain, gt_edges_plain,
                              frame_cache, dc_cache, DEEPCENTER_VETO_DETECTOR, edges)
        _audit_stage_snapshot(stem, "safe_div_geometry", base_nodes, base_edges, gt_nodes_plain, gt_edges_plain)
        _audit_stage_snapshot(stem, "safe_div_post_veto", base_nodes, base_edges, gt_nodes_plain, gt_edges_plain)
        _audit_stage_snapshot(stem, "safe_div_selected", nodes, edges, gt_nodes_plain, gt_edges_plain)
        edges = _audit_apply_division_geometry_filter(nodes, edges, stats)
        _audit_stage_snapshot(stem, "division_geometry_filter", nodes, edges, gt_nodes_plain, gt_edges_plain)
        if OUTPUT_PRUNE_ISOLATED:
            incident = {int(e["source_id"]) for e in edges} | {int(e["target_id"]) for e in edges}
            nodes = {nid: node for nid, node in nodes.items() if nid in incident}
            edges = [e for e in edges if int(e["source_id"]) in nodes and int(e["target_id"]) in nodes]
        _audit_stage_snapshot(stem, "prune_isolated", nodes, edges, gt_nodes_plain, gt_edges_plain)
        nodes, edges = filter_short_track_components(nodes, edges, stats)
        _audit_stage_snapshot(stem, "short_track", nodes, edges, gt_nodes_plain, gt_edges_plain)
        nodes = linefit_smooth_output_graph(nodes, edges, stats)
        _audit_stage_snapshot(stem, "final", nodes, edges, gt_nodes_plain, gt_edges_plain)
    finally:
        globals()["TEST_DIR"] = old_test_dir


if VALIDATOR_ENABLE and val_stems:
    for _stem in val_stems:
        _audit_replay_video(_stem)
    _audit_stage_path = WORKING_DIR / "division_stage_summary.csv"
    _audit_event_path = WORKING_DIR / "division_stage_audit.csv"
    _audit_candidate_path = WORKING_DIR / "division_candidate_audit.csv"
    import csv as _audit_csv
    for _path, _rows in [(_audit_stage_path, _audit_stage_rows), (_audit_event_path, _audit_event_rows), (_audit_candidate_path, _audit_candidate_rows)]:
        if _rows:
            with _path.open("w", newline="") as _f:
                _writer = _audit_csv.DictWriter(_f, fieldnames=list(_rows[0].keys()))
                _writer.writeheader()
                _writer.writerows(_rows)
    print("DIVISION AUDIT stage summary:")
    for _stage in _AUDIT_STAGES:
        _rows = [r for r in _audit_stage_rows if r["stage"] == _stage]
        if not _rows:
            continue
        print(f"  {_stage:<25} forks={sum(r['forks'] for r in _rows):4d} TP={sum(r['tp'] for r in _rows):2d} FN={sum(r['fn'] for r in _rows):2d}")
    print("DIVISION AUDIT candidate gate status:")
    _counts = _audit_Counter(r["gate_status"] for r in _audit_candidate_rows)
    for _status, _count in _counts.most_common():
        print(f"  {_status:<38} {_count}")
    print("Wrote", _audit_stage_path, _audit_event_path, _audit_candidate_path)
else:
    print("DIVISION AUDIT: validator disabled or no validation samples")
'''


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = payload["cells"]
    marker = "CELL 13 -- DIVISION STAGE AUDIT"
    existing = next((cell for cell in cells if marker in "".join(cell.get("source", []))), None)
    audit_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": AUDIT_SOURCE.splitlines(keepends=True),
    }
    if existing is not None:
        cells[cells.index(existing)] = audit_cell
    else:
        cells.append(audit_cell)
    NOTEBOOK.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"appended audit cell to {NOTEBOOK}")


if __name__ == "__main__":
    main()
