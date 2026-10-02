from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "EXP" / "EXP010" / "outputs" / "tracking_repo" / "predictions" / "unknown" / "unet_transformer_val" / "split_0"
NOTEBOOK = ROOT / "EXP" / "EXP010" / "CELL_infer_public_0942.ipynb"
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
MOTION_RELINK_TIGHT_UM = 6.0
MOTION_RELINK_RELAXED_UM = 10.0
MOTION_RELINK_VELOCITY_WEIGHT = 0.5
MOTION_RELINK_LEARNED_BONUS = 1.0
MOTION_RELINK_MAX_FRAME_NODES = 2600
OUTPUT_MOTION_RELINK = True


def load_motion_relink():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(payload["cells"][9]["source"])
    start = source.index("def _position_um")
    end = source.index("def close_single_frame_gaps", start)
    exec(source[start:end], globals())


def load_graph(stem: str):
    root = zarr.open(str(PREDICTIONS / f"{stem}.geff"), mode="r")
    ids = np.asarray(root["nodes/ids"])
    ts = np.asarray(root["nodes/props/t/values"])
    zs = np.asarray(root["nodes/props/z/values"])
    ys = np.asarray(root["nodes/props/y/values"])
    xs = np.asarray(root["nodes/props/x/values"])
    nodes = {
        int(nid): {"node_id": int(nid), "t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for nid, t, z, y, x in zip(ids, ts, zs, ys, xs)
    }
    edge_ids = np.asarray(root["edges/ids"])
    probs = np.asarray(root["edges/props/edge_prob/values"])
    learned = {(int(s), int(t)): float(p) for (s, t), p in zip(edge_ids, probs)}
    return nodes, edge_ids, learned


def main():
    load_motion_relink()
    cases = {
        "44b6_12dfb391": 32646,
        "44b6_267148e4": 839,
        "6bba_07e24132": 14487,
    }
    for stem, target in cases.items():
        nodes, raw_edges, learned = load_graph(stem)
        raw_in = [(int(s), int(t)) for s, t in raw_edges if int(t) == target]
        target_t = int(nodes[target]["t"])
        # The relinker advances in time, so later frames cannot affect the
        # predecessor selected for this target.
        nodes = {nid: node for nid, node in nodes.items() if int(node["t"]) <= target_t}
        stats = {
            "motion_relink_tight_edges": 0,
            "motion_relink_relaxed_edges": 0,
            "motion_relink_frames": 0,
            "motion_relink_edges": 0,
            "motion_relink_skipped_large_frame": 0,
        }
        motion = motion_relink_edges(nodes, stats, learned)
        motion_in = [
            (int(edge["source_id"]), int(edge["target_id"]), edge["motion_pass"])
            for edge in motion
            if int(edge["target_id"]) == target
        ]
        print(stem, "target", target, "raw_in", raw_in, "motion_in", motion_in)


if __name__ == "__main__":
    main()
