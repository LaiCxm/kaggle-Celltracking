from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "EXP" / "EXP016"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# EXP016: Pilkwang frozen detector vs original FOCUS3D

This is a node-level diagnostic only. It runs one public training volume and
the first eight annotated time points, so the original FOCUS3D instance-segmentation model
can be compared with the Pilkwang frozen TemporalUNet detector on exactly
the same raw image data. No submission is generated and no production graph
is modified.

The comparison records: per-frame node counts, 7 um GT matching, physical
distance errors, and whether the two detectors find the same or different
sparse-GT nodes. FOCUS3D is run as the original instance model; its
instance centroids are not replaced by the distilled point detector.
"""
    ),
    code(
        r'''from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exists:\n" + "\n".join(map(str, paths)))


COMP_ROOT = first_existing([
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
])
SUPPORT_ROOT = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/pilkwang/biohub-tracking-support-pack-50ep-v1"),
])
FOCUS_ROOT = first_existing([
    Path("/kaggle/input/datasets/qiweiyin/focus3d-nuclei-runtime"),
    Path("/kaggle/input/qiweiyin/focus3d-nuclei-runtime"),
])

SAMPLE_ID = "6bba_337b1b3a"
MAX_EVAL_FRAMES = 8
MATCH_RADIUS_UM = 7.0
SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
SAMPLE_PATH = COMP_ROOT / "train" / f"{SAMPLE_ID}.zarr"
GT_PATH = COMP_ROOT / "train" / f"{SAMPLE_ID}.geff"
assert SAMPLE_PATH.exists(), SAMPLE_PATH
assert GT_PATH.exists(), GT_PATH

# Kaggle's base image normally has these packages. If it does not, use the
# attached support-pack wheels without touching torch/numpy/scipy.
try:
    import zarr  # noqa: F401
    import tracksdata  # noqa: F401
    import geff  # noqa: F401
except Exception:
    wheels = SUPPORT_ROOT / "wheels"
    packages = [
        "tracksdata", "zarr==3.2.1", "numcodecs==0.15.1", "donfig==0.8.1.post1",
        "geff==1.2.0.1.1", "geff-spec==1.1.1", "polars==1.42.0",
        "polars-runtime-32==1.42.0", "bidict==0.23.1", "imagecodecs==2026.6.26",
        # tracksdata imports rustworkx at module import time; it is not present
        # in every Kaggle Python image, so keep its wheel in the offline set.
        "rustworkx==0.18.0",
        # tracksdata also imports its ILP solver eagerly, even though this
        # diagnostic disables ILP during prediction.
        "ilpy==0.6.0",
        "pyscipopt==6.2.1",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
         "--no-deps", "--find-links", str(wheels), *packages],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError((result.stdout or "")[-2000:] + (result.stderr or "")[-2000:])
    for root in ("tracksdata", "zarr", "geff", "polars"):
        for name in list(sys.modules):
            if name == root or name.startswith(root + "."):
                sys.modules.pop(name, None)

sys.path.insert(0, str(SUPPORT_ROOT / "repo" / "src"))
sys.path.insert(0, str(SUPPORT_ROOT / "repo" / "scripts"))

import torch
import zarr

if not torch.cuda.is_available():
    raise RuntimeError("EXP016 requires a Kaggle CUDA GPU")

# Labels are sparse and the first chronological frames may contain no GT.
# Select actual annotated time points so an empty slice cannot silently pass.
gt_group = zarr.open_group(str(GT_PATH), mode="r")
gt_all_t = np.asarray(gt_group["nodes/props/t/values"][:], dtype=np.int64)
annotated_frames = np.unique(gt_all_t)
if len(annotated_frames) == 0:
    raise RuntimeError(f"GT GEFF has no nodes: {GT_PATH}")
EVAL_FRAMES = annotated_frames[:MAX_EVAL_FRAMES]
PREDICT_MAX_FRAMES = int(EVAL_FRAMES[-1]) + 1
print("CUDA:", torch.cuda.get_device_name(0))
print("sample:", SAMPLE_ID, "eval_frames:", EVAL_FRAMES.tolist(),
      "predict_max_frames:", PREDICT_MAX_FRAMES, "scale_um:", SCALE_UM.tolist())
print("support:", SUPPORT_ROOT)
print("focus3d:", FOCUS_ROOT)
'''
    ),
    code(
        r'''# Load the Pilkwang frozen primary detector and reproduce its node extraction.
from predict_unet_transformer import PredictConfig, load_model, predict_video

DEVICE = torch.device("cuda")
PIL_WEIGHT = SUPPORT_ROOT / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
assert PIL_WEIGHT.exists(), PIL_WEIGHT

pil_model, pil_window, pil_downsample = load_model(PIL_WEIGHT, DEVICE)
pil_cfg = PredictConfig(
    det_threshold=0.96,
    det_tta=True,
    pool_kernel_um=3.0,
    use_ilp=False,
)

pil_start = time.perf_counter()
pil_coords, _pil_edges = predict_video(
    pil_model,
    SAMPLE_PATH,
    DEVICE,
    cfg=pil_cfg,
    window_size=pil_window,
    max_frames=PREDICT_MAX_FRAMES,
    unet_batch_size=4,
    downsample=pil_downsample,
)
pil_seconds = time.perf_counter() - pil_start
pil_coords = np.asarray(pil_coords, dtype=np.float32)
pil_coords = pil_coords[np.isin(pil_coords[:, 0].astype(np.int64), EVAL_FRAMES)]
print("Pilkwang primary nodes:", len(pil_coords), "runtime_s:", round(pil_seconds, 2))
print("Pilkwang settings:", {
    "weight": str(PIL_WEIGHT),
    "window": pil_window,
    "downsample": tuple(pil_downsample),
    "det_threshold": pil_cfg.det_threshold,
    "det_tta": pil_cfg.det_tta,
})
'''
    ),
    code(
        r'''# Run the original FOCUS3D instance-segmentation model on the same frames.
import contextlib
import shutil

import tifffile
import zarr
from scipy import ndimage

from biohub_tracking.io import open_dataset


def find_focus_weights(root: Path) -> tuple[Path, Path]:
    config = root / "configs" / "3d_test.yaml"
    candidates = [root / "models" / "model_final_nuclei.pth", root / "models " / "model_final_nuclei.pth"]
    return config, first_existing(candidates)


FOCUS_CONFIG, FOCUS_WEIGHTS = find_focus_weights(FOCUS_ROOT)
runtime_path = str(FOCUS_ROOT / "focus3d_runtime")
if runtime_path not in sys.path:
    sys.path.insert(0, runtime_path)
from focus3d.segmentation.FOCUS3D.inference_win import build_predictor, infer_volume, setup_cfg


def frame_centroids(instance_map: np.ndarray) -> np.ndarray:
    ids = np.unique(instance_map[instance_map > 0])
    if ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers = ndimage.center_of_mass(instance_map > 0, instance_map, ids)
    return np.asarray(centers, dtype=np.float32)


@contextlib.contextmanager
def quiet_runtime():
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


focus_cfg = setup_cfg(str(FOCUS_CONFIG), str(FOCUS_WEIGHTS), device="cuda")
focus_model = build_predictor(focus_cfg)
focus_model.eval()
sample_ds = open_dataset(SAMPLE_PATH, normalize=False, load_image=False, require_tracks=True)
zarr_arr = zarr.open_group(str(sample_ds.zarr_path), mode="r")["0"]
z_ratio = float(sample_ds.scale[0]) / float(sample_ds.scale[1])
focus_nodes = []
focus_instance_counts = []
VIZ_FRAME = int(EVAL_FRAMES[len(EVAL_FRAMES) // 2])
focus_instance_map_viz = None
focus_start = time.perf_counter()
work_dir = Path(tempfile.mkdtemp(prefix="exp016_focus3d_", dir="/kaggle/working"))
try:
    for t in EVAL_FRAMES:
        t = int(t)
        frame = np.ascontiguousarray(zarr_arr[t])
        image_path = work_dir / f"frame_{t:03d}.tif"
        output_dir = work_dir / f"out_{t:03d}"
        tifffile.imwrite(str(image_path), frame, metadata={"axes": "ZYX"}, photometric="minisblack")
        with quiet_runtime():
            result = infer_volume(
                image_path=image_path,
                config_file=FOCUS_CONFIG,
                weights_path=FOCUS_WEIGHTS,
                model=focus_model,
                device="cuda",
                output_dir=output_dir,
                z_ratio=z_ratio,
                lower_percentile=1.0,
                upper_percentile=99.0,
                data_loader_num_workers=0,
                cell_radius=15.0,
                background_threshold=float(np.median(frame)),
                stride=[32, 96, 96],
                batch_size=12,
                score_thresh=0.6,
                mask_thresh=0.5,
                min_edge_area=64,
                topk_postprocess=300,
                save_intermediate=False,
            )
        instance_map = np.asarray(result["instance_map"])
        if t == VIZ_FRAME:
            focus_instance_map_viz = instance_map.copy()
        centers = frame_centroids(instance_map)
        if len(centers):
            focus_nodes.extend([[t, float(z), float(y), float(x)] for z, y, x in centers])
        focus_instance_counts.append(int(len(centers)))
        image_path.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)
finally:
    shutil.rmtree(work_dir, ignore_errors=True)
focus_seconds = time.perf_counter() - focus_start
focus_nodes = np.asarray(focus_nodes, dtype=np.float32).reshape(-1, 4)
print("FOCUS3D instance-centroid nodes:", len(focus_nodes), "runtime_s:", round(focus_seconds, 2))
print("FOCUS3D settings:", {
    "weights": str(FOCUS_WEIGHTS),
    "score_thresh": 0.6,
    "mask_thresh": 0.5,
    "stride": [32, 96, 96],
    "cell_radius": 15.0,
})
'''
    ),
    code(
        r'''# Match nodes in physical units and report complementarity.
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import zarr


def geff_nodes_direct(geff_path: Path, eval_frames: np.ndarray) -> np.ndarray:
    """Read GT arrays directly to avoid an empty tracksdata node view."""
    group = zarr.open_group(str(geff_path), mode="r")
    t = np.asarray(group["nodes/props/t/values"][:], dtype=np.int64)
    z = np.asarray(group["nodes/props/z/values"][:], dtype=np.float32)
    y = np.asarray(group["nodes/props/y/values"][:], dtype=np.float32)
    x = np.asarray(group["nodes/props/x/values"][:], dtype=np.float32)
    keep = np.isin(t, eval_frames)
    nodes = np.column_stack((t[keep], z[keep], y[keep], x[keep])).astype(np.float32)
    if len(nodes) == 0:
        raise RuntimeError(f"GT GEFF contains no nodes in selected frames: {geff_path}")
    return nodes


def match_nodes(pred: np.ndarray, truth: np.ndarray, radius_um: float = 7.0):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 4)
    pairs = []
    for t in sorted(set(pred[:, 0].astype(int)) | set(truth[:, 0].astype(int))):
        p_idx = np.flatnonzero(pred[:, 0].astype(int) == t)
        g_idx = np.flatnonzero(truth[:, 0].astype(int) == t)
        if not len(p_idx) or not len(g_idx):
            continue
        p_um = pred[p_idx, 1:] * SCALE_UM
        g_um = truth[g_idx, 1:] * SCALE_UM
        distance = np.linalg.norm(p_um[:, None, :] - g_um[None, :, :], axis=2)
        gated = np.where(distance <= radius_um, distance, 1e9)
        rows, cols = linear_sum_assignment(gated)
        for r, c in zip(rows, cols):
            if gated[r, c] <= radius_um:
                pairs.append((int(p_idx[r]), int(g_idx[c]), float(distance[r, c])))
    return pairs


# Small deterministic sanity test for the matching primitive.
toy_truth = np.asarray([[0, 10, 10, 10]], dtype=np.float32)
toy_pred = np.asarray([[0, 10, 10, 11]], dtype=np.float32)
assert len(match_nodes(toy_pred, toy_truth, radius_um=7.0)) == 1
assert len(match_nodes(toy_pred, toy_truth, radius_um=0.1)) == 0

truth_nodes = geff_nodes_direct(GT_PATH, EVAL_FRAMES)
print("GT GEFF:", GT_PATH, "nodes:", len(truth_nodes))
predictions = {"Pilkwang-primary": pil_coords, "FOCUS3D-centroid": focus_nodes}
matched = {name: match_nodes(nodes, truth_nodes, MATCH_RADIUS_UM) for name, nodes in predictions.items()}

per_frame = []
for t in EVAL_FRAMES:
    t = int(t)
    gt_n = int(np.sum(truth_nodes[:, 0].astype(int) == t))
    row = {"frame": t, "gt_nodes": gt_n}
    for name, nodes in predictions.items():
        key = name.lower().replace("-", "_")
        n_pred = int(np.sum(nodes[:, 0].astype(int) == t))
        distances = [d for p, g, d in matched[name] if int(nodes[p, 0]) == t]
        row[f"{key}_nodes"] = n_pred
        row[f"{key}_matched"] = len(distances)
        row[f"{key}_recall"] = len(distances) / gt_n if gt_n else float("nan")
        row[f"{key}_precision"] = len(distances) / n_pred if n_pred else float("nan")
        row[f"{key}_mean_match_um"] = float(np.mean(distances)) if distances else float("nan")
        row[f"{key}_max_match_um"] = float(np.max(distances)) if distances else float("nan")
    per_frame.append(row)

pair_rows = []
for name, nodes in predictions.items():
    pairs = matched[name]
    distances = [d for _, _, d in pairs]
    pair_rows.append({
        "model": name,
        "pred_nodes": int(len(nodes)),
        "gt_nodes": int(len(truth_nodes)),
        "matched_nodes": int(len(pairs)),
        "gt_recall": float(len(pairs) / len(truth_nodes)) if len(truth_nodes) else float("nan"),
        "pred_precision": float(len(pairs) / len(nodes)) if len(nodes) else float("nan"),
        "mean_match_um": float(np.mean(distances)) if distances else float("nan"),
        "max_match_um": float(np.max(distances)) if distances else float("nan"),
    })

pil_gt = {g for _, g, _ in matched["Pilkwang-primary"]}
focus_gt = {g for _, g, _ in matched["FOCUS3D-centroid"]}
complementarity = {
    "gt_nodes_hit_by_both": len(pil_gt & focus_gt),
    "gt_nodes_only_pilkwang": len(pil_gt - focus_gt),
    "gt_nodes_only_focus3d": len(focus_gt - pil_gt),
    "gt_nodes_missed_by_both": len(set(range(len(truth_nodes))) - (pil_gt | focus_gt)),
    "focus3d_extra_gt_unmatched_nodes": int(len(focus_nodes) - len(focus_gt)),
    "pilkwang_extra_gt_unmatched_nodes": int(len(pil_coords) - len(pil_gt)),
}

import pandas as pd
display(pd.DataFrame(pair_rows))
display(pd.DataFrame(per_frame))
print("complementarity:", json.dumps(complementarity, indent=2))
'''
    ),
    code(
        r'''# Persist machine-readable diagnostics in the Kaggle output.
OUT_DIR = Path("/kaggle/working/exp016_focus_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(pair_rows).to_csv(OUT_DIR / "summary.csv", index=False)
pd.DataFrame(per_frame).to_csv(OUT_DIR / "per_frame.csv", index=False)
with (OUT_DIR / "comparison.json").open("w", encoding="utf-8") as f:
    json.dump({
        "sample_id": SAMPLE_ID,
        "eval_frames": EVAL_FRAMES.tolist(),
        "predict_max_frames": PREDICT_MAX_FRAMES,
        "match_radius_um": MATCH_RADIUS_UM,
        "scale_um": SCALE_UM.tolist(),
        "pilkwang_weight": str(PIL_WEIGHT),
        "focus3d_weights": str(FOCUS_WEIGHTS),
        "runtime_seconds": {"pilkwang": pil_seconds, "focus3d": focus_seconds},
        "complementarity": complementarity,
        "summary": pair_rows,
    }, f, indent=2)
np.save(OUT_DIR / "pilkwang_nodes.npy", pil_coords)
np.save(OUT_DIR / "focus3d_centroids.npy", focus_nodes)
print("Wrote:", sorted(str(p) for p in OUT_DIR.iterdir()))
'''
    ),
    code(
        r'''# Save two interpretable 2D views for one annotated 3D frame.
import matplotlib.pyplot as plt

raw_frame = np.asarray(zarr_arr[VIZ_FRAME])
projection = np.max(raw_frame.astype(np.float32), axis=0)
lo, hi = np.percentile(projection, [1.0, 99.5])
background = np.clip((projection - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

def save_overlay(nodes: np.ndarray, color: str, label: str, path: Path) -> None:
    points = nodes[nodes[:, 0].astype(np.int64) == VIZ_FRAME]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.imshow(background, cmap="gray", interpolation="nearest")
    if len(points):
        ax.scatter(points[:, 3], points[:, 2], s=8, c=color, linewidths=0, alpha=0.9)
    ax.set_title(f"{SAMPLE_ID} frame {VIZ_FRAME} - {label} ({len(points)} predicted centers)")
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

save_overlay(pil_coords, "red", "Pilkwang", OUT_DIR / f"frame_{VIZ_FRAME:03d}_pilkwang_red.png")
save_overlay(focus_nodes, "lime", "FOCUS3D", OUT_DIR / f"frame_{VIZ_FRAME:03d}_focus3d_green.png")

# A single optical Z plane with only centers close to that plane. This avoids
# the misleading density caused by projecting all 64 depth planes together.
Z_SLICE = int(raw_frame.shape[0] // 2)
plane = raw_frame[Z_SLICE].astype(np.float32)
plane_lo, plane_hi = np.percentile(plane, [1.0, 99.5])
plane_background = np.clip((plane - plane_lo) / max(plane_hi - plane_lo, 1e-6), 0.0, 1.0)

def save_plane_overlay(nodes: np.ndarray, color: str, label: str, tolerance: float, path: Path) -> int:
    points = nodes[
        (nodes[:, 0].astype(np.int64) == VIZ_FRAME)
        & (np.abs(nodes[:, 1] - Z_SLICE) <= tolerance)
    ]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.imshow(plane_background, cmap="gray", interpolation="nearest")
    if len(points):
        ax.scatter(points[:, 3], points[:, 2], s=18, c=color, linewidths=0, alpha=0.95)
    ax.set_title(
        f"{SAMPLE_ID} frame {VIZ_FRAME}, Z={Z_SLICE} - {label} "
        f"({len(points)} centers within Z +/- {tolerance:g})"
    )
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

assert focus_instance_map_viz is not None, "FOCUS3D visualization frame was not captured"
focus_centers_viz = frame_centroids(focus_instance_map_viz)
for tolerance in (2.0, 3.0, 4.0):
    save_plane_overlay(
        pil_coords, "red", "Pilkwang", tolerance,
        OUT_DIR / f"frame_{VIZ_FRAME:03d}_z{Z_SLICE:03d}_tol{int(tolerance)}_pilkwang_red.png",
    )
    selected_ids = [
        index + 1 for index, center in enumerate(focus_centers_viz)
        if abs(float(center[0]) - Z_SLICE) <= tolerance
    ]
    focus_plane_mask = np.isin(focus_instance_map_viz[Z_SLICE], selected_ids)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.imshow(plane_background, cmap="gray", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~focus_plane_mask, focus_plane_mask), cmap="Greens", alpha=0.38, interpolation="nearest")
    ax.contour(focus_plane_mask.astype(np.uint8), levels=[0.5], colors=["lime"], linewidths=0.55)
    ax.set_title(
        f"{SAMPLE_ID} frame {VIZ_FRAME}, Z={Z_SLICE} - FOCUS3D instance mask "
        f"({len(selected_ids)} instances, centroid Z +/- {tolerance:g})"
    )
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / f"frame_{VIZ_FRAME:03d}_z{Z_SLICE:03d}_tol{int(tolerance)}_focus3d_mask_green.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
print("Visualization frame:", VIZ_FRAME)
print("Wrote overlays:", sorted(p.name for p in OUT_DIR.glob("*.png")))
'''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"title": "EXP016 Pilkwang vs FOCUS3D node comparison"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = OUT_DIR / "CELL_compare_pilkwang_focus3d.ipynb"
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(out_path)
