"""Local FOCUS3D instance-mask evidence for conservative division rewiring.

The module deliberately has no competition-specific imports.  It operates on
integer ``instance_map`` arrays and plain node/edge dictionaries so that the
geometry and graph invariants can be tested on CPU before the GPU notebook is
submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable
from pathlib import Path
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


class FocusMaskProvider:
    """Lazy adapter around the original FOCUS3D runtime dataset.

    The large model is loaded only when a local division conflict needs a
    frame.  Failures are converted to an unavailable provider so the caller
    can retain the EXP017 graph unchanged.
    """

    def __init__(self, cache_dir: str | Path, scale_um=(1.625, 0.40625, 0.40625), max_frames: int = 80):
        self.cache_dir = Path(cache_dir)
        self.scale_um = tuple(float(v) for v in scale_um)
        self.max_frames = int(max_frames)
        self._maps: dict[tuple[str, int], np.ndarray] = {}
        self._model = None
        self._config = None
        self._weights = None
        self.error: str | None = None
        self.calls = 0

    @property
    def available(self) -> bool:
        return self.error is None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        roots = [
            Path(os.environ.get("BIOHUB_FOCUS3D_ROOT", "")),
            Path("/kaggle/input/datasets/qiweiyin/focus3d-nuclei-runtime"),
            Path("/kaggle/input/focus3d-nuclei-runtime"),
        ]
        root = next((p for p in roots if str(p) and (p / "focus3d_runtime").exists()), None)
        if root is None:
            self.error = "FOCUS3D runtime dataset not mounted"
            return False
        try:
            runtime = root / "focus3d_runtime"
            if str(runtime) not in sys.path:
                sys.path.insert(0, str(runtime))
            from focus3d.segmentation.FOCUS3D.inference_win import build_predictor, infer_volume, setup_cfg
            config = root / "configs" / "3d_test.yaml"
            weights = root / "models" / "model_final_nuclei.pth"
            if not weights.exists():
                weights = root / "models " / "model_final_nuclei.pth"
            if not config.exists() or not weights.exists():
                raise FileNotFoundError(f"missing config/weights under {root}")
            self._config, self._weights = config, weights
            self._infer_volume = infer_volume
            self._model = build_predictor(setup_cfg(str(config), str(weights), device="cuda"))
            self._model.eval()
            return True
        except Exception as exc:  # pragma: no cover - depends on Kaggle runtime
            self.error = f"FOCUS3D load failed: {type(exc).__name__}: {exc}"
            return False

    def get(self, dataset: str, t: int, frame_loader: Callable[[int], np.ndarray]) -> np.ndarray | None:
        key = (str(dataset), int(t))
        if key in self._maps:
            return self._maps[key]
        if len(self._maps) >= self.max_frames or not self._load():
            return None
        try:
            import tifffile
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{dataset}__t{int(t):04d}.tif"
            if not path.exists():
                tifffile.imwrite(path, np.asarray(frame_loader(int(t))))
            result = self._infer_volume(
                image_path=str(path), config_file=str(self._config), weights_path=str(self._weights),
                model=self._model, device="cuda", output_dir=str(self.cache_dir / "outputs"),
                z_ratio=self.scale_um[0] / self.scale_um[1], lower_percentile=1.0,
                upper_percentile=99.0, data_loader_num_workers=0, cell_radius=15.0,
                background_threshold=float(np.median(frame_loader(int(t)))), stride=[32, 96, 96],
                batch_size=12, score_thresh=0.6, mask_thresh=0.5, min_edge_area=64,
                topk_postprocess=300, save_intermediate=False,
            )
            arr = np.asarray(result["instance_map"], dtype=np.int32)
            if arr.ndim != 3:
                raise ValueError(f"instance_map shape={arr.shape}")
            self._maps[key] = arr
            self.calls += 1
            return arr
        except Exception as exc:  # pragma: no cover - depends on Kaggle runtime
            self.error = f"FOCUS3D inference failed at {dataset}/t{t}: {type(exc).__name__}: {exc}"
            return None


def build_mask_frame_matches(
    nodes_by_id: dict[int, dict[str, object]],
    frames: dict[int, np.ndarray],
    scale_um: tuple[float, float, float],
    radius_um: float = 7.0,
) -> dict[int, dict[int, MaskInstance]]:
    """Match only nodes in each requested frame to that frame's instances."""
    out: dict[int, dict[int, MaskInstance]] = {}
    for t, arr in frames.items():
        frame_nodes = {i: n for i, n in nodes_by_id.items() if int(n["t"]) == int(t)}
        out[int(t)] = match_nodes_to_instances(frame_nodes, arr, scale_um, radius_um)
    return out


@dataclass(frozen=True)
class MaskInstance:
    label: int
    coords: np.ndarray  # N x 3 integer voxel coordinates (z, y, x)
    centroid: tuple[float, float, float]
    volume: int


def extract_instances(instance_map: np.ndarray) -> dict[int, MaskInstance]:
    """Extract non-background instances while retaining their full voxels."""
    arr = np.asarray(instance_map)
    if arr.ndim != 3:
        raise ValueError(f"instance_map must be 3-D, got {arr.shape}")
    labels = np.unique(arr)
    out: dict[int, MaskInstance] = {}
    for raw_label in labels:
        label = int(raw_label)
        if label <= 0:
            continue
        coords = np.argwhere(arr == raw_label).astype(np.int32, copy=False)
        if coords.size == 0:
            continue
        c = tuple(float(v) for v in coords.mean(axis=0))
        out[label] = MaskInstance(label, coords, c, int(coords.shape[0]))
    return out


def _pdist_um(a: Iterable[float], b: Iterable[float], scale_um: tuple[float, float, float]) -> float:
    d = np.asarray(tuple(a), dtype=np.float64) - np.asarray(tuple(b), dtype=np.float64)
    s = np.asarray(scale_um, dtype=np.float64)
    return float(np.sqrt(np.sum((d * s) ** 2)))


def match_nodes_to_instances(
    nodes: dict[int, dict[str, object]],
    instance_map: np.ndarray,
    scale_um: tuple[float, float, float],
    radius_um: float = 7.0,
) -> dict[int, MaskInstance]:
    """One-to-one node/instance attachment with a physical-distance gate.

    Ambiguous or out-of-radius matches are omitted.  The caller can therefore
    fall back to the base graph when FOCUS3D is unavailable or uncertain.
    """
    instances = extract_instances(instance_map)
    if not nodes or not instances:
        return {}
    node_ids = sorted(nodes)
    labels = sorted(instances)
    node_points = [
        (float(nodes[n]["z"]), float(nodes[n]["y"]), float(nodes[n]["x"]))
        for n in node_ids
    ]
    inst_points = [instances[k].centroid for k in labels]
    cost = np.asarray(
        [[_pdist_um(a, b, scale_um) for b in inst_points] for a in node_points],
        dtype=np.float64,
    )
    rows, cols = linear_sum_assignment(cost)
    result: dict[int, MaskInstance] = {}
    for r, c in zip(rows, cols):
        if float(cost[r, c]) <= float(radius_um):
            result[node_ids[int(r)]] = instances[labels[int(c)]]
    return result


def _shifted_set(mask: MaskInstance, shift: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    d = np.asarray(shift, dtype=np.int64)
    return {tuple((row + d).tolist()) for row in mask.coords}


def mask_iou_after_shift(
    source: MaskInstance,
    target: MaskInstance,
    shift: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Return IoU and overlap fractions after translating source to target time."""
    src = _shifted_set(source, shift)
    dst = {tuple(row.tolist()) for row in target.coords}
    inter = len(src & dst)
    union = len(src | dst)
    if union == 0:
        return 0.0, 0.0, 0.0
    return float(inter / union), float(inter / max(len(src), 1)), float(inter / max(len(dst), 1))


def _voxel_shift(source_node: dict[str, object], target_node: dict[str, object]) -> tuple[int, int, int]:
    return tuple(
        int(round(float(target_node[k]) - float(source_node[k]))) for k in ("z", "y", "x")
    )


def continuation_mask_score(
    source_node: dict[str, object],
    target_node: dict[str, object],
    source_mask: MaskInstance,
    target_mask: MaskInstance,
) -> float:
    """Contour agreement for an ordinary one-frame continuation."""
    iou, src_cov, dst_cov = mask_iou_after_shift(
        source_mask, target_mask, _voxel_shift(source_node, target_node)
    )
    # IoU is the primary contour signal; coverage terms make tiny fragments
    # unable to win solely because their union is small.
    return float(0.60 * iou + 0.20 * src_cov + 0.20 * dst_cov)


def division_mask_score(
    parent_node: dict[str, object],
    daughter1_node: dict[str, object],
    daughter2_node: dict[str, object],
    parent_mask: MaskInstance,
    daughter1_mask: MaskInstance,
    daughter2_mask: MaskInstance,
) -> tuple[float, dict[str, float]]:
    """Score whether two daughter contours explain one parent contour.

    The parent is translated separately along each observed daughter motion,
    then the union coverage and daughter volume balance are measured from the
    actual 3-D voxel sets.  No centroid-only feature is used in this score.
    """
    d1_shift = _voxel_shift(parent_node, daughter1_node)
    d2_shift = _voxel_shift(parent_node, daughter2_node)
    p1 = _shifted_set(parent_mask, d1_shift)
    p2 = _shifted_set(parent_mask, d2_shift)
    d1 = {tuple(row.tolist()) for row in daughter1_mask.coords}
    d2 = {tuple(row.tolist()) for row in daughter2_mask.coords}
    daughters = d1 | d2
    if not daughters:
        return 0.0, {"union_parent_coverage": 0.0, "volume_balance": 0.0, "daughter_iou": 0.0}
    parent_union = p1 | p2
    covered = len(parent_union & daughters) / max(len(parent_union), 1)
    balance = min(daughter1_mask.volume, daughter2_mask.volume) / max(
        daughter1_mask.volume, daughter2_mask.volume, 1
    )
    daughter_iou = len(d1 & d2) / max(len(d1 | d2), 1)
    # Distinct instance IDs normally make daughter_iou zero.  It is retained as
    # a sanity penalty for malformed/overlapping masks, not as positive evidence.
    score = 0.65 * covered + 0.35 * balance - 0.25 * daughter_iou
    return float(max(0.0, min(1.0, score))), {
        "union_parent_coverage": float(covered),
        "volume_balance": float(balance),
        "daughter_iou": float(daughter_iou),
    }


def local_mask_rewire(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    masks_by_frame: dict[int, dict[int, MaskInstance]],
    *,
    max_parent_um: float = 14.0,
    min_division_score: float = 0.20,
    min_margin: float = 0.10,
    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Conservatively rewire occupied daughter conflicts using mask evidence.

    For each P->D1 and Q->D conflict, compare H0 (Q->D) with H1 (P->D).
    Exactly one old edge is removed and one new edge is added only when H1 has
    a clear contour margin.  Existing node coordinates and all unrelated edges
    are untouched.
    """
    if edge_distance_um is None:
        edge_distance_um = lambda a, b: math.dist(
            [float(a[k]) for k in ("z", "y", "x")], [float(b[k]) for k in ("z", "y", "x")]
        )
    out = [dict(e) for e in edges]
    stats = {"focus_checked": 0, "focus_accepted": 0, "focus_rejected_margin": 0, "focus_missing_mask": 0}
    changed: set[tuple[int, int]] = set()
    active_pairs = {(int(e["source_id"]), int(e["target_id"])) for e in out}
    by_source: dict[int, list[dict[str, object]]] = {}
    by_target: dict[int, list[dict[str, object]]] = {}
    for e in out:
        s, t = int(e["source_id"]), int(e["target_id"])
        by_source.setdefault(s, []).append(e)
        by_target.setdefault(t, []).append(e)
    for p, p_edges in list(by_source.items()):
        if len(p_edges) != 1:
            continue
        e1 = p_edges[0]
        d1 = int(e1["target_id"])
        if (p, d1) not in active_pairs:
            continue
        if p not in nodes_by_id or d1 not in nodes_by_id:
            continue
        pn, d1n = nodes_by_id[p], nodes_by_id[d1]
        if int(d1n["t"]) != int(pn["t"]) + 1:
            continue
        frame_p, frame_d = int(pn["t"]), int(d1n["t"])
        pm = masks_by_frame.get(frame_p, {}).get(p)
        m1 = masks_by_frame.get(frame_d, {}).get(d1)
        if pm is None or m1 is None:
            continue
        for d, incoming in list(by_target.items()):
            if d == d1 or len(incoming) != 1 or d not in nodes_by_id:
                continue
            q = int(incoming[0]["source_id"])
            if q == p or q not in nodes_by_id:
                continue
            dn, qn = nodes_by_id[d], nodes_by_id[q]
            if int(dn["t"]) != frame_d or int(qn["t"]) != frame_p:
                continue
            if edge_distance_um(pn, dn) > max_parent_um:
                continue
            md = masks_by_frame.get(frame_d, {}).get(d)
            qm = masks_by_frame.get(frame_p, {}).get(q)
            if md is None or qm is None:
                stats["focus_missing_mask"] += 1
                continue
            stats["focus_checked"] += 1
            h0 = continuation_mask_score(qn, dn, qm, md)
            h1, _ = division_mask_score(pn, d1n, dn, pm, m1, md)
            if h1 < min_division_score or h1 <= h0 + min_margin:
                stats["focus_rejected_margin"] += 1
                continue
            key = (q, d)
            if key in changed:
                continue
            out = [e for e in out if not (int(e["source_id"]) == q and int(e["target_id"]) == d)]
            active_pairs.discard((q, d))
            out.append({
                "source_id": p,
                "target_id": d,
                "edge_prob": None,
                "distance_um": float(edge_distance_um(pn, dn)),
                "focus_mask_division": 1,
                "focus_h0": float(h0),
                "focus_h1": float(h1),
            })
            active_pairs.add((p, d))
            changed.add(key)
            stats["focus_accepted"] += 1
            # A parent may receive at most one local replacement in this pass.
            break
    return out, stats


def apply_focus_mask_arbitration(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    *,
    dataset: str,
    provider: FocusMaskProvider | None,
    frame_loader: Callable[[int], np.ndarray],
    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    radius_um: float = 7.0,
    max_parent_um: float = 14.0,
    min_division_score: float = 0.20,
    min_margin: float = 0.10,
    max_conflicts: int = 4,
    max_frames_per_dataset: int = 2,
    stats: dict[str, int] | None = None,
    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,
) -> list[dict[str, object]]:
    """Load only frames touched by an occupied-daughter conflict and rewire."""
    if provider is None or not provider.available:
        return edges
    by_source: dict[int, list[dict[str, object]]] = {}
    by_target: dict[int, list[dict[str, object]]] = {}
    for e in edges:
        by_source.setdefault(int(e["source_id"]), []).append(e)
        by_target.setdefault(int(e["target_id"]), []).append(e)
    ranked_conflicts: list[tuple[float, int, int, int, int]] = []
    for p, es in by_source.items():
        if len(es) != 1 or p not in nodes_by_id:
            continue
        d1 = int(es[0]["target_id"])
        if d1 not in nodes_by_id:
            continue
        pn, d1n = nodes_by_id[p], nodes_by_id[d1]
        if int(d1n["t"]) != int(pn["t"]) + 1:
            continue
        for d, inc in by_target.items():
            if d == d1 or len(inc) != 1 or d not in nodes_by_id:
                continue
            q = int(inc[0]["source_id"])
            if q == p or q not in nodes_by_id:
                continue
            if int(nodes_by_id[d]["t"]) != int(d1n["t"]):
                continue
            if edge_distance_um is None:
                dist = math.dist(
                    [float(pn[k]) for k in ("z", "y", "x")],
                    [float(nodes_by_id[d][k]) for k in ("z", "y", "x")],
                )
            else:
                dist = edge_distance_um(pn, nodes_by_id[d])
            if dist <= max_parent_um:
                sister_dist = edge_distance_um(d1n, nodes_by_id[d]) if edge_distance_um else math.dist(
                    [float(d1n[k]) for k in ("z", "y", "x")],
                    [float(nodes_by_id[d][k]) for k in ("z", "y", "x")],
                )
                # Keep only compact, division-like conflicts.  Without this
                # gate every ordinary occupied target would trigger FOCUS3D.
                if sister_dist <= max_parent_um:
                    ranked_conflicts.append((float(dist + 0.35 * sister_dist), p, d1, q, d))
    ranked_conflicts.sort(key=lambda x: x[0])
    selected = ranked_conflicts[: max(0, int(max_conflicts))]
    requested = set()
    for _, p, d1, q, d in selected:
        requested.update((int(nodes_by_id[p]["t"]), int(nodes_by_id[d]["t"])))
    if max_frames_per_dataset > 0 and len(requested) > max_frames_per_dataset:
        keep_frames = {int(nodes_by_id[p]["t"]) for _, p, _, _, _ in selected[:max_frames_per_dataset]}
        requested = keep_frames | {int(nodes_by_id[p]["t"]) + 1 for _, p, _, _, _ in selected[:max_frames_per_dataset]}
    if not selected or not requested:
        return edges
    frame_maps: dict[int, np.ndarray] = {}
    for t in sorted(requested):
        arr = provider.get(dataset, t, frame_loader)
        if arr is not None:
            frame_maps[t] = arr
    if len(frame_maps) < 2:
        if stats is not None:
            stats["focus_runtime_unavailable"] = stats.get("focus_runtime_unavailable", 0) + 1
        return edges
    matches = build_mask_frame_matches(nodes_by_id, frame_maps, scale_um, radius_um)
    masks_by_frame = {t: m for t, m in matches.items()}
    out, local_stats = local_mask_rewire(
        nodes_by_id, edges, masks_by_frame, max_parent_um=max_parent_um,
        min_division_score=min_division_score, min_margin=min_margin,
        edge_distance_um=edge_distance_um,
    )
    if stats is not None:
        for k, v in local_stats.items():
            stats[k] = stats.get(k, 0) + int(v)
        stats["focus_frames_loaded"] = stats.get("focus_frames_loaded", 0) + len(frame_maps)
        stats["focus_nodes_matched"] = stats.get("focus_nodes_matched", 0) + sum(
            len(m) for m in masks_by_frame.values()
        )
    return out
