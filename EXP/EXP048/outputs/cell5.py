
# EXP019 FOCUS3D strategy modules (self-contained copies)
_focus_base_source = '"""Local FOCUS3D instance-mask evidence for conservative division rewiring.\n\nThe module deliberately has no competition-specific imports.  It operates on\ninteger ``instance_map`` arrays and plain node/edge dictionaries so that the\ngeometry and graph invariants can be tested on CPU before the GPU notebook is\nsubmitted.\n"""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nimport math\nfrom typing import Callable, Iterable\nfrom pathlib import Path\nimport os\nimport sys\n\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment\n\n\nclass FocusMaskProvider:\n    """Lazy adapter around the original FOCUS3D runtime dataset.\n\n    The large model is loaded only when a local division conflict needs a\n    frame.  Failures are converted to an unavailable provider so the caller\n    can retain the EXP017 graph unchanged.\n    """\n\n    def __init__(self, cache_dir: str | Path, scale_um=(1.625, 0.40625, 0.40625), max_frames: int = 80):\n        self.cache_dir = Path(cache_dir)\n        self.scale_um = tuple(float(v) for v in scale_um)\n        self.max_frames = int(max_frames)\n        self._maps: dict[tuple[str, int], np.ndarray] = {}\n        self._model = None\n        self._config = None\n        self._weights = None\n        self.error: str | None = None\n        self.calls = 0\n\n    @property\n    def available(self) -> bool:\n        return self.error is None\n\n    def _load(self) -> bool:\n        if self._model is not None:\n            return True\n        roots = [\n            Path(os.environ.get("BIOHUB_FOCUS3D_ROOT", "")),\n            Path("/kaggle/input/datasets/qiweiyin/focus3d-nuclei-runtime"),\n            Path("/kaggle/input/focus3d-nuclei-runtime"),\n        ]\n        root = next((p for p in roots if str(p) and (p / "focus3d_runtime").exists()), None)\n        if root is None:\n            self.error = "FOCUS3D runtime dataset not mounted"\n            return False\n        try:\n            runtime = root / "focus3d_runtime"\n            if str(runtime) not in sys.path:\n                sys.path.insert(0, str(runtime))\n            from focus3d.segmentation.FOCUS3D.inference_win import build_predictor, infer_volume, setup_cfg\n            config = root / "configs" / "3d_test.yaml"\n            weights = root / "models" / "model_final_nuclei.pth"\n            if not weights.exists():\n                weights = root / "models " / "model_final_nuclei.pth"\n            if not config.exists() or not weights.exists():\n                raise FileNotFoundError(f"missing config/weights under {root}")\n            self._config, self._weights = config, weights\n            self._infer_volume = infer_volume\n            self._model = build_predictor(setup_cfg(str(config), str(weights), device="cuda"))\n            self._model.eval()\n            return True\n        except Exception as exc:  # pragma: no cover - depends on Kaggle runtime\n            self.error = f"FOCUS3D load failed: {type(exc).__name__}: {exc}"\n            return False\n\n    def get(self, dataset: str, t: int, frame_loader: Callable[[int], np.ndarray]) -> np.ndarray | None:\n        key = (str(dataset), int(t))\n        if key in self._maps:\n            return self._maps[key]\n        if len(self._maps) >= self.max_frames or not self._load():\n            return None\n        try:\n            import tifffile\n            self.cache_dir.mkdir(parents=True, exist_ok=True)\n            path = self.cache_dir / f"{dataset}__t{int(t):04d}.tif"\n            if not path.exists():\n                tifffile.imwrite(path, np.asarray(frame_loader(int(t))))\n            result = self._infer_volume(\n                image_path=str(path), config_file=str(self._config), weights_path=str(self._weights),\n                model=self._model, device="cuda", output_dir=str(self.cache_dir / "outputs"),\n                z_ratio=self.scale_um[0] / self.scale_um[1], lower_percentile=1.0,\n                upper_percentile=99.0, data_loader_num_workers=0, cell_radius=15.0,\n                background_threshold=float(np.median(frame_loader(int(t)))), stride=[32, 96, 96],\n                batch_size=12, score_thresh=0.6, mask_thresh=0.5, min_edge_area=64,\n                topk_postprocess=300, save_intermediate=False,\n            )\n            arr = np.asarray(result["instance_map"], dtype=np.int32)\n            if arr.ndim != 3:\n                raise ValueError(f"instance_map shape={arr.shape}")\n            self._maps[key] = arr\n            self.calls += 1\n            return arr\n        except Exception as exc:  # pragma: no cover - depends on Kaggle runtime\n            self.error = f"FOCUS3D inference failed at {dataset}/t{t}: {type(exc).__name__}: {exc}"\n            return None\n\n\ndef build_mask_frame_matches(\n    nodes_by_id: dict[int, dict[str, object]],\n    frames: dict[int, np.ndarray],\n    scale_um: tuple[float, float, float],\n    radius_um: float = 7.0,\n) -> dict[int, dict[int, MaskInstance]]:\n    """Match only nodes in each requested frame to that frame\'s instances."""\n    out: dict[int, dict[int, MaskInstance]] = {}\n    for t, arr in frames.items():\n        frame_nodes = {i: n for i, n in nodes_by_id.items() if int(n["t"]) == int(t)}\n        out[int(t)] = match_nodes_to_instances(frame_nodes, arr, scale_um, radius_um)\n    return out\n\n\n@dataclass(frozen=True)\nclass MaskInstance:\n    label: int\n    coords: np.ndarray  # N x 3 integer voxel coordinates (z, y, x)\n    centroid: tuple[float, float, float]\n    volume: int\n\n\ndef extract_instances(instance_map: np.ndarray) -> dict[int, MaskInstance]:\n    """Extract non-background instances while retaining their full voxels."""\n    arr = np.asarray(instance_map)\n    if arr.ndim != 3:\n        raise ValueError(f"instance_map must be 3-D, got {arr.shape}")\n    labels = np.unique(arr)\n    out: dict[int, MaskInstance] = {}\n    for raw_label in labels:\n        label = int(raw_label)\n        if label <= 0:\n            continue\n        coords = np.argwhere(arr == raw_label).astype(np.int32, copy=False)\n        if coords.size == 0:\n            continue\n        c = tuple(float(v) for v in coords.mean(axis=0))\n        out[label] = MaskInstance(label, coords, c, int(coords.shape[0]))\n    return out\n\n\ndef _pdist_um(a: Iterable[float], b: Iterable[float], scale_um: tuple[float, float, float]) -> float:\n    d = np.asarray(tuple(a), dtype=np.float64) - np.asarray(tuple(b), dtype=np.float64)\n    s = np.asarray(scale_um, dtype=np.float64)\n    return float(np.sqrt(np.sum((d * s) ** 2)))\n\n\ndef match_nodes_to_instances(\n    nodes: dict[int, dict[str, object]],\n    instance_map: np.ndarray,\n    scale_um: tuple[float, float, float],\n    radius_um: float = 7.0,\n) -> dict[int, MaskInstance]:\n    """One-to-one node/instance attachment with a physical-distance gate.\n\n    Ambiguous or out-of-radius matches are omitted.  The caller can therefore\n    fall back to the base graph when FOCUS3D is unavailable or uncertain.\n    """\n    instances = extract_instances(instance_map)\n    if not nodes or not instances:\n        return {}\n    node_ids = sorted(nodes)\n    labels = sorted(instances)\n    node_points = [\n        (float(nodes[n]["z"]), float(nodes[n]["y"]), float(nodes[n]["x"]))\n        for n in node_ids\n    ]\n    inst_points = [instances[k].centroid for k in labels]\n    cost = np.asarray(\n        [[_pdist_um(a, b, scale_um) for b in inst_points] for a in node_points],\n        dtype=np.float64,\n    )\n    rows, cols = linear_sum_assignment(cost)\n    result: dict[int, MaskInstance] = {}\n    for r, c in zip(rows, cols):\n        if float(cost[r, c]) <= float(radius_um):\n            result[node_ids[int(r)]] = instances[labels[int(c)]]\n    return result\n\n\ndef _shifted_set(mask: MaskInstance, shift: tuple[int, int, int]) -> set[tuple[int, int, int]]:\n    d = np.asarray(shift, dtype=np.int64)\n    return {tuple((row + d).tolist()) for row in mask.coords}\n\n\ndef mask_iou_after_shift(\n    source: MaskInstance,\n    target: MaskInstance,\n    shift: tuple[int, int, int],\n) -> tuple[float, float, float]:\n    """Return IoU and overlap fractions after translating source to target time."""\n    src = _shifted_set(source, shift)\n    dst = {tuple(row.tolist()) for row in target.coords}\n    inter = len(src & dst)\n    union = len(src | dst)\n    if union == 0:\n        return 0.0, 0.0, 0.0\n    return float(inter / union), float(inter / max(len(src), 1)), float(inter / max(len(dst), 1))\n\n\ndef _voxel_shift(source_node: dict[str, object], target_node: dict[str, object]) -> tuple[int, int, int]:\n    return tuple(\n        int(round(float(target_node[k]) - float(source_node[k]))) for k in ("z", "y", "x")\n    )\n\n\ndef continuation_mask_score(\n    source_node: dict[str, object],\n    target_node: dict[str, object],\n    source_mask: MaskInstance,\n    target_mask: MaskInstance,\n) -> float:\n    """Contour agreement for an ordinary one-frame continuation."""\n    iou, src_cov, dst_cov = mask_iou_after_shift(\n        source_mask, target_mask, _voxel_shift(source_node, target_node)\n    )\n    # IoU is the primary contour signal; coverage terms make tiny fragments\n    # unable to win solely because their union is small.\n    return float(0.60 * iou + 0.20 * src_cov + 0.20 * dst_cov)\n\n\ndef division_mask_score(\n    parent_node: dict[str, object],\n    daughter1_node: dict[str, object],\n    daughter2_node: dict[str, object],\n    parent_mask: MaskInstance,\n    daughter1_mask: MaskInstance,\n    daughter2_mask: MaskInstance,\n) -> tuple[float, dict[str, float]]:\n    """Score whether two daughter contours explain one parent contour.\n\n    The parent is translated separately along each observed daughter motion,\n    then the union coverage and daughter volume balance are measured from the\n    actual 3-D voxel sets.  No centroid-only feature is used in this score.\n    """\n    d1_shift = _voxel_shift(parent_node, daughter1_node)\n    d2_shift = _voxel_shift(parent_node, daughter2_node)\n    p1 = _shifted_set(parent_mask, d1_shift)\n    p2 = _shifted_set(parent_mask, d2_shift)\n    d1 = {tuple(row.tolist()) for row in daughter1_mask.coords}\n    d2 = {tuple(row.tolist()) for row in daughter2_mask.coords}\n    daughters = d1 | d2\n    if not daughters:\n        return 0.0, {"union_parent_coverage": 0.0, "volume_balance": 0.0, "daughter_iou": 0.0}\n    parent_union = p1 | p2\n    covered = len(parent_union & daughters) / max(len(parent_union), 1)\n    balance = min(daughter1_mask.volume, daughter2_mask.volume) / max(\n        daughter1_mask.volume, daughter2_mask.volume, 1\n    )\n    daughter_iou = len(d1 & d2) / max(len(d1 | d2), 1)\n    # Distinct instance IDs normally make daughter_iou zero.  It is retained as\n    # a sanity penalty for malformed/overlapping masks, not as positive evidence.\n    score = 0.65 * covered + 0.35 * balance - 0.25 * daughter_iou\n    return float(max(0.0, min(1.0, score))), {\n        "union_parent_coverage": float(covered),\n        "volume_balance": float(balance),\n        "daughter_iou": float(daughter_iou),\n    }\n\n\ndef local_mask_rewire(\n    nodes_by_id: dict[int, dict[str, object]],\n    edges: list[dict[str, object]],\n    masks_by_frame: dict[int, dict[int, MaskInstance]],\n    *,\n    max_parent_um: float = 14.0,\n    min_division_score: float = 0.20,\n    min_margin: float = 0.10,\n    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,\n) -> tuple[list[dict[str, object]], dict[str, int]]:\n    """Conservatively rewire occupied daughter conflicts using mask evidence.\n\n    For each P->D1 and Q->D conflict, compare H0 (Q->D) with H1 (P->D).\n    Exactly one old edge is removed and one new edge is added only when H1 has\n    a clear contour margin.  Existing node coordinates and all unrelated edges\n    are untouched.\n    """\n    if edge_distance_um is None:\n        edge_distance_um = lambda a, b: math.dist(\n            [float(a[k]) for k in ("z", "y", "x")], [float(b[k]) for k in ("z", "y", "x")]\n        )\n    out = [dict(e) for e in edges]\n    stats = {"focus_checked": 0, "focus_accepted": 0, "focus_rejected_margin": 0, "focus_missing_mask": 0}\n    changed: set[tuple[int, int]] = set()\n    active_pairs = {(int(e["source_id"]), int(e["target_id"])) for e in out}\n    by_source: dict[int, list[dict[str, object]]] = {}\n    by_target: dict[int, list[dict[str, object]]] = {}\n    for e in out:\n        s, t = int(e["source_id"]), int(e["target_id"])\n        by_source.setdefault(s, []).append(e)\n        by_target.setdefault(t, []).append(e)\n    for p, p_edges in list(by_source.items()):\n        if len(p_edges) != 1:\n            continue\n        e1 = p_edges[0]\n        d1 = int(e1["target_id"])\n        if (p, d1) not in active_pairs:\n            continue\n        if p not in nodes_by_id or d1 not in nodes_by_id:\n            continue\n        pn, d1n = nodes_by_id[p], nodes_by_id[d1]\n        if int(d1n["t"]) != int(pn["t"]) + 1:\n            continue\n        frame_p, frame_d = int(pn["t"]), int(d1n["t"])\n        pm = masks_by_frame.get(frame_p, {}).get(p)\n        m1 = masks_by_frame.get(frame_d, {}).get(d1)\n        if pm is None or m1 is None:\n            continue\n        for d, incoming in list(by_target.items()):\n            if d == d1 or len(incoming) != 1 or d not in nodes_by_id:\n                continue\n            q = int(incoming[0]["source_id"])\n            if q == p or q not in nodes_by_id:\n                continue\n            dn, qn = nodes_by_id[d], nodes_by_id[q]\n            if int(dn["t"]) != frame_d or int(qn["t"]) != frame_p:\n                continue\n            if edge_distance_um(pn, dn) > max_parent_um:\n                continue\n            md = masks_by_frame.get(frame_d, {}).get(d)\n            qm = masks_by_frame.get(frame_p, {}).get(q)\n            if md is None or qm is None:\n                stats["focus_missing_mask"] += 1\n                continue\n            stats["focus_checked"] += 1\n            h0 = continuation_mask_score(qn, dn, qm, md)\n            h1, _ = division_mask_score(pn, d1n, dn, pm, m1, md)\n            if h1 < min_division_score or h1 <= h0 + min_margin:\n                stats["focus_rejected_margin"] += 1\n                continue\n            key = (q, d)\n            if key in changed:\n                continue\n            out = [e for e in out if not (int(e["source_id"]) == q and int(e["target_id"]) == d)]\n            active_pairs.discard((q, d))\n            out.append({\n                "source_id": p,\n                "target_id": d,\n                "edge_prob": None,\n                "distance_um": float(edge_distance_um(pn, dn)),\n                "focus_mask_division": 1,\n                "focus_h0": float(h0),\n                "focus_h1": float(h1),\n            })\n            active_pairs.add((p, d))\n            changed.add(key)\n            stats["focus_accepted"] += 1\n            # A parent may receive at most one local replacement in this pass.\n            break\n    return out, stats\n\n\ndef apply_focus_mask_arbitration(\n    nodes_by_id: dict[int, dict[str, object]],\n    edges: list[dict[str, object]],\n    *,\n    dataset: str,\n    provider: FocusMaskProvider | None,\n    frame_loader: Callable[[int], np.ndarray],\n    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),\n    radius_um: float = 7.0,\n    max_parent_um: float = 14.0,\n    min_division_score: float = 0.20,\n    min_margin: float = 0.10,\n    max_conflicts: int = 4,\n    max_frames_per_dataset: int = 2,\n    stats: dict[str, int] | None = None,\n    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,\n) -> list[dict[str, object]]:\n    """Load only frames touched by an occupied-daughter conflict and rewire."""\n    if provider is None or not provider.available:\n        return edges\n    by_source: dict[int, list[dict[str, object]]] = {}\n    by_target: dict[int, list[dict[str, object]]] = {}\n    for e in edges:\n        by_source.setdefault(int(e["source_id"]), []).append(e)\n        by_target.setdefault(int(e["target_id"]), []).append(e)\n    ranked_conflicts: list[tuple[float, int, int, int, int]] = []\n    for p, es in by_source.items():\n        if len(es) != 1 or p not in nodes_by_id:\n            continue\n        d1 = int(es[0]["target_id"])\n        if d1 not in nodes_by_id:\n            continue\n        pn, d1n = nodes_by_id[p], nodes_by_id[d1]\n        if int(d1n["t"]) != int(pn["t"]) + 1:\n            continue\n        for d, inc in by_target.items():\n            if d == d1 or len(inc) != 1 or d not in nodes_by_id:\n                continue\n            q = int(inc[0]["source_id"])\n            if q == p or q not in nodes_by_id:\n                continue\n            if int(nodes_by_id[d]["t"]) != int(d1n["t"]):\n                continue\n            if edge_distance_um is None:\n                dist = math.dist(\n                    [float(pn[k]) for k in ("z", "y", "x")],\n                    [float(nodes_by_id[d][k]) for k in ("z", "y", "x")],\n                )\n            else:\n                dist = edge_distance_um(pn, nodes_by_id[d])\n            if dist <= max_parent_um:\n                sister_dist = edge_distance_um(d1n, nodes_by_id[d]) if edge_distance_um else math.dist(\n                    [float(d1n[k]) for k in ("z", "y", "x")],\n                    [float(nodes_by_id[d][k]) for k in ("z", "y", "x")],\n                )\n                # Keep only compact, division-like conflicts.  Without this\n                # gate every ordinary occupied target would trigger FOCUS3D.\n                if sister_dist <= max_parent_um:\n                    ranked_conflicts.append((float(dist + 0.35 * sister_dist), p, d1, q, d))\n    ranked_conflicts.sort(key=lambda x: x[0])\n    selected = ranked_conflicts[: max(0, int(max_conflicts))]\n    requested = set()\n    for _, p, d1, q, d in selected:\n        requested.update((int(nodes_by_id[p]["t"]), int(nodes_by_id[d]["t"])))\n    if max_frames_per_dataset > 0 and len(requested) > max_frames_per_dataset:\n        keep_frames = {int(nodes_by_id[p]["t"]) for _, p, _, _, _ in selected[:max_frames_per_dataset]}\n        requested = keep_frames | {int(nodes_by_id[p]["t"]) + 1 for _, p, _, _, _ in selected[:max_frames_per_dataset]}\n    if not selected or not requested:\n        return edges\n    frame_maps: dict[int, np.ndarray] = {}\n    for t in sorted(requested):\n        arr = provider.get(dataset, t, frame_loader)\n        if arr is not None:\n            frame_maps[t] = arr\n    if len(frame_maps) < 2:\n        if stats is not None:\n            stats["focus_runtime_unavailable"] = stats.get("focus_runtime_unavailable", 0) + 1\n        return edges\n    matches = build_mask_frame_matches(nodes_by_id, frame_maps, scale_um, radius_um)\n    masks_by_frame = {t: m for t, m in matches.items()}\n    out, local_stats = local_mask_rewire(\n        nodes_by_id, edges, masks_by_frame, max_parent_um=max_parent_um,\n        min_division_score=min_division_score, min_margin=min_margin,\n        edge_distance_um=edge_distance_um,\n    )\n    if stats is not None:\n        for k, v in local_stats.items():\n            stats[k] = stats.get(k, 0) + int(v)\n        stats["focus_frames_loaded"] = stats.get("focus_frames_loaded", 0) + len(frame_maps)\n        stats["focus_nodes_matched"] = stats.get("focus_nodes_matched", 0) + sum(\n            len(m) for m in masks_by_frame.values()\n        )\n    return out\n'
_focus_base_path = WORKING_DIR / "focus_mask_arbitration.py"
_focus_base_path.write_text(_focus_base_source, encoding="utf-8")
_focus_strategy_source = '"""FOCUS3D mask-use strategies for the EXP019 mechanism comparison."""\nfrom __future__ import annotations\n\nimport math\nfrom typing import Callable\n\nimport numpy as np\n\nfrom focus_mask_arbitration import (\n    FocusMaskProvider,\n    MaskInstance,\n    apply_focus_mask_arbitration,\n    build_mask_frame_matches,\n    continuation_mask_score,\n    division_mask_score,\n)\n\n\nConflict = tuple[float, int, int, int, int]  # rank, parent, daughter1, old_parent, daughter2\n\n\ndef _distance(\n    a: dict[str, object],\n    b: dict[str, object],\n    fn: Callable[[dict[str, object], dict[str, object]], float] | None,\n) -> float:\n    if fn is not None:\n        return float(fn(a, b))\n    return float(math.dist([float(a[k]) for k in ("z", "y", "x")], [float(b[k]) for k in ("z", "y", "x")]))\n\n\ndef rank_occupied_daughter_conflicts(\n    nodes_by_id: dict[int, dict[str, object]],\n    edges: list[dict[str, object]],\n    *,\n    max_parent_um: float = 14.0,\n    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,\n) -> list[Conflict]:\n    """Rank compact P->D1, Q->D conflicts without consulting labels."""\n    by_source: dict[int, list[dict[str, object]]] = {}\n    by_target: dict[int, list[dict[str, object]]] = {}\n    for edge in edges:\n        by_source.setdefault(int(edge["source_id"]), []).append(edge)\n        by_target.setdefault(int(edge["target_id"]), []).append(edge)\n\n    ranked: list[Conflict] = []\n    for parent, outgoing in by_source.items():\n        if len(outgoing) != 1 or parent not in nodes_by_id:\n            continue\n        daughter1 = int(outgoing[0]["target_id"])\n        if daughter1 not in nodes_by_id:\n            continue\n        pn, d1n = nodes_by_id[parent], nodes_by_id[daughter1]\n        if int(d1n["t"]) != int(pn["t"]) + 1:\n            continue\n        for daughter2, incoming in by_target.items():\n            if daughter2 == daughter1 or len(incoming) != 1 or daughter2 not in nodes_by_id:\n                continue\n            old_parent = int(incoming[0]["source_id"])\n            if old_parent == parent or old_parent not in nodes_by_id:\n                continue\n            d2n, qn = nodes_by_id[daughter2], nodes_by_id[old_parent]\n            if int(d2n["t"]) != int(d1n["t"]) or int(qn["t"]) != int(pn["t"]):\n                continue\n            parent_distance = _distance(pn, d2n, edge_distance_um)\n            sister_distance = _distance(d1n, d2n, edge_distance_um)\n            if parent_distance <= max_parent_um and sister_distance <= max_parent_um:\n                ranked.append((parent_distance + 0.35 * sister_distance, parent, daughter1, old_parent, daughter2))\n    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))\n    return ranked\n\n\ndef _select_with_frame_budget(\n    ranked: list[Conflict],\n    nodes_by_id: dict[int, dict[str, object]],\n    *,\n    max_items: int,\n    max_frames: int,\n) -> tuple[list[Conflict], set[int]]:\n    selected: list[Conflict] = []\n    frames: set[int] = set()\n    for item in ranked:\n        if len(selected) >= max(0, int(max_items)):\n            break\n        _, parent, _, _, daughter2 = item\n        needed = {int(nodes_by_id[parent]["t"]), int(nodes_by_id[daughter2]["t"])}\n        if max_frames > 0 and len(frames | needed) > int(max_frames):\n            continue\n        selected.append(item)\n        frames.update(needed)\n    return selected, frames\n\n\ndef _load_matches(\n    nodes_by_id: dict[int, dict[str, object]],\n    *,\n    dataset: str,\n    frames: set[int],\n    provider: FocusMaskProvider,\n    frame_loader: Callable[[int], np.ndarray],\n    scale_um: tuple[float, float, float],\n    radius_um: float,\n) -> tuple[dict[int, dict[int, MaskInstance]], int]:\n    maps: dict[int, np.ndarray] = {}\n    for frame in sorted(frames):\n        instance_map = provider.get(dataset, frame, frame_loader)\n        if instance_map is not None:\n            maps[frame] = instance_map\n    if not maps:\n        return {}, 0\n    return build_mask_frame_matches(nodes_by_id, maps, scale_um, radius_um), len(maps)\n\n\ndef apply_selected_rescue(\n    nodes_by_id: dict[int, dict[str, object]],\n    edges: list[dict[str, object]],\n    *,\n    dataset: str,\n    provider: FocusMaskProvider | None,\n    frame_loader: Callable[[int], np.ndarray],\n    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),\n    radius_um: float = 7.0,\n    max_parent_um: float = 14.0,\n    min_division_score: float = 0.20,\n    min_margin: float = 0.10,\n    max_conflicts: int = 4,\n    max_frames: int = 8,\n    stats: dict[str, int] | None = None,\n    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,\n) -> list[dict[str, object]]:\n    """Evaluate and modify only the explicitly selected conflict tuples."""\n    if provider is None or not provider.available:\n        return edges\n    ranked = rank_occupied_daughter_conflicts(\n        nodes_by_id, edges, max_parent_um=max_parent_um, edge_distance_um=edge_distance_um\n    )\n    selected, frames = _select_with_frame_budget(\n        ranked, nodes_by_id, max_items=max_conflicts, max_frames=max_frames\n    )\n    if stats is not None:\n        stats["focus_ranked_conflicts"] = stats.get("focus_ranked_conflicts", 0) + len(ranked)\n        stats["focus_selected_conflicts"] = stats.get("focus_selected_conflicts", 0) + len(selected)\n    if not selected:\n        return edges\n    matches, loaded = _load_matches(\n        nodes_by_id, dataset=dataset, frames=frames, provider=provider,\n        frame_loader=frame_loader, scale_um=scale_um, radius_um=radius_um,\n    )\n    if stats is not None:\n        stats["focus_frames_loaded"] = stats.get("focus_frames_loaded", 0) + loaded\n        stats["focus_nodes_matched"] = stats.get("focus_nodes_matched", 0) + sum(len(v) for v in matches.values())\n\n    out = [dict(edge) for edge in edges]\n    active = {(int(edge["source_id"]), int(edge["target_id"])) for edge in out}\n    used_parents: set[int] = set()\n    for _, parent, daughter1, old_parent, daughter2 in selected:\n        if stats is not None:\n            stats["focus_checked"] = stats.get("focus_checked", 0) + 1\n        if parent in used_parents or (parent, daughter1) not in active or (old_parent, daughter2) not in active:\n            continue\n        pn, d1n = nodes_by_id[parent], nodes_by_id[daughter1]\n        qn, d2n = nodes_by_id[old_parent], nodes_by_id[daughter2]\n        pm = matches.get(int(pn["t"]), {}).get(parent)\n        d1m = matches.get(int(d1n["t"]), {}).get(daughter1)\n        qm = matches.get(int(qn["t"]), {}).get(old_parent)\n        d2m = matches.get(int(d2n["t"]), {}).get(daughter2)\n        if any(mask is None for mask in (pm, d1m, qm, d2m)):\n            if stats is not None:\n                stats["focus_missing_mask"] = stats.get("focus_missing_mask", 0) + 1\n            continue\n        h0 = continuation_mask_score(qn, d2n, qm, d2m)\n        h1, details = division_mask_score(pn, d1n, d2n, pm, d1m, d2m)\n        if h1 < min_division_score or h1 <= h0 + min_margin:\n            if stats is not None:\n                stats["focus_rejected_margin"] = stats.get("focus_rejected_margin", 0) + 1\n            continue\n        out = [\n            edge for edge in out\n            if not (int(edge["source_id"]) == old_parent and int(edge["target_id"]) == daughter2)\n        ]\n        out.append({\n            "source_id": parent,\n            "target_id": daughter2,\n            "edge_prob": None,\n            "distance_um": _distance(pn, d2n, edge_distance_um),\n            "focus_mask_division": 1,\n            "focus_h0": float(h0),\n            "focus_h1": float(h1),\n            "focus_volume_balance": float(details["volume_balance"]),\n        })\n        active.remove((old_parent, daughter2))\n        active.add((parent, daughter2))\n        used_parents.add(parent)\n        if stats is not None:\n            stats["focus_accepted"] = stats.get("focus_accepted", 0) + 1\n    return out\n\n\ndef apply_safe_division_veto(\n    nodes_by_id: dict[int, dict[str, object]],\n    edges: list[dict[str, object]],\n    *,\n    dataset: str,\n    provider: FocusMaskProvider | None,\n    frame_loader: Callable[[int], np.ndarray],\n    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),\n    radius_um: float = 7.0,\n    min_score: float = 0.18,\n    min_volume_balance: float = 0.25,\n    max_divisions: int = 8,\n    max_frames: int = 8,\n    stats: dict[str, int] | None = None,\n    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,\n) -> list[dict[str, object]]:\n    """Remove only a safe-div-added edge when two mask tests strongly disagree."""\n    if provider is None or not provider.available:\n        return edges\n    by_source: dict[int, list[dict[str, object]]] = {}\n    for edge in edges:\n        by_source.setdefault(int(edge["source_id"]), []).append(edge)\n    candidates: list[tuple[float, int, dict[str, object], dict[str, object]]] = []\n    for parent, outgoing in by_source.items():\n        if len(outgoing) != 2 or parent not in nodes_by_id:\n            continue\n        added = [edge for edge in outgoing if int(edge.get("safe_division", 0)) == 1]\n        existing = [edge for edge in outgoing if int(edge.get("safe_division", 0)) != 1]\n        if len(added) != 1 or len(existing) != 1:\n            continue\n        d1, d2 = int(existing[0]["target_id"]), int(added[0]["target_id"])\n        if d1 not in nodes_by_id or d2 not in nodes_by_id:\n            continue\n        pn, d1n, d2n = nodes_by_id[parent], nodes_by_id[d1], nodes_by_id[d2]\n        a = _distance(pn, d1n, edge_distance_um)\n        b = _distance(pn, d2n, edge_distance_um)\n        asymmetry = abs(a - b) / max((a + b) / 2.0, 1e-6)\n        candidates.append((-asymmetry, parent, existing[0], added[0]))\n    candidates.sort(key=lambda item: (item[0], item[1]))\n\n    selected: list[tuple[float, int, dict[str, object], dict[str, object]]] = []\n    frames: set[int] = set()\n    for item in candidates:\n        if len(selected) >= max(0, int(max_divisions)):\n            break\n        _, parent, _, added = item\n        daughter2 = int(added["target_id"])\n        needed = {int(nodes_by_id[parent]["t"]), int(nodes_by_id[daughter2]["t"])}\n        if max_frames > 0 and len(frames | needed) > int(max_frames):\n            continue\n        selected.append(item)\n        frames.update(needed)\n    if stats is not None:\n        stats["focus_veto_candidates"] = stats.get("focus_veto_candidates", 0) + len(candidates)\n        stats["focus_veto_selected"] = stats.get("focus_veto_selected", 0) + len(selected)\n    if not selected:\n        return edges\n    matches, loaded = _load_matches(\n        nodes_by_id, dataset=dataset, frames=frames, provider=provider,\n        frame_loader=frame_loader, scale_um=scale_um, radius_um=radius_um,\n    )\n    if stats is not None:\n        stats["focus_frames_loaded"] = stats.get("focus_frames_loaded", 0) + loaded\n        stats["focus_nodes_matched"] = stats.get("focus_nodes_matched", 0) + sum(len(v) for v in matches.values())\n\n    remove_pairs: set[tuple[int, int]] = set()\n    for _, parent, existing, added in selected:\n        daughter1, daughter2 = int(existing["target_id"]), int(added["target_id"])\n        pn, d1n, d2n = nodes_by_id[parent], nodes_by_id[daughter1], nodes_by_id[daughter2]\n        pm = matches.get(int(pn["t"]), {}).get(parent)\n        d1m = matches.get(int(d1n["t"]), {}).get(daughter1)\n        d2m = matches.get(int(d2n["t"]), {}).get(daughter2)\n        if stats is not None:\n            stats["focus_veto_checked"] = stats.get("focus_veto_checked", 0) + 1\n        if any(mask is None for mask in (pm, d1m, d2m)):\n            if stats is not None:\n                stats["focus_veto_missing_mask"] = stats.get("focus_veto_missing_mask", 0) + 1\n            continue\n        score, details = division_mask_score(pn, d1n, d2n, pm, d1m, d2m)\n        # Conservative contradiction: both the combined shape score and the\n        # daughter-volume balance must fail. Missing masks never cause deletion.\n        if score < min_score and float(details["volume_balance"]) < min_volume_balance:\n            remove_pairs.add((parent, daughter2))\n            if stats is not None:\n                stats["focus_veto_removed"] = stats.get("focus_veto_removed", 0) + 1\n    return [\n        edge for edge in edges\n        if (int(edge["source_id"]), int(edge["target_id"])) not in remove_pairs\n    ]\n\n\ndef apply_focus_strategy(\n    nodes_by_id: dict[int, dict[str, object]],\n    edges: list[dict[str, object]],\n    *,\n    mode: str,\n    dataset: str,\n    provider: FocusMaskProvider | None,\n    frame_loader: Callable[[int], np.ndarray],\n    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),\n    radius_um: float = 7.0,\n    max_parent_um: float = 14.0,\n    min_division_score: float = 0.20,\n    min_margin: float = 0.10,\n    max_conflicts: int = 4,\n    max_frames: int = 8,\n    stats: dict[str, int] | None = None,\n    edge_distance_um: Callable[[dict[str, object], dict[str, object]], float] | None = None,\n) -> list[dict[str, object]]:\n    """Dispatch one pre-registered mask-use strategy."""\n    mode = str(mode).strip().lower()\n    if mode == "off":\n        return edges\n    if mode == "broad_rescue":\n        return apply_focus_mask_arbitration(\n            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,\n            scale_um=scale_um, radius_um=radius_um, max_parent_um=max_parent_um,\n            min_division_score=min_division_score, min_margin=min_margin,\n            # Preserve the scored EXP018-v3 behaviour: its value 2 means two\n            # conflict frame-pairs and resulted in at most four loaded frames.\n            max_conflicts=max_conflicts, max_frames_per_dataset=2,\n            stats=stats, edge_distance_um=edge_distance_um,\n        )\n    if mode == "selected_rescue":\n        return apply_selected_rescue(\n            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,\n            scale_um=scale_um, radius_um=radius_um, max_parent_um=max_parent_um,\n            min_division_score=min_division_score, min_margin=min_margin,\n            max_conflicts=max_conflicts, max_frames=max_frames,\n            stats=stats, edge_distance_um=edge_distance_um,\n        )\n    if mode == "veto_only":\n        return apply_safe_division_veto(\n            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,\n            scale_um=scale_um, radius_um=radius_um, max_frames=max_frames,\n            stats=stats, edge_distance_um=edge_distance_um,\n        )\n    if mode == "hybrid_selected":\n        rescued = apply_selected_rescue(\n            nodes_by_id, edges, dataset=dataset, provider=provider, frame_loader=frame_loader,\n            scale_um=scale_um, radius_um=radius_um, max_parent_um=max_parent_um,\n            min_division_score=min_division_score, min_margin=min_margin,\n            max_conflicts=max_conflicts, max_frames=max_frames,\n            stats=stats, edge_distance_um=edge_distance_um,\n        )\n        return apply_safe_division_veto(\n            nodes_by_id, rescued, dataset=dataset, provider=provider, frame_loader=frame_loader,\n            scale_um=scale_um, radius_um=radius_um, max_frames=max_frames,\n            stats=stats, edge_distance_um=edge_distance_um,\n        )\n    raise ValueError(f"unknown FOCUS strategy: {mode}")\n'
_focus_strategy_path = WORKING_DIR / "focus_mask_strategies.py"
_focus_strategy_path.write_text(_focus_strategy_source, encoding="utf-8")
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))
from focus_mask_arbitration import FocusMaskProvider
from focus_mask_strategies import apply_focus_strategy

FOCUS_STRATEGY_MODE = "off"
FOCUS_MATCH_RADIUS_UM = 7.0
FOCUS_PARENT_MAX_UM = 14.0
FOCUS_MIN_DIV_SCORE = 0.20
FOCUS_MIN_MARGIN = 0.10
FOCUS_MAX_CONFLICTS = 4
FOCUS_MAX_FRAMES = 4
FOCUS_PROVIDER = FocusMaskProvider(
    WORKING_DIR / "focus3d_mask_cache",
    scale_um=(1.625, 0.40625, 0.40625),
    max_frames=80,
)
print("EXP019 FOCUS strategies ready; production test pass remains FOCUS-off")
import tracksdata as td
import numpy as np
import blosc2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1



def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            return frame
    except Exception:
        pass
    import zarr
    frame = np.asarray(zarr.open(zarr_path / "0", mode="r")[t])
    frame_cache[t] = frame
    return frame


def refine_synthetic_midpoint(
    dataset: str | None,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    if not GAP_REFINE_SYNTHETIC or dataset is None:
        return midpoint
    try:
        frame = read_test_frame(dataset, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - GAP_REFINE_WIN_Z)
        z1 = min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0 = max(0, y - GAP_REFINE_WIN_YX)
        y1 = min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0 = max(0, x - GAP_REFINE_WIN_YX)
        x1 = min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint



def _dc_pool_frame_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(
        ratio,
        float(getattr(cfg, "norm_clip_lo", -0.5)),
        float(getattr(cfg, "norm_clip_hi", 6.0)),
    ).astype(np.float32)


def _dc_manifest_weight_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        print("Could not read DeepCenter manifest:", manifest_path, type(exc).__name__, exc)
        return []
    root = manifest_path.parent
    sections: list[dict[str, object]] = []
    for section in [
        manifest.get("model", {}),
        manifest.get("models", {}).get("full_frame_center", {}) if isinstance(manifest.get("models", {}), dict) else {},
        manifest.get("full_frame_center", {}),
    ]:
        if isinstance(section, dict):
            sections.append(section)
    candidates: list[Path] = []
    for section in sections:
        for key in ("weight_path", "path"):
            rel = section.get(key)
            if isinstance(rel, str) and rel:
                candidates.append(root / rel)
        for key in ("last_checkpoint", "best_checkpoint"):
            item = section.get(key)
            if isinstance(item, dict):
                rel = item.get("path")
                if isinstance(rel, str) and rel:
                    candidates.append(root / rel)
    for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
        candidates.append(root / "weights" / "full_frame_center" / name)
        candidates.append(root / name)
    candidates.append(root / DEEPCENTER_RELATIVE)
    return candidates


def _dc_checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", DEEPCENTER_CHECKPOINT_DEFAULT).strip()
    if explicit:
        candidates.append(Path(explicit))
    manifest_explicit = os.environ.get("BIOHUB_DEEPCENTER_MANIFEST", DEEPCENTER_MANIFEST_DEFAULT).strip()
    if manifest_explicit:
        candidates.extend(_dc_manifest_weight_paths(Path(manifest_explicit)))

    input_root = Path("/kaggle/input")
    preferred_dirs = [
        Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),
        Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),
    ]
    for directory in preferred_dirs:
        candidates.extend(_dc_manifest_weight_paths(directory / "ARTIFACT_MANIFEST.json"))
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.append(directory / "weights" / "full_frame_center" / name)
            candidates.append(directory / name)
    if input_root.exists():
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.extend(sorted(input_root.glob(f"**/full_frame_center/**/{name}")))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        path = path.expanduser()
        try:
            key = path.resolve() if path.exists() else path
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


try:
    import torch
except Exception as _dc_torch_error:
    torch = None


if torch is not None:
    class _DCConvBlock3d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class _DCDeepCenterUNet3D(torch.nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = _DCConvBlock3d(in_channels, c)
            self.down1 = torch.nn.MaxPool3d(2, 2)
            self.enc2 = _DCConvBlock3d(c, c * 2)
            self.down2 = torch.nn.MaxPool3d(2, 2)
            self.enc3 = _DCConvBlock3d(c * 2, c * 4)
            self.down3 = torch.nn.MaxPool3d(2, 2)
            self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
            self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = _DCConvBlock3d(c * 8, c * 4)
            self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = _DCConvBlock3d(c * 4, c * 2)
            self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = _DCConvBlock3d(c * 2, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)
else:
    _DCConvBlock3d = None
    _DCDeepCenterUNet3D = None

def load_deepcenter_veto_detector() -> dict[str, object] | None:
    if not USE_DEEPCENTER_VETO:
        print("DeepCenter add-only repair gate disabled by configuration.")
        return None
    if torch is None:
        if REQUIRE_DEEPCENTER_VETO:
            raise ImportError("torch is required for DeepCenter add-only repair gate")
        print("DeepCenter add-only repair gate skipped because torch is unavailable.")
        return None
    from types import SimpleNamespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_errors: list[str] = []
    for checkpoint_path in _dc_checkpoint_candidates():
        if not checkpoint_path.exists():
            continue
        try:
            print("Trying DeepCenter add-only gate checkpoint:", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError("checkpoint has no model_state")
            checkpoint_epoch = int(checkpoint.get("epoch", -1))
            if DEEPCENTER_EXPECTED_EPOCH > 0 and checkpoint_epoch != DEEPCENTER_EXPECTED_EPOCH:
                raise ValueError(
                    f"expected DeepCenter epoch {DEEPCENTER_EXPECTED_EPOCH}, got {checkpoint_epoch}"
                )
            cfg = SimpleNamespace(**checkpoint.get("config", {}))
            model = _DCDeepCenterUNet3D(base_channels=int(getattr(cfg, "base_channels", 24)))
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()
            print("Loaded DeepCenter add-only gate checkpoint:", checkpoint_path)
            print("DeepCenter checkpoint epoch:", checkpoint.get("epoch"), "best_score:", checkpoint.get("best_score"))
            return {
                "model": model,
                "cfg": cfg,
                "device": device,
                "path": checkpoint_path,
                "torch": torch,
            }
        except Exception as exc:
            load_errors.append(f"{checkpoint_path}: {type(exc).__name__}: {exc}")
            print("Skipping incompatible DeepCenter checkpoint:", checkpoint_path, "|", type(exc).__name__, exc)
    message = "No usable DeepCenter checkpoint found for add-only repair gate."
    if REQUIRE_DEEPCENTER_VETO:
        checked = "\n".join(str(p) for p in _dc_checkpoint_candidates()[:80])
        errors = "\n".join(load_errors[-20:])
        raise FileNotFoundError(message + "\nChecked:\n" + checked + ("\nLoad errors:\n" + errors if errors else ""))
    print(message)
    return None


def _dc_cache_trim(cache: dict[tuple[str, int], np.ndarray]) -> None:
    limit = max(1, int(DEEPCENTER_SCORE_CACHE_MAX_FRAMES))
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def deepcenter_heatmap_for_frame(
    dataset: str,
    t: int,
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray | None:
    if detector_bundle is None:
        return None
    key = (dataset, int(t))
    cached = heatmap_cache.get(key)
    if cached is not None:
        return cached
    model = detector_bundle["model"]
    cfg = detector_bundle["cfg"]
    device = detector_bundle["device"]
    torch_mod = detector_bundle["torch"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    volume = read_test_frame(dataset, int(t), frame_cache)
    pooled = _dc_pool_frame_xy(volume, pool_factor)
    image = _dc_normalize_dynamic_range(pooled, cfg)
    with torch_mod.no_grad():
        tensor = torch_mod.from_numpy(image[None, None, ...]).to(device=device, dtype=torch_mod.float32)
        logits = model(tensor)
        # BIOHUB_DEEPCENTER_TTA: average the veto model's logits over the same
        # 8-view D4 group the detection and edge paths already use. This was the
        # last stage in the pipeline scored from a single un-augmented view, and
        # the veto it drives is the only gate whose threshold moves the score in
        # both directions (0.15 -> -0.002, 0.35 -> -0.005 around 0.25).
        if os.environ.get("BIOHUB_DEEPCENTER_TTA", "0") != "0":
            acc = logits.clone(); nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                acc = acc + model(tensor.flip(dims)).flip(dims); nv += 1
            if tensor.shape[-1] == tensor.shape[-2]:
                for k in (1, 3):
                    acc = acc + torch_mod.rot90(model(torch_mod.rot90(tensor, k, dims=(-2, -1))), -k, dims=(-2, -1)); nv += 1
                acc = acc + model(tensor.transpose(-1, -2)).transpose(-1, -2); nv += 1
                at = torch_mod.rot90(tensor, 1, dims=(-2, -1)).transpose(-1, -2)
                acc = acc + torch_mod.rot90(model(at).transpose(-1, -2), -1, dims=(-2, -1)); nv += 1
            delta = float((acc / nv - logits).abs().mean())
            if delta == 0.0:
                raise RuntimeError("DEEPCENTER_TTA_NO_OP: averaged veto logits identical to the single view")
            if not getattr(deepcenter_heatmap_for_frame, "_tta_announced", False):
                print("DEEPCENTER_TTA_ACTIVE views=", nv, "mean_abs_logit_delta=", round(delta, 6), flush=True)
                deepcenter_heatmap_for_frame._tta_announced = True
            logits = acc / nv
        heatmap = torch_mod.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    heatmap_cache[key] = heatmap
    _dc_cache_trim(heatmap_cache)
    return heatmap


def deepcenter_score_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> float | None:
    if not USE_DEEPCENTER_VETO or detector_bundle is None or dataset is None:
        return None
    heatmap = deepcenter_heatmap_for_frame(dataset, int(t), detector_bundle, frame_cache, heatmap_cache)
    if heatmap is None or heatmap.size == 0:
        return None
    cfg = detector_bundle["cfg"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / max(pool_factor, 1)))
    x = int(round(float(point[2]) / max(pool_factor, 1)))
    z0, z1 = max(0, z - DEEPCENTER_SCORE_WIN_Z), min(heatmap.shape[0], z + DEEPCENTER_SCORE_WIN_Z + 1)
    y0, y1 = max(0, y - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[1], y + DEEPCENTER_SCORE_WIN_YX + 1)
    x0, x1 = max(0, x - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[2], x + DEEPCENTER_SCORE_WIN_YX + 1)
    patch = heatmap[z0:z1, y0:y1, x0:x1]
    if patch.size == 0:
        return None
    score = float(np.max(patch))
    return score if np.isfinite(score) else None


def deepcenter_accept_repair_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
    stats: dict[str, int],
    prefix: str,
    threshold: float,
) -> bool:
    if not USE_DEEPCENTER_VETO:
        return True
    if detector_bundle is None or dataset is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    stats[f"deepcenter_{prefix}_checked"] += 1
    score = deepcenter_score_point(dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache)
    if score is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    if score < float(threshold):
        stats[f"deepcenter_{prefix}_rejected"] += 1
        return False
    stats[f"deepcenter_{prefix}_accepted"] += 1
    return True

def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


def motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_dist[r, c]),
                float(motion_dist[r, c]),
                float(prob_matrix[r, c]),
            ))
        return matches

    times = sorted(ids_by_t)
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for pass_name, gate_um in (("tight", MOTION_RELINK_TIGHT_UM), ("relaxed", MOTION_RELINK_RELAXED_UM)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
                if pass_name == "tight":
                    stats["motion_relink_tight_edges"] += 1
                else:
                    stats["motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges

def close_single_frame_gaps(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        GAP_CLOSE_MAX_ADDED_ABS,
        max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC))) if GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []

    density_cache: dict[int, dict[int, float]] = {}

    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)
        if cached is not None:
            return cached

        frame_ids = all_ids_by_t.get(t, [])
        if len(frame_ids) <= 1:
            result = {
                node_id: GAP_DENSITY_REFERENCE_UM
                for node_id in frame_ids
            }
            density_cache[t] = result
            return result

        positions = np.stack(
            [_position_um(nodes_by_id[node_id]) for node_id in frame_ids]
        )
        tree = cKDTree(positions)
        query_k = min(
            len(frame_ids),
            max(2, GAP_DENSITY_NEIGHBORS + 1),
        )
        distances, _ = tree.query(positions, k=query_k)
        if distances.ndim == 1:
            distances = distances[:, None]

        result: dict[int, float] = {}
        for idx, node_id in enumerate(frame_ids):
            neighbour_distances = distances[idx, 1:]
            neighbour_distances = neighbour_distances[
                np.isfinite(neighbour_distances)
            ]
            spacing = (
                float(np.median(neighbour_distances))
                if neighbour_distances.size
                else GAP_DENSITY_REFERENCE_UM
            )
            result[node_id] = spacing

        density_cache[t] = result
        stats["gap_density_nodes_scored"] += len(result)
        return result

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros(
                (len(end_ids), len(start_ids)),
                dtype=np.float64,
            )
            adaptive_threshold = np.full_like(d, threshold_um)

            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)

                    if GAP_DENSITY_ADAPTIVE:
                        local_spacing = 0.5 * (
                            source_spacing.get(
                                end_ids[i],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                            + target_spacing.get(
                                start_ids[j],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                        )
                        step_delta = float(
                            np.clip(
                                GAP_DENSITY_GAIN
                                * (
                                    local_spacing
                                    - GAP_DENSITY_REFERENCE_UM
                                ),
                                -GAP_DENSITY_MAX_STEP_DELTA_UM,
                                GAP_DENSITY_MAX_STEP_DELTA_UM,
                            )
                        )
                        adaptive_threshold[i, j] = (
                            threshold_um + step_delta * (gap + 1)
                        )
                        stats[
                            "gap_density_step_delta_milli_sum"
                        ] += int(round(1000.0 * step_delta))

            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold

            stats["gap_density_candidates_expanded"] += int(
                (adaptive_allowed & ~base_allowed).sum()
            )
            stats["gap_density_candidates_restricted"] += int(
                (base_allowed & ~adaptive_allowed).sum()
            )
            stats["gap_candidates"] += int(adaptive_allowed.sum())

            if not np.isfinite(d).any():
                continue

            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                if not base_allowed[r, c]:
                    stats[
                        "gap_density_selected_outside_base"
                    ] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id: int | None = None
                middle_reused = False
                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(dataset, mid_t, mid_point, frame_cache, stats)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": refined_point[0],
                        "y": refined_point[1],
                        "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                gap_span_um = float(d[r, c])
                marginal_gap = gap_span_um >= DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM
                synthetic_middle = int(middle.get("gap_synthetic", 0)) == 1
                requires_center_confirmation = (
                    DEEPCENTER_GAP_VETO and marginal_gap and synthetic_middle
                )
                if DEEPCENTER_GAP_VETO and not marginal_gap:
                    stats["deepcenter_gap_bypassed_strong_motion"] += 1
                elif DEEPCENTER_GAP_VETO and not synthetic_middle:
                    stats["deepcenter_gap_bypassed_observed_node"] += 1
                if requires_center_confirmation and not deepcenter_accept_repair_point(
                    dataset,
                    mid_t,
                    node_point(middle),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "gap",
                    DEEPCENTER_GAP_THRESHOLD,
                ):
                    if int(middle.get("gap_synthetic", 0)) == 1:
                        nodes_by_id.pop(middle_id, None)
                        synthetic_added = max(0, synthetic_added - 1)
                        stats["gap_inserted_synthetic"] = max(0, stats["gap_inserted_synthetic"] - 1)
                    continue
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def recover_strict_gap2(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, object]] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(dataset, mid_t, midpoint, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": refined_point[0],
                "y": refined_point[1],
                "x": refined_point[2],
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id,
                "target_id": node_id,
                "edge_prob": None,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], current),
                "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id,
            "target_id": start_id,
            "edge_prob": None,
            "distance_um": edge_distance_um(nodes_by_id[previous_id], target),
            "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
 
    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))
 
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
 
    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()  
 
    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]
        if not source_ids or not candidate_ids:
            continue
 
        
        
        
        
        
        candidate_tree = None
        if SAFE_DIV_REQUIRE_MUTUAL_NN:
            candidate_positions = np.stack([_position_um(nodes_by_id[cid]) for cid in candidate_ids])
            candidate_tree = cKDTree(candidate_positions)
 
        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
 
            
            
            
            
            mutual_nn_id = None
            if candidate_tree is not None:
                _, nn_idx = candidate_tree.query(_position_um(existing_child))
                mutual_nn_id = candidate_ids[int(nn_idx)]
 
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue
 
                
                if SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id:
                    stats["safe_division_mutual_nn_rejected"] += 1
                    continue
 
                
                
                
                
                if SAFE_DIV_REQUIRE_DIVERGENCE:
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])
                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats["safe_division_divergence_rejected"] += 1
                        continue
                    c1_grandchild = nodes_by_id.get(int(c1_succ[0]["target_id"]))
                    q_grandchild = nodes_by_id.get(int(q_succ[0]["target_id"]))
                    if (
                        c1_grandchild is None or q_grandchild is None
                        or int(c1_grandchild["t"]) != t + 2
                        or int(q_grandchild["t"]) != t + 2
                    ):
                        stats["safe_division_divergence_rejected"] += 1
                        continue
                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)
                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
                        stats["safe_division_divergence_rejected"] += 1
                        continue
 
                stats["safe_division_geometric_candidates"] += 1
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                    dataset,
                    int(candidate["t"]),
                    node_point(candidate),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "safe_div",
                    DEEPCENTER_SAFE_DIV_THRESHOLD,
                ):
                    continue
                
                
                
                
                
                
                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:
                    _sym_denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / _sym_denom > SAFE_DIV_SISTER_SYMMETRY_TAU:
                        stats["safe_division_symmetry_rejected"] += 1
                        continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))
 
        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            candidate = nodes_by_id[candidate_id]
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1
 
    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if ADAPTIVE_SHORT_TRACK_RESCUE:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC:
            budget = min(
                SHORT_TRACK_RESCUE_MAX_NODES_ABS,
                max(0, int(round(len(nodes_by_id) * SHORT_TRACK_RESCUE_MAX_NODES_FRAC))),
            )
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals: list[tuple[float, int, float, int, list[int]]] = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < SHORT_TRACK_RESCUE_MIN_LEN or len(members) >= OUTPUT_MIN_TRACK_LEN:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs: list[float] = []
                dists: list[float] = []
                for edge in c_edges:
                    try:
                        prob = float(edge.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(edge.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB:
                    continue
                if mean_dist > SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = 0
            rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


def filter_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    stats = {
        "raw_edges": len(raw_edges),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_density_nodes_scored": 0,
        "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": 0,
        "safe_division_geometric_candidates": 0,  
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "safe_division_mutual_nn_rejected": 0,
        "safe_division_divergence_rejected": 0,
        "safe_division_symmetry_rejected": 0,  
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_observed_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
    }

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge["distance_um"] = distance_um
        if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    print(f"  [{dataset}] after edge-filter+motion-relink: {len(nodes_by_id)} nodes, {len(edges)} edges")
    repair_frame_cache: dict[int, np.ndarray] = {}
    deepcenter_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    nodes_by_id, edges = close_single_frame_gaps(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)
    print(f"  [{dataset}] after gap-closing (single-frame + gap2): {len(nodes_by_id)} nodes, {len(edges)} edges")
    edges = add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )


    if FOCUS_STRATEGY_MODE != "off" and dataset is not None and edges:
        edges = apply_focus_strategy(
            nodes_by_id,
            edges,
            mode=FOCUS_STRATEGY_MODE,
            dataset=dataset,
            provider=FOCUS_PROVIDER,
            frame_loader=lambda _t: read_test_frame(dataset, int(_t), repair_frame_cache),
            scale_um=VOXEL_SCALE_UM,
            radius_um=FOCUS_MATCH_RADIUS_UM,
            max_parent_um=FOCUS_PARENT_MAX_UM,
            min_division_score=FOCUS_MIN_DIV_SCORE,
            min_margin=FOCUS_MIN_MARGIN,
            max_conflicts=FOCUS_MAX_CONFLICTS,
            max_frames=FOCUS_MAX_FRAMES,
            stats=stats,
            edge_distance_um=edge_distance_um,
        )
        print(
            f"  [{dataset}] FOCUS strategy={FOCUS_STRATEGY_MODE} "
            f"checked={stats.get('focus_checked', 0)} accepted={stats.get('focus_accepted', 0)} "
            f"veto_removed={stats.get('focus_veto_removed', 0)} "
            f"frames={stats.get('focus_frames_loaded', 0)}"
        )

    _geo_cands = stats['safe_division_geometric_candidates']
    _post_veto_cands = stats['safe_division_candidates']
    _rejected_by_dc = _geo_cands - _post_veto_cands
    print(
        f"  [{dataset}] after safe-division repair: {len(nodes_by_id)} nodes, {len(edges)} edges"
        f" (geometric_candidates={_geo_cands}, deepcenter_rejected={_rejected_by_dc},"
        f" post_veto_candidates={_post_veto_cands}, added={stats['safe_divisions_added']},"
        f" cap_skipped={stats['safe_division_skipped_cap']},"
        f" mutual_nn_rejected={stats['safe_division_mutual_nn_rejected']},"
        f" divergence_rejected={stats['safe_division_divergence_rejected']})"
    )
    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)

        filtered: list[dict[str, object]] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue

            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1["distance_um"])
            d2 = float(top2["distance_um"])
            sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
            valid_division = (
                max(d1, d2) <= DIV_PARENT_MAX_UM
                and sister <= DIV_SISTER_MAX_UM
                and int(nodes_by_id[int(top1["target_id"])] ["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])] ["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [edge for edge in edges if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id]

    print(f"  [{dataset}] after division-geometry-filter+prune-isolated: {len(nodes_by_id)} nodes, {len(edges)} edges")
    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    print(f"  [{dataset}] after short-track filtering: {len(nodes_by_id)} nodes, {len(edges)} edges"
          f" (components_removed={stats['short_track_components_removed']})")
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)
    print(f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")

    return nodes_by_id, edges, stats


DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()

def write_test_submission(tag: str = "base") -> None:
    
    
    geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
    print(f"Found {len(geffs)} prediction graphs")
    if len(geffs) != len(test_stems):
        found = {path.stem for path in geffs}
        missing = sorted(set(test_stems) - found)
        raise RuntimeError(f"Expected {len(test_stems)} graphs, found {len(geffs)}. Missing: {missing[:10]}")

    stats_rows: list[dict[str, object]] = []
    seen_datasets: set[str] = set()
    row_id = 0
    total_nodes = 0
    total_edges = 0

    with SUBMISSION_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for geff_path in geffs:
            dataset = geff_path.stem
            seen_datasets.add(dataset)
            graph = graph_from_geff(geff_path)

            nodes_by_id: dict[int, dict[str, object]] = {}
            for row in graph.node_attrs().iter_rows(named=True):
                node_id = int(row["node_id"])
                nodes_by_id[node_id] = {
                    "node_id": node_id,
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }

            raw_edges: list[dict[str, object]] = []
            for row in graph.edge_attrs().iter_rows(named=True):
                edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
                raw_edges.append({
                    "source_id": int(row["source_id"]),
                    "target_id": int(row["target_id"]),
                    "edge_prob": None if edge_prob is None else float(edge_prob),
                })

            raw_node_count = len(nodes_by_id)
            nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
            if not nodes_by_id:
                raise AssertionError(f"{dataset}: post-processing removed every node")

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": max(0, int(round(float(node["z"])))),
                    "y": max(0, int(round(float(node["y"])))),
                    "x": max(0, int(round(float(node["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                })
                row_id += 1

            division_sources: dict[int, int] = {}
            for edge in edges:
                source_id = int(edge["source_id"])
                target_id = int(edge["target_id"])
                if source_id not in nodes_by_id or target_id not in nodes_by_id:
                    raise AssertionError(f"{dataset}: dangling edge after filtering")
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": source_id,
                    "target_id": target_id,
                })
                row_id += 1
                division_sources[source_id] = division_sources.get(source_id, 0) + 1

            node_count = len(nodes_by_id)
            edge_count = len(edges)
            total_nodes += node_count
            total_edges += edge_count
            stats_rows.append({
                "dataset": dataset,
                "raw_nodes": raw_node_count,
                "nodes": node_count,
                "raw_edges": filter_stats["raw_edges"],
                "edges": edge_count,
                "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
                "edge_to_node_ratio": edge_count / max(node_count, 1),
                "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
                **filter_stats,
            })

    expected_datasets = set(test_stems)
    missing_datasets = sorted(expected_datasets - seen_datasets)
    extra_datasets = sorted(seen_datasets - expected_datasets)
    if missing_datasets or extra_datasets:
        raise AssertionError({"missing": missing_datasets[:10], "extra": extra_datasets[:10]})
    assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
    assert total_nodes > 0, "No node rows produced"

    header = SUBMISSION_PATH.open().readline().strip().split(",")
    assert header == CSV_COLUMNS, f"Bad CSV header: {header}"

    stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
    stats["predict_minutes_total"] = predict_seconds / 60.0
    stats["experiment_tag"] = f"{EXPERIMENT_TAG}:{tag}"
    stats.to_csv(RUN_STATS_PATH, index=False)

    print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")
    print(f"Node rows: {total_nodes:,} | edge rows: {total_edges:,}")
    print(f"Wrote {RUN_STATS_PATH}")
    display(pd.read_csv(SUBMISSION_PATH, nrows=8))


print("EXP019 CV-only: no test submission is generated.")

