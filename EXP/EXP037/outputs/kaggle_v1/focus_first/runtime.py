"""One-pass FOCUS3D runtime adapter for EXP021.

Unlike the legacy ``FocusMaskProvider``, this adapter keeps the final
confidence map and run metadata together with the instance map.  It is lazy,
frame-cached, and has no graph mutation side effects.  The Kaggle notebook can
wrap ``get`` with its own image loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Callable, Any

import numpy as np


@dataclass(frozen=True)
class FocusFrameObservation:
    dataset: str
    t: int
    instance_map: np.ndarray
    confidence_map: np.ndarray | None
    log_info: dict[str, Any]


class FocusObservationProvider:
    """Lazy, one-inference-per-frame FOCUS3D observation cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
        max_frames: int = 80,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.scale_um = tuple(float(v) for v in scale_um)
        self.max_frames = int(max_frames)
        self._observations: dict[tuple[str, int], FocusFrameObservation] = {}
        self._model = None
        self._config: Path | None = None
        self._weights: Path | None = None
        self._infer_volume = None
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
        root = next((path for path in roots if str(path) and (path / "focus3d_runtime").exists()), None)
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
        except Exception as exc:  # pragma: no cover - Kaggle-only dependency
            self.error = f"FOCUS3D load failed: {type(exc).__name__}: {exc}"
            return False

    def get(
        self,
        dataset: str,
        t: int,
        frame_loader: Callable[[int], np.ndarray],
    ) -> FocusFrameObservation | None:
        key = (str(dataset), int(t))
        if key in self._observations:
            return self._observations[key]
        if len(self._observations) >= self.max_frames or not self._load():
            return None
        try:
            import tifffile

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{dataset}__t{int(t):04d}.tif"
            frame = np.asarray(frame_loader(int(t)))
            if not path.exists():
                tifffile.imwrite(path, frame)
            result = self._infer_volume(
                image_path=str(path),
                config_file=str(self._config),
                weights_path=str(self._weights),
                model=self._model,
                device="cuda",
                output_dir=str(self.cache_dir / "outputs"),
                z_ratio=self.scale_um[0] / self.scale_um[1],
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
            instance_map = np.asarray(result["instance_map"], dtype=np.int32)
            if instance_map.ndim != 3:
                raise ValueError(f"instance_map shape={instance_map.shape}")
            confidence = result.get("confidence_map")
            confidence_map = None if confidence is None else np.asarray(confidence, dtype=np.float32)
            if confidence_map is not None and confidence_map.shape != instance_map.shape:
                raise ValueError(
                    f"confidence_map shape={confidence_map.shape} != instance_map shape={instance_map.shape}"
                )
            log_info = dict(result.get("log_info") or {})
            observation = FocusFrameObservation(
                dataset=str(dataset),
                t=int(t),
                instance_map=instance_map,
                confidence_map=confidence_map,
                log_info=log_info,
            )
            self._observations[key] = observation
            self.calls += 1
            return observation
        except Exception as exc:  # pragma: no cover - Kaggle-only dependency
            self.error = f"FOCUS3D inference failed at {dataset}/t{t}: {type(exc).__name__}: {exc}"
            return None

