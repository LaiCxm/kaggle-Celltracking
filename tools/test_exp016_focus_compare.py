from pathlib import Path

import numpy as np
import zarr


def select_annotated_frames(times: np.ndarray, limit: int) -> np.ndarray:
    frames = np.unique(np.asarray(times, dtype=np.int64))
    if len(frames) == 0:
        raise RuntimeError("GT GEFF has no nodes")
    return frames[:limit]


def read_selected_nodes(path: Path, frames: np.ndarray) -> np.ndarray:
    group = zarr.open_group(str(path), mode="r")
    t = np.asarray(group["nodes/props/t/values"][:], dtype=np.int64)
    values = [
        np.asarray(group[f"nodes/props/{axis}/values"][:], dtype=np.float32)
        for axis in "zyx"
    ]
    keep = np.isin(t, frames)
    nodes = np.column_stack((t[keep], *(value[keep] for value in values)))
    if len(nodes) == 0:
        raise RuntimeError("selected frames contain no GT nodes")
    return nodes


def test_sparse_frame_selection() -> None:
    times = np.asarray([17, 17, 23, 41, 41, 56], dtype=np.int64)
    assert select_annotated_frames(times, 3).tolist() == [17, 23, 41]


def test_direct_geff_read(tmp_path: Path) -> None:
    group = zarr.open_group(str(tmp_path / "sample.geff"), mode="w")
    group.create_array("nodes/props/t/values", data=np.asarray([7, 11, 11, 20]))
    for offset, axis in enumerate("zyx"):
        group.create_array(
            f"nodes/props/{axis}/values",
            data=np.asarray([1, 2, 3, 4], dtype=np.float32) + offset,
        )
    frames = select_annotated_frames(group["nodes/props/t/values"][:], 2)
    nodes = read_selected_nodes(tmp_path / "sample.geff", frames)
    assert frames.tolist() == [7, 11]
    assert nodes.shape == (3, 4)
    assert nodes[:, 0].astype(int).tolist() == [7, 11, 11]
