"""Create EXP032: wider-video validation of FOCUS3D mask geometry ranking."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import create_exp031_focus_mask_group_ablation as source
except ImportError:  # pragma: no cover
    import create_exp031_focus_mask_group_ablation as source  # type: ignore


ROOT = source.ROOT
OUT = ROOT / "EXP" / "EXP032" / "CELL_train_wide_mask_geometry.ipynb"
WIDE_SETUP = source.WIDE_SETUP.replace("TARGET_VIDEO_COUNT = 24", "TARGET_VIDEO_COUNT = 48")
RANKER = source.RANKER
OUTPUTS = source.OUTPUTS.replace(
    "exp031_focus_mask_group_ablation", "exp032_wide_mask_geometry"
).replace("'experiment': 'EXP031'", "'experiment': 'EXP032'")
EMBEDDED_MODULES = source.EMBEDDED_MODULES
PILKWANG = source.PILKWANG
FOCUS = source.FOCUS
CANDIDATE_AND_MASK_FEATURES = source.CANDIDATE_AND_MASK_FEATURES
cell = source.cell


def main() -> None:
    notebook = {
        "cells": [
            cell(
                "# EXP032 wide-video FOCUS3D mask geometry validation\n\n"
                "Expand the EXP031 panel from 24 to 48 videos while keeping the "
                "candidate gate and leave-one-video-out grouped ablation fixed.",
                "markdown", "title",
            ),
            cell(EMBEDDED_MODULES(), "code", "embedded-modules"),
            cell(WIDE_SETUP, "code", "wide-event-sampling"),
            cell(PILKWANG, "code", "pilkwang-centers"),
            cell(FOCUS, "code", "focus3d-instances"),
            cell(CANDIDATE_AND_MASK_FEATURES, "code", "candidate-and-mask-features"),
            cell(RANKER, "code", "grouped-oof-rankers"),
            cell(OUTPUTS, "code", "write-diagnostics"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"title": "EXP032 wide-video FOCUS3D mask geometry validation"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
