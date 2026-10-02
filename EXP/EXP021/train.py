"""EXP021 compatibility entry point.

EXP021 V1 is an inference/candidate-layer experiment and intentionally has no
trainable network. This entry point preserves the repository CLI contract while
making that fact explicit in ``outputs/results.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP021 candidate-layer diagnostic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--use_wandb", default="False")
    parser.add_argument("--create_oof", default="True")
    args = parser.parse_args()
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "EXP021",
        "stage": "observation_and_candidates",
        "status": "diagnostic_only",
        "cv": None,
        "mean_cv": None,
        "oof_score": None,
        "folds": [int(item) for item in str(args.folds).split(",") if item.strip()],
        "message": "No trainable model or official CV is run in EXP021 V1.",
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["status", "score"])
        csv.writer(handle).writerow(["diagnostic_only", ""])
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
