"""Run the EXP021 V1 candidate layer on a prepared JSON observation file.

The GPU notebook adapter will later call the same functions directly.  This
small CLI makes the data contract testable without importing Kaggle runtime
code. The input is a JSON list with records containing ``centers`` and one
``focus`` record per frame; masks are supplied as integer ``instance_map``
arrays by Python callers, not embedded in JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.focus_first.candidates import candidate_to_dict, enumerate_continuations, enumerate_divisions
from tools.focus_first.observations import (
    CenterObservation,
    build_unified_nodes,
    node_to_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP021 FOCUS3D-first candidate audit")
    parser.add_argument("--input", type=Path, required=True, help="JSON observation manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("input manifest must be a JSON list")
    # This manifest-only entry point intentionally reports the schema until the
    # Kaggle adapter supplies numpy instance maps and confidence maps.
    nodes = []
    for frame in payload:
        t = int(frame["t"])
        centers = [CenterObservation(**item, t=t) for item in frame.get("centers", [])]
        nodes.extend(node_to_dict(node) for node in build_unified_nodes(centers, []))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "unified_nodes.json").write_text(
        json.dumps(nodes, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "candidate_layer_manifest.json").write_text(
        json.dumps(
            {
                "status": "schema_only_without_instance_maps",
                "nodes": len(nodes),
                "continuations": 0,
                "divisions": 0,
                "note": "Use tools.focus_first directly from the GPU adapter for FOCUS masks.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
