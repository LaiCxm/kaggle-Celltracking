from __future__ import annotations

"""Create the EXP011 no-division submission ablation from EXP010."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP010" / "CELL_infer_public_0942.ipynb"
TARGET_DIR = ROOT / "EXP" / "EXP011"
TARGET = TARGET_DIR / "CELL_infer_no_division_ablation.ipynb"

CONFIG_INSERT = '''
# EXP011 ablation: run the full EXP010 pipeline, but remove fork edges only
# immediately before writing submission.csv. Nodes and one best continuation
# per source are retained, so this is a division-output ablation rather than
# a detector or edge-model ablation.
os.environ["BIOHUB_ABLATION_DISABLE_DIVISION"] = "1"
'''

FINAL_INSERT_FUNCTION = '''

def ablation_remove_division_edges(edges: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Remove all forks at the final serialization boundary."""
    by_source: dict[int, list[dict[str, object]]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(edge)
    kept: list[dict[str, object]] = []
    removed = 0
    sources = 0
    for source_id, source_edges in by_source.items():
        if len(source_edges) > 1:
            sources += 1
            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            kept.append(ranked[0])
            removed += len(ranked) - 1
        else:
            kept.extend(source_edges)
    return kept, {
        "ablation_division_sources": sources,
        "ablation_division_edges_removed": removed,
    }
'''

FINAL_INSERT_CALL = '''
        if os.environ.get("BIOHUB_ABLATION_DISABLE_DIVISION", "0") != "0":
            edges, _ablation_stats = ablation_remove_division_edges(edges)
            print(
                f"  [{dataset}] EXP011 no-division ablation: "
                f"sources={_ablation_stats['ablation_division_sources']}, "
                f"removed_edges={_ablation_stats['ablation_division_edges_removed']}, "
                f"remaining_edges={len(edges)}",
                flush=True,
            )
'''


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = payload["cells"]

    intro = "".join(cells[0].get("source", []))
    intro = intro.replace(
        "# Biohub 0.942 LB, one knob past the public line",
        "# EXP011 no-division ablation of the public 0.942 pipeline",
        1,
    )
    intro += (
        "\n\nThis EXP011 ablation runs the complete EXP010 frozen-weight pipeline. "
        "It removes only final fork edges before submission serialization, "
        "so the score difference measures the cost of submitting no divisions "
        "under the same node and continuation pipeline.\n"
    )
    cells[0]["source"] = intro.splitlines(keepends=True)

    config = "".join(cells[4].get("source", []))
    if "BIOHUB_ABLATION_DISABLE_DIVISION" not in config:
        config += CONFIG_INSERT
    cells[4]["source"] = config.splitlines(keepends=True)

    output = "".join(cells[9].get("source", []))
    if "def ablation_remove_division_edges" not in output:
        function_anchor = "with SUBMISSION_PATH.open(\"w\", newline=\"\") as f:\n"
        if function_anchor not in output:
            raise RuntimeError("submission helper anchor not found")
        output = output.replace(function_anchor, FINAL_INSERT_FUNCTION + "\n" + function_anchor, 1)
        call_anchor = "        if not nodes_by_id:\n            raise AssertionError(f\"{dataset}: post-processing removed every node\")\n"
        if call_anchor not in output:
            raise RuntimeError("submission serialization anchor not found")
        output = output.replace(call_anchor, call_anchor + FINAL_INSERT_CALL, 1)
    cells[9]["source"] = output.splitlines(keepends=True)

    manifest = "".join(cells[13].get("source", []))
    manifest += (
        "\nprint(\"EXP011 no-division ablation: \" + "
        "str(os.environ.get(\"BIOHUB_ABLATION_DISABLE_DIVISION\", \"0\") != \"0\"))\n"
    )
    cells[13]["source"] = manifest.splitlines(keepends=True)

    payload["metadata"]["title"] = "EXP011 no-division ablation"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    shutil.copy2(
        ROOT / "EXP" / "EXP010" / "source-kernel-metadata.json",
        TARGET_DIR / "source-kernel-metadata.json",
    )
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
