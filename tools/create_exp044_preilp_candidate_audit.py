"""Create EXP044: label-free pre-ILP candidate edge audit notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP010" / "CELL_infer_public_0942.ipynb"
OUTPUT = ROOT / "EXP" / "EXP044" / "CELL_preilp_candidate_audit.ipynb"
MODULE = ROOT / "tools" / "audit_preilp_candidates.py"


CANDIDATE_RUNTIME = r'''
    _candidate_audit_arm = os.environ.get("BIOHUB_CANDIDATE_AUDIT", "").strip()
    if _candidate_audit_arm:
        _candidate_shard = os.environ.get("BIOHUB_GPU_SHARD", "single").replace("/", "_")
        _candidate_edge_array = np.asarray(all_edges, dtype=np.float64).reshape((-1, 4))
        _candidate_path = (
            Path("/kaggle/working")
            / f"edge_candidates_{_candidate_audit_arm}_{_candidate_shard}_{ds_path.stem}.npz"
        )
        np.savez_compressed(
            _candidate_path,
            coords=np.ascontiguousarray(coords.astype(np.float32, copy=False)),
            edges=_candidate_edge_array,
        )

'''


def audit_cell() -> dict:
    encoded = base64.b64encode(MODULE.read_bytes()).decode("ascii")
    source = f'''# EXP044: inspect the candidate edge pool immediately before ILP.
# This is a diagnostic only. It never changes the validation or test graph.
import base64 as _exp044_base64
import subprocess as _exp044_subprocess

_exp044_tool = WORKING_DIR / "audit_preilp_candidates.py"
_exp044_tool.write_bytes(_exp044_base64.b64decode({encoded!r}))
_exp044_out = WORKING_DIR / "preilp_candidate_audit"
_exp044_cmd = [
    sys.executable, str(_exp044_tool),
    "--pred-dir", str(WORKING_DIR),
    "--gt-dir", str(TRAIN_DIR),
    "--out", str(_exp044_out),
    "--stems", *val_stems,
]
_exp044_run = _exp044_subprocess.run(_exp044_cmd, text=True, capture_output=True)
print(_exp044_run.stdout)
if _exp044_run.returncode != 0:
    print(_exp044_run.stderr)
    raise RuntimeError("EXP044 pre-ILP candidate audit failed")
if not (_exp044_out / "REPORT.md").is_file():
    raise RuntimeError("EXP044 audit report was not created")
print("EXP044 pre-ILP candidate audit complete:", _exp044_out)
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
        "id": "exp044-preilp-candidate-audit",
    }


def cleanup_cell() -> dict:
    source = '''# EXP044 is diagnostic-only: prevent accidental competition submission.
_exp044_submission = WORKING_DIR / "submission.csv"
_exp044_reference = WORKING_DIR / "preilp_reference_submission.csv"
if _exp044_submission.exists():
    _exp044_submission.replace(_exp044_reference)
print("EXP044 diagnostic complete; submission.csv was renamed to", _exp044_reference)
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
        "id": "exp044-no-submission-cleanup",
    }


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # The copied EXP010 notebook starts with an optional public-provenance
    # display cell.  It requires three CSV files that are irrelevant to this
    # audit and are not part of the runtime data contract.  Keeping that cell
    # would make a diagnostic run fail before inference when the optional
    # provenance dataset is not attached.
    cells[2]["source"] = [
        "# EXP044: optional public-provenance display is intentionally skipped.\n",
        "print(\"EXP044: skipped optional provenance display; audit inputs are self-contained.\")\n",
    ]

    # The copied public pipeline remains unchanged except for a label-free
    # manifest written by predict_video before graph construction and ILP.
    cell4 = "".join(cells[4]["source"])
    cell4 = cell4.replace(
        'os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"',
        'os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "preilp_audit"\n'
        'os.environ["BIOHUB_CANDIDATE_AUDIT"] = "preilp"',
    )
    cells[4]["source"] = cell4.splitlines(True)

    cell8 = "".join(cells[8]["source"])
    # The public notebook contains an older coordinate-manifest patch that is
    # neither needed nor stable across support-pack revisions.  Remove that
    # whole block and inject only the EXP044 edge-pool hook.  The hook uses a
    # whitespace-tolerant regular expression because support packs have
    # changed the exact formatting of the final return line.
    coordinate_start = cell8.find("_coordinate_manifest_old =")
    list_test_marker = "def list_test_stems"
    coordinate_end = cell8.find(list_test_marker, coordinate_start)
    if coordinate_start < 0 or coordinate_end < 0:
        raise RuntimeError("EXP010 coordinate patch block boundaries not found")
    coordinate_block = cell8[coordinate_start:coordinate_end]
    compile_marker = 'compile(_s, str(_ps), "exec")'
    compile_index = coordinate_block.find(compile_marker)
    if compile_index < 0:
        raise RuntimeError("EXP010 script compile marker not found")
    compile_tail = coordinate_block[compile_index:]
    compile_tail = compile_tail.replace(
        'print("Pre-ILP detector-coordinate manifest hook applied")',
        'print("EXP044 pre-ILP candidate-edge hook applied")',
    )
    insertion = '''
import re as _exp044_re
_candidate_runtime_new = ''' + repr(CANDIDATE_RUNTIME) + '''
_candidate_return_pattern = _exp044_re.compile(
    r"(?m)^(?P<indent>[ \\t]*)return\\s+coords\\s*,\\s*all_edges\\s*$"
)
_candidate_matches = list(_candidate_return_pattern.finditer(_s))
if len(_candidate_matches) != 1:
    raise RuntimeError(
        "EXP044 candidate hook expected one return coords/all_edges line, found "
        f"{len(_candidate_matches)}"
    )
_candidate_match = _candidate_matches[0]
_s = (
    _s[:_candidate_match.start()]
    + _candidate_runtime_new
    + _s[_candidate_match.start():]
)
'''
    cell8 = cell8[:coordinate_start] + insertion + compile_tail + cell8[coordinate_end:]
    cells[8]["source"] = cell8.splitlines(True)

    # Validation prediction is complete in Cell 11; audit its manifests before
    # the existing official-style diagnostic scorer runs.
    cells.insert(12, audit_cell())
    cells.append(cleanup_cell())
    cells[0]["source"] = [
        "# EXP044: Pre-ILP candidate edge audit\n\n",
        "以 EXP010 的冻结双种子推理链为底座，只记录检测后、建图/ILP 前的候选边池。\n",
        "本 Notebook 只在四个完整训练视频上做标签审计，不修改图、不生成可提交评分结果。\n",
    ]
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP044 Pre-ILP Candidate Edge Audit"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
