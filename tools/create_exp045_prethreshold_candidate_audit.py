"""Create EXP045: audit raw edge scores before threshold and degree budgets."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP010" / "CELL_infer_public_0942.ipynb"
OUTPUT = ROOT / "EXP" / "EXP045" / "CELL_prethreshold_candidate_audit.ipynb"
MODULE = ROOT / "tools" / "audit_prethreshold_candidates.py"


RAW_CAPTURE = r'''
            # EXP045: retain a compact union of source/target top-k pairs before
            # the probability threshold and greedy degree budgets are applied.
            _audit_top_k = int(os.environ.get("BIOHUB_PRETHRESHOLD_TOP_K", "16"))
            if _audit_top_k < 1:
                raise ValueError("BIOHUB_PRETHRESHOLD_TOP_K must be positive")
            _raw_pairs: set[tuple[int, int]] = set()
            _source_ranks: dict[tuple[int, int], int] = {}
            _target_ranks: dict[tuple[int, int], int] = {}
            for _i in range(n_src):
                _source_order = np.argsort(-probs[_i], kind="stable")
                for _rank, _j in enumerate(_source_order[:_audit_top_k], start=1):
                    _j = int(_j)
                    _raw_pairs.add((_i, _j))
                    _source_ranks[(_i, _j)] = int(_rank)
            for _j in range(n_tgt):
                _target_order = np.argsort(-probs[:, _j], kind="stable")
                for _rank, _i in enumerate(_target_order[:_audit_top_k], start=1):
                    _i = int(_i)
                    _raw_pairs.add((_i, _j))
                    _target_ranks[(_i, _j)] = int(_rank)
            for _i, _j in sorted(_raw_pairs):
                _gi, _gj = int(idx_src[_i]), int(idx_tgt[_j])
                prethreshold_edges.append((
                    _gi,
                    _gj,
                    float(probs[_i, _j]),
                    int(_source_ranks.get((_i, _j), 0)),
                    int(_target_ranks.get((_i, _j), 0)),
                    int(t_src),
                ))
'''


FINAL_RUNTIME = r'''
    _candidate_audit_arm = os.environ.get("BIOHUB_CANDIDATE_AUDIT", "").strip()
    if _candidate_audit_arm:
        _candidate_shard = os.environ.get("BIOHUB_GPU_SHARD", "single").replace("/", "_")
        _candidate_edge_array = np.asarray(all_edges, dtype=np.float64).reshape((-1, 4))
        _prethreshold_edge_array = np.asarray(prethreshold_edges, dtype=np.float64).reshape((-1, 6))
        _candidate_path = (
            Path("/kaggle/working")
            / f"edge_candidates_{_candidate_audit_arm}_{_candidate_shard}_{ds_path.stem}.npz"
        )
        np.savez_compressed(
            _candidate_path,
            coords=np.ascontiguousarray(coords.astype(np.float32, copy=False)),
            edges=_candidate_edge_array,
            raw_edges=_prethreshold_edge_array,
        )
'''


def audit_cell() -> dict:
    encoded = base64.b64encode(MODULE.read_bytes()).decode("ascii")
    source = f'''# EXP045: audit raw candidate scores before threshold and budgets.
# This is diagnostic only. It never changes the graph or competition output.
import base64 as _exp045_base64
import subprocess as _exp045_subprocess

_exp045_tool = WORKING_DIR / "audit_prethreshold_candidates.py"
_exp045_tool.write_bytes(_exp045_base64.b64decode({encoded!r}))
_exp045_out = WORKING_DIR / "prethreshold_candidate_audit"
_exp045_cmd = [
    sys.executable, str(_exp045_tool),
    "--pred-dir", str(WORKING_DIR),
    "--gt-dir", str(TRAIN_DIR),
    "--out", str(_exp045_out),
    "--stems", *val_stems,
]
_exp045_run = _exp045_subprocess.run(_exp045_cmd, text=True, capture_output=True)
print(_exp045_run.stdout)
if _exp045_run.returncode != 0:
    print(_exp045_run.stderr)
    raise RuntimeError("EXP045 pre-threshold candidate audit failed")
if not (_exp045_out / "REPORT.md").is_file():
    raise RuntimeError("EXP045 audit report was not created")
print("EXP045 pre-threshold candidate audit complete:", _exp045_out)
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
        "id": "exp045-prethreshold-candidate-audit",
    }


def cleanup_cell() -> dict:
    source = '''# EXP045 is diagnostic-only: prevent accidental competition submission.
_exp045_submission = WORKING_DIR / "submission.csv"
_exp045_reference = WORKING_DIR / "prethreshold_reference_submission.csv"
if _exp045_submission.exists():
    _exp045_submission.replace(_exp045_reference)
print("EXP045 diagnostic complete; submission.csv was renamed to", _exp045_reference)
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
        "id": "exp045-no-submission-cleanup",
    }


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # The optional public-provenance display is not part of the inference
    # contract and is not guaranteed to be mounted in this diagnostic kernel.
    cells[2]["source"] = [
        "# EXP045: optional public-provenance display is intentionally skipped.\n",
        'print("EXP045: skipped optional provenance display; audit inputs are self-contained.")\n',
    ]

    cell4 = "".join(cells[4]["source"])
    marker = 'os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"'
    if marker not in cell4:
        raise RuntimeError("EXP010 diagnostic-arm marker not found")
    cell4 = cell4.replace(
        marker,
        'os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "prethreshold_audit"\n'
        'os.environ["BIOHUB_CANDIDATE_AUDIT"] = "prethreshold"\n'
        'os.environ["BIOHUB_PRETHRESHOLD_TOP_K"] = "16"',
        1,
    )
    cells[4]["source"] = cell4.splitlines(True)

    # Cell 8 edits the runtime prediction script.  Remove the older coordinate
    # manifest patch and replace it with the two robust raw-edge hooks below.
    cell8 = "".join(cells[8]["source"])
    coordinate_start = cell8.find("_coordinate_manifest_old =")
    list_test_marker = "def list_test_stems"
    coordinate_end = cell8.find(list_test_marker, coordinate_start)
    if coordinate_start < 0 or coordinate_end < 0:
        raise RuntimeError("EXP010 runtime patch boundaries not found")
    coordinate_block = cell8[coordinate_start:coordinate_end]
    compile_marker = 'compile(_s, str(_ps), "exec")'
    compile_index = coordinate_block.find(compile_marker)
    if compile_index < 0:
        raise RuntimeError("EXP010 runtime compile marker not found")
    compile_tail = coordinate_block[compile_index:]
    compile_tail = compile_tail.replace(
        'print("Pre-ILP detector-coordinate manifest hook applied")',
        'print("EXP045 pre-threshold candidate hooks applied")',
    )

    runtime_patch = '''
import re as _exp045_re
_prethreshold_runtime_init = ''' + repr(
        '''    prethreshold_edges: list[tuple[int, int, float, int, int, int]] = []\n'''
    ) + '''
_prethreshold_init_marker = "    all_edges: list[tuple[int, int, float, float]] = []"
_prethreshold_init_count = _s.count(_prethreshold_init_marker)
if _prethreshold_init_count != 1:
    raise RuntimeError(
        "EXP045 expected one all_edges initialization, found "
        f"{_prethreshold_init_count}"
    )
_s = _s.replace(
    _prethreshold_init_marker,
    _prethreshold_init_marker + "\\n" + _prethreshold_runtime_init,
    1,
)
_prethreshold_capture = ''' + repr(RAW_CAPTURE) + '''
_prethreshold_prob_marker = "            else:\\n                probs = torch.sigmoid(raw).cpu().numpy()"
_prethreshold_prob_count = _s.count(_prethreshold_prob_marker)
if _prethreshold_prob_count != 1:
    raise RuntimeError(
        "EXP045 expected one probability conversion block, found "
        f"{_prethreshold_prob_count}"
    )
_s = _s.replace(
    _prethreshold_prob_marker,
    _prethreshold_prob_marker + "\\n" + _prethreshold_capture,
    1,
)
_candidate_runtime_new = ''' + repr(FINAL_RUNTIME) + '''
_candidate_return_pattern = _exp045_re.compile(
    r"(?m)^(?P<indent>[ \\t]*)return\\s+coords\\s*,\\s*all_edges\\s*$"
)
_candidate_matches = list(_candidate_return_pattern.finditer(_s))
if len(_candidate_matches) != 1:
    raise RuntimeError(
        "EXP045 expected one return coords/all_edges line, found "
        f"{len(_candidate_matches)}"
    )
_candidate_match = _candidate_matches[0]
_s = (
    _s[:_candidate_match.start()]
    + _candidate_runtime_new
    + _s[_candidate_match.start():]
)
'''
    cell8 = cell8[:coordinate_start] + runtime_patch + compile_tail + cell8[coordinate_end:]
    cells[8]["source"] = cell8.splitlines(True)

    # Validation prediction is complete before this inserted audit cell in the
    # public notebook, matching the EXP044 placement.
    cells.insert(12, audit_cell())
    cells.append(cleanup_cell())
    cells[0]["source"] = [
        "# EXP045: Pre-threshold candidate score audit\n\n",
        "以 EXP010 的冻结双种子推理链为底座，保存候选阈值和节点预算之前的 top-16 原始边。\n",
        "本 Notebook 只在四个完整训练视频上做标签审计，不修改图、不产生正式 CV/LB。\n",
    ]
    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP045 Pre-Threshold Candidate Score Audit"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
