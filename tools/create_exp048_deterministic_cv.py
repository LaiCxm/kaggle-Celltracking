"""Create EXP048: deterministic official-CV control for graph reproducibility."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "EXP" / "EXP047" / "CELL_division_second_edge_rescue_off_control.ipynb"
OUTPUT = ROOT / "EXP" / "EXP048" / "CELL_deterministic_official_cv_audit.ipynb"


DETERMINISM_PATCH = r'''
# EXP048: deterministic runtime guard for repeatable graph outputs.
import re as _exp048_re
_exp048_seed = int(os.environ.get("BIOHUB_DETERMINISTIC_SEED", "20260923"))
os.environ["PYTHONHASHSEED"] = str(_exp048_seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_s = _ps.read_text()
_exp048_anchor = "import torch\n"
_exp048_code = """
import random as _exp048_random
_exp048_random.seed(__SEED__)
np.random.seed(__SEED__)
torch.manual_seed(__SEED__)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(__SEED__)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
""".replace("__SEED__", str(_exp048_seed))
if _s.count(_exp048_anchor) != 1:
    raise RuntimeError("EXP048 expected one torch import anchor")
_s = _s.replace(_exp048_anchor, _exp048_anchor + _exp048_code, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print("EXP048 deterministic runtime guard applied", _exp048_seed)
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    cell0 += (
        "\n# EXP048 deterministic audit; EXP047 rescue remains disabled.\n"
        'os.environ["BIOHUB_DETERMINISTIC_SEED"] = "20260923"\n'
        'os.environ["PYTHONHASHSEED"] = "20260923"\n'
        'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\n'
    )
    notebook["cells"][0]["source"] = cell0.splitlines(True)

    cell4 = "".join(notebook["cells"][4]["source"])
    anchor = "_ps = REPO_DIR / \"scripts\" / \"predict_unet_transformer.py\"\n"
    if cell4.count(anchor) != 1:
        raise RuntimeError("EXP048 notebook runtime anchor not found")
    cell4 = cell4.replace(anchor, anchor + DETERMINISM_PATCH, 1)
    notebook["cells"][4]["source"] = cell4.splitlines(True)

    notebook.setdefault("metadata", {}).setdefault("kaggle", {})["title"] = (
        "EXP048 Deterministic Official CV Audit"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
