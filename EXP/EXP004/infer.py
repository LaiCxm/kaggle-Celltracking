"""EXP004 推理入口（工作流契约兼容）。

EXP004 是纯推理复现实验，核心逻辑在 CELL_infer_harmonic_divsub.ipynb 中。
真实推理请用：
  python tools/push_custom_kernel.py --exp EXP004 --kernel CELL_infer_harmonic_divsub.ipynb ...
"""
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--submission", default="/kaggle/working/submission.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        import pandas as pd
        rows = [
            [0, "dummy", "node", 1, 0, 32, 128, 128, -1, -1],
            [1, "dummy", "edge", -1, -1, -1, -1, -1, 1, 2],
        ]
        df = pd.DataFrame(rows, columns=["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
        Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.submission, index=False)
        print("SMOKE_OK", args.submission)
        return

    raise SystemExit("EXP004 真实推理请用 GPU 内核: tools/push_custom_kernel.py --kernel CELL_infer_harmonic_divsub.ipynb")

if __name__ == "__main__":
    main()
