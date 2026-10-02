"""EXP007 推理入口（工作流契约兼容）。"""
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
        rows = [[0, "dummy", "node", 1, 0, 32, 128, 128, -1, -1]]
        df = pd.DataFrame(rows, columns=["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
        Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.submission, index=False)
        print("SMOKE_OK")
        return
    raise SystemExit("EXP007 请用 GPU 内核: tools/push_custom_kernel.py --kernel CELL_diag_division_score_A.ipynb")

if __name__ == "__main__":
    main()
