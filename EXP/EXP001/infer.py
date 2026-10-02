"""EXP001 推理入口(工作流契约兼容)。

说明:EXP001 是纯推理复现实验,核心逻辑完整包含在 infer_clean_repro.ipynb 中
(自包含依赖安装 + predict_unet_transformer + 后处理 + fixed-8 CV)。
本文件只是满足 AGENTS.md 接口契约的薄包装,真正执行以 notebook 内核为准:
  python tools/push_custom_kernel.py --exp EXP001 --kernel infer_clean_repro.ipynb ...

它支持一个可选的本机/CPU 冒烟模式(--smoke),用规则检测(dummy)验证 pipeline,
不替代 GPU 推理。
"""
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--exp", default="EXP001")
    p.add_argument("--child", default="child-exp000")
    p.add_argument("--competition", default="biohub-cell-tracking-during-development")
    p.add_argument("--submission", default="/kaggle/working/submission.csv")
    p.add_argument("--smoke", action="store_true", help="CPU 冒烟,不跑 GPU 推理")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if str(args.config).endswith("json") else None
    print("EXP001 inference stub. Real logic lives in infer_clean_repro.ipynb (GPU kernel).")
    print("submission ->", args.submission)

    if args.smoke:
        import pandas as pd
        # 冒烟:生成一个最小合法 submission(3 个 node + 2 条 edge),验证 schema
        rows = [
            [0, "dummy", "node", 1, 0, 32, 128, 128, -1, -1],
            [1, "dummy", "node", 2, 1, 32, 128, 128, -1, -1],
            [2, "dummy", "node", 3, 2, 32, 128, 128, -1, -1],
            [3, "dummy", "edge", -1, -1, -1, -1, -1, 1, 2],
            [4, "dummy", "edge", -1, -1, -1, -1, -1, 2, 3],
        ]
        df = pd.DataFrame(rows, columns=["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
        Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.submission, index=False)
        print("SMOKE_OK wrote", args.submission)
        return

    raise SystemExit("EXP001 真实推理请用 GPU 内核: tools/push_custom_kernel.py --kernel infer_clean_repro.ipynb")


if __name__ == "__main__":
    main()
