"""pull_results.py — 从 Kaggle Dataset 拉取训练结果到本地 EXP 目录。

用法:
  python tools/pull_results.py --exp EXP000 --child child-exp000
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def get_username():
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    cfg_dir = pathlib.Path(os.environ.get("KAGGLE_CONFIG_DIR", pathlib.Path.home() / ".kaggle"))
    kp = cfg_dir / "kaggle.json"
    if kp.exists():
        import json
        data = json.loads(kp.read_text(encoding="utf-8"))
        if data.get("username"):
            return data["username"]
    try:
        r = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True)
        m = re.search(r"username:\s*(\S+)", r.stdout + r.stderr)
        if m:
            return m.group(1)
    except Exception:
        pass
    raise SystemExit("无法确定 KAGGLE_USERNAME:请配置 kaggle.json 或设置环境变量")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--child", required=True)
    args = ap.parse_args()

    user = get_username()
    dataset = f"{user}/exp-results-{args.exp.lower()}-{args.child.lower()}"
    dest = ROOT / "EXP" / args.exp / "outputs" / args.child
    dest.mkdir(parents=True, exist_ok=True)

    print(f"下载 {dataset} -> {dest}")
    r = subprocess.run(["kaggle", "datasets", "download", "-d", dataset,
                        "-p", str(dest), "--unzip"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("完成。查看结果: EXP_SUMMARY.md 更新 + 分析 oof_predictions.csv / results.json")


if __name__ == "__main__":
    main()
