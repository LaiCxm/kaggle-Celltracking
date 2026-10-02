"""push_inference.py — 渲染推理提交内核并推送到 Kaggle 执行。

用法:
  python tools/push_inference.py --exp EXP000 --child child-exp000 --competition rsna-knee-abnormality-detection [--dry]
"""
import argparse
import base64
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "infer_kernel_template.ipynb"


def get_username(allow_placeholder=False):
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    cfg_dir = pathlib.Path(os.environ.get("KAGGLE_CONFIG_DIR", pathlib.Path.home() / ".kaggle"))
    kp = cfg_dir / "kaggle.json"
    if kp.exists():
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
    if allow_placeholder:
        return "YOUR_USERNAME"
    raise SystemExit("无法确定 KAGGLE_USERNAME:请配置 kaggle.json 或设置环境变量")


def restore_cell(path: pathlib.Path, target_name: str = None) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    name = target_name or path.name
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "import base64, pathlib\n",
            f'pathlib.Path("/kaggle/working/{name}").write_bytes(base64.b64decode("{b64}"))\n',
            f'print("restored {name}")\n',
        ],
    }


def render(exp: str, child: str, competition: str, gpu: bool,
           no_competition: bool = False, data_sources: list = None) -> pathlib.Path:
    exp_dir = ROOT / "EXP" / exp
    cfg_yaml = exp_dir / "config" / f"{child}.yaml"
    infer_py = exp_dir / "infer.py"
    train_py = exp_dir / "train.py"
    if not cfg_yaml.exists() or not infer_py.exists():
        raise SystemExit(f"缺少文件:{cfg_yaml} 或 {infer_py}")

    nb = json.loads(TEMPLATE.read_text(encoding="utf-8-sig"))
    markers = {"__EXP_NAME__": exp, "__CHILD_EXP_NAME__": child, "__COMPETITION__": competition}
    for cell in nb["cells"]:
        for i, line in enumerate(cell.get("source", [])):
            for marker, value in markers.items():
                if marker in line:
                    cell["source"][i] = line.replace(marker, value)

    restore = [restore_cell(cfg_yaml, "config.yaml")]
    for name in ("train.py", "metric.py", "infer.py"):
        p = exp_dir / name
        if p.exists():
            restore.append(restore_cell(p))
    nb["cells"][2:2] = restore

    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="kernel_infer_"))
    (out_dir / "infer_submit.ipynb").write_text(json.dumps(nb), encoding="utf-8")

    user = get_username(allow_placeholder=True)
    results_sources = [f"{user}/exp-results-{exp.lower()}-{child.lower()}"]
    slug = f"{exp.lower()}-{child.lower()}-infer"
    metadata = {
        "id": f"{user}/{slug}",
        "title": f"{exp.lower()} {child.lower()} infer",
        "code_file": "infer_submit.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "competition_sources": [] if no_competition else [f"competitions/{competition}"],
        "dataset_sources": (data_sources or []) + results_sources,
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--child", required=True)
    ap.add_argument("--competition", required=True)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-competition", action="store_true")
    ap.add_argument("--data-source", action="append", default=None,
                    help="额外挂载的 dataset,格式 user/slug,可重复")
    args = ap.parse_args()

    out_dir = render(args.exp, args.child, args.competition, gpu=not args.no_gpu,
                     no_competition=args.no_competition, data_sources=args.data_source)
    print(f"[1/2] 渲染完成: {out_dir}")

    if args.dry:
        print("[dry] 跳过推送。")
        return

    slug = f"{args.exp.lower()}-{args.child.lower()}-infer"
    print(f"[2/2] 推送内核 {get_username()}/{slug} ...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(out_dir)],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"推送成功。查看状态: kaggle kernels status {get_username()}/{slug}")


if __name__ == "__main__":
    main()
