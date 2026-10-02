"""push_custom_kernel.py — 把 EXP/{exp} 下的指定 notebook 直接推为推理内核。

适用场景:复现类实验(纯推理、零训练),notebook 本身自包含完整逻辑,
不需要走 push_inference.py 的 infer.py 契约。

用法:
  python tools/push_custom_kernel.py --exp EXP001 --kernel infer_clean_repro.ipynb \
      --competition biohub-cell-tracking-during-development \
      --data-source pilkwang/biohub-tracking-support-pack-50ep-v1 \
      [--no-gpu] [--dry]

输出内核 id: {user}/{exp_lower}-{kernel_lower}-infer
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--kernel", required=True, help="EXP/{exp} 下的 .ipynb 文件名")
    ap.add_argument("--competition", required=True)
    ap.add_argument("--data-source", action="append", default=None,
                    help="挂载的 dataset,格式 owner/slug,可重复")
    ap.add_argument("--kernel-slug", default=None,
                    help="自定义内核名(不含用户前缀),默认 {exp_lower}-{kernel_lower}-infer")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4",
                    help="Kaggle machine_shape(SDK 枚举): NvidiaTeslaT4 / NvidiaTeslaP100 / Tpu1VmV38。默认 NvidiaTeslaT4(本比赛仅允许 T4 x2)")
    ap.add_argument("--no-internet", action="store_true",
                    help="禁用内核 internet 访问(某些比赛禁止外网,依赖必须走离线 wheels)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    kernel_path = ROOT / "EXP" / args.exp / args.kernel
    if not kernel_path.exists():
        raise SystemExit(f"内核不存在:{kernel_path}")

    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="kernel_custom_"))
    shutil = __import__("shutil")
    shutil.copy(kernel_path, out_dir / args.kernel)

    user = get_username(allow_placeholder=True)
    base = f"{args.exp.lower()}-{args.kernel.lower().replace('.ipynb', '')}"
    slug = args.kernel_slug or f"{base}-infer"
    metadata = {
        "id": f"{user}/{slug}",
        "title": args.title or f"{args.exp.lower()} {args.kernel.lower()} infer",
        "code_file": args.kernel,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not args.no_gpu,
        "enable_internet": not args.no_internet,
        "machine_shape": "none" if args.no_gpu else args.accelerator,        "competition_sources": [f"competitions/{args.competition}"],
        "dataset_sources": args.data_source or [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[1/2] 内核目录:{out_dir}")
    print("metadata:", json.dumps(metadata, indent=2))

    if args.dry:
        print("[dry] 跳过推送。检查 kernel-metadata.json 与 notebook 即可。")
        return

    print(f"[2/2] 推送内核 {user}/{slug} ...")
    push_cmd = ["kaggle", "kernels", "push", "-p", str(out_dir)]
    if not args.no_gpu:
        push_cmd += ["--accelerator", args.accelerator]
    r = subprocess.run(push_cmd, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"推送成功。状态: kaggle kernels status {user}/{slug}")


if __name__ == "__main__":
    main()
