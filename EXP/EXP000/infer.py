"""EXP000 推理骨架:从结果 Dataset 加载模型,对 test 推理,生成 submission。

契约:
  - 模型权重从 /kaggle/input/exp-results-{exp}-{child}/models/ 读取
  - 输出 submission.csv,列: id_column + target_cols
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import SimpleCNN, ImageCSVDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--child", required=True)
    p.add_argument("--competition", required=True)
    p.add_argument("--submission", default="/kaggle/working/submission.csv")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config.endswith(".json") \
        else _load_yaml(args.config)

    results_dir = Path(f"/kaggle/input/exp-results-{args.exp.lower()}-{args.child.lower()}")
    data_dir = Path(f"/kaggle/input/competitions/{args.competition}")

    test = pd.read_csv(data_dir / cfg["test_csv"])
    image_col = cfg["image_col"]
    target_cols = cfg["target_cols"]
    id_column = cfg.get("id_column", "id")
    num_classes = len(target_cols)

    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_files = sorted((results_dir / "models").glob("best_model_fold*.pth"))
    if not model_files:
        raise SystemExit(f"未找到模型:{results_dir}/models/")

    ds = ImageCSVDataset(test, data_dir, image_col, target_cols, tfm)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    preds = []
    for mf in model_files:
        model = SimpleCNN(num_classes)
        model.load_state_dict(torch.load(mf, map_location="cpu"))
        model.to(device).eval()
        fold_preds = []
        with torch.no_grad():
            for x, _ in dl:
                fold_preds.append(model(x.to(device)).cpu().numpy())
        preds.append(np.concatenate(fold_preds, axis=0))

    pred = np.mean(preds, axis=0)
    sub = pd.DataFrame(pred, columns=target_cols)
    sub.insert(0, id_column, test[id_column].values)
    sub.to_csv(args.submission, index=False)
    print(sub.head())
    print(f"submission written: {args.submission}")


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
