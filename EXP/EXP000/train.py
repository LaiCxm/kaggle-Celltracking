"""EXP000 训练骨架:可端到端运行的最小 CNN 基线。

此文件是接口契约的实现参考,agent 新开 EXP 时以此为准重写:
  - CLI 参数固定:--config --folds --use_wandb --create_oof
  - 输出固定:outputs/models/best_model_fold{i}.pth、oof_predictions.csv、results.json
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--use_wandb", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--create_oof", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ImageCSVDataset(Dataset):
    def __init__(self, df, data_dir, image_col, target_cols, tfm):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.image_col = image_col
        self.target_cols = target_cols
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.data_dir / str(row[self.image_col])).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor(row[self.target_cols].astype(np.float32).values)
        return x, y


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(args.output_dir or f"outputs")
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg["data_dir"])
    train = pd.read_csv(data_dir / cfg["train_csv"])
    image_col = cfg["image_col"]
    target_cols = cfg["target_cols"]
    num_classes = len(target_cols)
    metric = cfg.get("metric", "accuracy")

    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg.get("epochs", 3)
    lr = cfg.get("lr", 1e-3)
    batch_size = cfg.get("batch_size", 16)

    folds = [int(f) for f in args.folds.split(",") if f != ""]
    kf = KFold(n_splits=cfg.get("n_folds", 5), shuffle=True, random_state=seed)

    fold_scores = {}
    oof = np.zeros((len(train), num_classes))
    train_idx = np.arange(len(train))

    for fold in folds:
        trn_i, val_i = list(kf.split(train_idx))[fold]
        trn_ds = ImageCSVDataset(train.iloc[trn_i], data_dir, image_col, target_cols, tfm)
        val_ds = ImageCSVDataset(train.iloc[val_i], data_dir, image_col, target_cols, tfm)
        trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        model = SimpleCNN(num_classes).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        best_score, best_state = -1e9, None
        for ep in range(epochs):
            model.train()
            for x, y in trn_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
            model.eval()
            correct = total = 0
            preds = []
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    preds.append(out.argmax(1).cpu().numpy())
                    correct += (out.argmax(1) == y.argmax(1)).sum().item()
                    total += y.size(0)
            score = correct / total if total else 0.0
            print(f"fold {fold} ep {ep}: {metric}={score:.4f}")
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        torch.save(best_state, models_dir / f"best_model_fold{fold}.pth")
        fold_scores[str(fold)] = float(best_score)

        if args.create_oof:
            model.eval()
            preds = []
            with torch.no_grad():
                for x, _ in val_dl:
                    preds.append(model(x).cpu().numpy())
            oof[val_i] = np.concatenate(preds, axis=0)

    mean_cv = float(np.mean(list(fold_scores.values())))
    result = {"mean_cv": mean_cv, "metric": metric, "fold_scores": fold_scores,
              "exp": cfg.get("exp_name", "EXP000")}
    if args.create_oof:
        oof_df = pd.DataFrame(oof, columns=target_cols)
        oof_df.insert(0, cfg.get("id_column", "id"), train[cfg.get("id_column", "id")].values)
        oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
        result["oof_score"] = float(mean_cv)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
