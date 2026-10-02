"""Local prototype: division proposal model from raw image volumes.

The model is independent of the upper tracking pipeline.
It takes two consecutive 3D frames (t, t+1) as input and predicts:
- division probability at the window center
- parent / daughter1 / daughter2 offsets relative to window center

We train on the 16-sequence synthetic subset.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / "tmp_synth" / "sequences"
OUT_DIR = ROOT / "EXP" / "EXP008" / "outputs"
MODEL_DIR = ROOT / "EXP" / "EXP008" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 16
HALF = WINDOW // 2
TRAIN_SEQS = list(range(12))
VAL_SEQS = list(range(12, 16))
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequence(path: Path):
    data = np.load(path)
    return data["volumes"], data["nodes"], data["edges"], data["divisions"]


def extract_window(volumes, t, center, half=HALF):
    """Extract 2-frame window centered at (z,y,x). Returns (2, W, W, W) float32."""
    T, Z, Y, X = volumes.shape
    t = int(t)
    z, y, x = [int(round(v)) for v in center]
    z0, z1 = max(0, z - half), min(Z, z + half)
    y0, y1 = max(0, y - half), min(Y, y + half)
    x0, x1 = max(0, x - half), min(X, x + half)
    win = np.zeros((2, WINDOW, WINDOW, WINDOW), dtype=np.float32)
    for ti, tt in enumerate([t, t + 1]):
        if tt < 0 or tt >= T:
            continue
        patch = volumes[tt, z0:z1, y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            continue
        lo = float(np.percentile(patch, 1))
        hi = float(np.percentile(patch, 99))
        if hi > lo:
            patch = (patch - lo) / (hi - lo)
        win[ti, :patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
    return win


def normalize_offset(coord, center, half=HALF):
    """Return offset in [-1, 1] relative to window half-size."""
    return [(coord[i] - center[i]) / half for i in range(3)]


def build_samples(seq_paths, max_neg_per_seq=1000, seed=SEED):
    rng = random.Random(seed)
    samples = []
    for seq_path in seq_paths:
        volumes, nodes, edges, divisions = load_sequence(seq_path)
        T = volumes.shape[0]
        out_edges = {}
        for s, t in edges:
            out_edges.setdefault(int(s), []).append(int(t))
        division_set = set(int(d) for d in divisions)

        # positives
        for p_idx in divisions:
            p_idx = int(p_idx)
            p = nodes[p_idx]
            t = int(p[0])
            children = out_edges.get(p_idx, [])
            if len(children) < 2:
                continue
            d1, d2 = nodes[int(children[0])], nodes[int(children[1])]
            if int(d1[0]) != t + 1 or int(d2[0]) != t + 1:
                continue
            center = [float(p[1]), float(p[2]), float(p[3])]
            win = extract_window(volumes, t, center)
            off_p = normalize_offset([float(p[1]), float(p[2]), float(p[3])], center)
            off_d1 = normalize_offset([float(d1[1]), float(d1[2]), float(d1[3])], center)
            off_d2 = normalize_offset([float(d2[1]), float(d2[2]), float(d2[3])], center)
            targets = np.array(off_p + off_d1 + off_d2, dtype=np.float32)
            samples.append((win, 1, targets))

        # negatives
        neg_count = 0
        attempts = 0
        while neg_count < max_neg_per_seq and attempts < max_neg_per_seq * 10:
            attempts += 1
            t = rng.randint(0, T - 2)
            z = rng.uniform(0, volumes.shape[1] - 1)
            y = rng.uniform(0, volumes.shape[2] - 1)
            x = rng.uniform(0, volumes.shape[3] - 1)
            center = [z, y, x]
            # skip if too close to a division parent
            close = False
            for p_idx in division_set:
                pp = nodes[p_idx]
                if int(pp[0]) == t:
                    dist = np.linalg.norm(np.array([pp[1], pp[2], pp[3]]) - np.array(center))
                    if dist < WINDOW * 0.5:
                        close = True
                        break
            if close:
                continue
            win = extract_window(volumes, t, center)
            targets = np.zeros(9, dtype=np.float32)
            samples.append((win, 0, targets))
            neg_count += 1
    return samples


class DivisionProposalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.cls = nn.Linear(32, 1)
        self.reg = nn.Linear(32, 9)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        return self.cls(feat).squeeze(-1), self.reg(feat)


def collate(batch):
    x = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    off = torch.from_numpy(np.stack([b[2] for b in batch])).float()
    return x, y, off


def main() -> None:
    set_seed()
    seq_paths = sorted(SEQ_DIR.glob("seq_*.npz"))
    if len(seq_paths) < 16:
        raise SystemExit(f"Expected 16 sequences, found {len(seq_paths)}")
    train_paths = [seq_paths[i] for i in TRAIN_SEQS]
    val_paths = [seq_paths[i] for i in VAL_SEQS]

    print("Building train samples ...")
    train_samples = build_samples(train_paths)
    print("Building val samples ...")
    val_samples = build_samples(val_paths)
    n_pos_train = sum(1 for s in train_samples if s[1] == 1)
    n_neg_train = len(train_samples) - n_pos_train
    n_pos_val = sum(1 for s in val_samples if s[1] == 1)
    n_neg_val = len(val_samples) - n_pos_val
    print(f"Train: {len(train_samples)} ({n_pos_train} pos / {n_neg_train} neg)")
    print(f"Val:   {len(val_samples)} ({n_pos_val} pos / {n_neg_val} neg)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = DivisionProposalNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()

    train_loader = torch.utils.data.DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_samples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    best_auc = 0.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y, off in train_loader:
            x, y, off = x.to(device), y.to(device), off.to(device)
            optimizer.zero_grad()
            logits, pred_off = model(x)
            loss_cls = bce(logits, y)
            pos = y > 0.5
            if pos.any():
                loss_reg = smooth_l1(pred_off[pos], off[pos])
            else:
                loss_reg = torch.tensor(0.0, device=device)
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(train_samples)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y, off in val_loader:
                x = x.to(device)
                logits, _ = model(x)
                preds.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y.numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        if len(np.unique(labels)) == 2:
            auc = roc_auc_score(labels, preds)
        else:
            auc = float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": auc})
        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | val_auc={auc:.4f}")
        if auc == auc and auc > best_auc:
            best_auc = float(auc)
            torch.save(model.state_dict(), MODEL_DIR / "division_proposal_synth.pth")

    report = {
        "status": "ok",
        "n_train": len(train_samples),
        "n_train_pos": n_pos_train,
        "n_train_neg": n_neg_train,
        "n_val": len(val_samples),
        "n_val_pos": n_pos_val,
        "n_val_neg": n_neg_val,
        "best_val_auc": best_auc,
        "history": history[-10:],
        "model_path": str(MODEL_DIR / "division_proposal_synth.pth"),
    }
    (OUT_DIR / "synth_division_proposal_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
