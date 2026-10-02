"""Local training for a synthetic-data division verifier.

This is the first local prototype:
- Uses the 16-sequence community subset (tmp_synth/sequences).
- Builds positive/negative division samples from synthetic lineage graphs.
- Trains a small 3D patch encoder + MLP classifier.
- Reports held-out AUC on synthetic sequences.

Run:
    python EXP/EXP006/train_synth_division.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / "tmp_synth" / "sequences"
OUT_DIR = ROOT / "EXP" / "EXP006" / "outputs"
MODEL_DIR = ROOT / "EXP" / "EXP006" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VOXEL_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
PATCH_R = 4
TRAIN_SEQS = list(range(12))
VAL_SEQS = list(range(12, 16))
NEG_PER_SEQ_CAP = 4000
EPOCHS = 30
BATCH_SIZE = 64
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


def extract_patch(vol: np.ndarray, t: int, z: float, y: float, x: float, r: int = PATCH_R) -> np.ndarray:
    """Extract a cubic patch, clipping at borders. Returns float32 normalized roughly [0,1]."""
    T, Z, Y, X = vol.shape
    t = int(round(t))
    z = int(round(z))
    y = int(round(y))
    x = int(round(x))
    if t < 0 or t >= T:
        return np.zeros((1, 2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=np.float32)
    z0, z1 = max(0, z - r), min(Z, z + r + 1)
    y0, y1 = max(0, y - r), min(Y, y + r + 1)
    x0, x1 = max(0, x - r), min(X, x + r + 1)
    patch = vol[t, z0:z1, y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return np.zeros((1, 2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=np.float32)
    # normalize by global q99-ish per patch to be robust
    lo = float(np.percentile(patch, 1))
    hi = float(np.percentile(patch, 99))
    if hi > lo:
        patch = (patch - lo) / (hi - lo)
    else:
        patch = np.zeros_like(patch)
    # pad to fixed size
    padded = np.zeros((2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=np.float32)
    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
    return padded[None, ...]


def physical_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = (a[:3] - b[:3]) * VOXEL_UM
    return float(np.linalg.norm(d))


def angle_cos(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    va = (a[:3] - p[:3]) * VOXEL_UM
    vb = (b[:3] - p[:3]) * VOXEL_UM
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def build_samples(seq_paths: list[Path], max_neg_per_seq: int = NEG_PER_SEQ_CAP, seed: int = SEED):
    rng = random.Random(seed)
    samples = []  # (patches, feat, label)
    for seq_path in seq_paths:
        volumes, nodes, edges, divisions = load_sequence(seq_path)
        T = volumes.shape[0]
        out_edges: dict[int, list[int]] = {}
        for s, t in edges:
            s = int(s)
            t = int(t)
            out_edges.setdefault(s, []).append(t)
        division_set = set(int(d) for d in divisions)
        next_frame_nodes: dict[int, list[int]] = {}
        for i, row in enumerate(nodes):
            ti = int(row[0])
            next_frame_nodes.setdefault(ti, []).append(i)

        # positives
        for p_idx in divisions:
            p_idx = int(p_idx)
            p = nodes[p_idx]
            t = int(p[0])
            children = out_edges.get(p_idx, [])
            if len(children) < 2:
                continue
            d1_idx, d2_idx = int(children[0]), int(children[1])
            d1, d2 = nodes[d1_idx], nodes[d2_idx]
            if int(d1[0]) != t + 1 or int(d2[0]) != t + 1:
                continue
            p_patch = extract_patch(volumes, t, p[1], p[2], p[3])
            d1_patch = extract_patch(volumes, t + 1, d1[1], d1[2], d1[3])
            d2_patch = extract_patch(volumes, t + 1, d2[1], d2[2], d2[3])
            feat = np.array([
                physical_dist(p, d1),
                physical_dist(p, d2),
                physical_dist(d1, d2),
                angle_cos(p, d1, d2),
                float(np.mean(p_patch)),
                float(np.mean(d1_patch) + np.mean(d2_patch)) / 2.0,
            ], dtype=np.float32)
            samples.append((p_patch, d1_patch, d2_patch, feat, 1))

        # negatives
        neg_count = 0
        all_indices = list(range(len(nodes)))
        rng.shuffle(all_indices)
        for p_idx in all_indices:
            if neg_count >= max_neg_per_seq:
                break
            p_idx = int(p_idx)
            if p_idx in division_set:
                continue
            p = nodes[p_idx]
            t = int(p[0])
            if t >= T - 1:
                continue
            children = out_edges.get(p_idx, [])
            if len(children) >= 2:
                continue
            next_ids = [i for i in next_frame_nodes.get(t + 1, []) if i != p_idx]
            if len(next_ids) < 2:
                continue

            if len(children) == 1:
                d1_idx = int(children[0])
                d1 = nodes[d1_idx]
                # hard negative: choose the nearest other cell to d1 as false second daughter
                cands = [i for i in next_ids if i != d1_idx]
                if not cands:
                    continue
                d2_idx = min(cands, key=lambda i: physical_dist(d1, nodes[i]))
                d2 = nodes[d2_idx]
            else:
                # parent with no children: choose two nearest cells in next frame as false daughters
                cands = sorted(next_ids, key=lambda i: physical_dist(p, nodes[i]))
                if len(cands) < 2:
                    continue
                d1_idx, d2_idx = cands[0], cands[1]
                d1, d2 = nodes[d1_idx], nodes[d2_idx]

            p_patch = extract_patch(volumes, t, p[1], p[2], p[3])
            d1_patch = extract_patch(volumes, t + 1, d1[1], d1[2], d1[3])
            d2_patch = extract_patch(volumes, t + 1, d2[1], d2[2], d2[3])
            feat = np.array([
                physical_dist(p, d1),
                physical_dist(p, d2),
                physical_dist(d1, d2),
                angle_cos(p, d1, d2),
                float(np.mean(p_patch)),
                float(np.mean(d1_patch) + np.mean(d2_patch)) / 2.0,
            ], dtype=np.float32)
            samples.append((p_patch, d1_patch, d2_patch, feat, 0))
            neg_count += 1

    return samples


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class DivisionVerifier(nn.Module):
    def __init__(self, feat_dim: int = 6):
        super().__init__()
        self.encoder = PatchEncoder()
        self.head = nn.Sequential(
            nn.Linear(32 * 3 + feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, p1, p2, p3, feat):
        e1 = self.encoder(p1)
        e2 = self.encoder(p2)
        e3 = self.encoder(p3)
        x = torch.cat([e1, e2, e3, feat], dim=1)
        return self.head(x).squeeze(-1)


def collate(batch):
    p1 = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    p2 = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    p3 = torch.from_numpy(np.stack([b[2] for b in batch])).float()
    feat = torch.from_numpy(np.stack([b[3] for b in batch])).float()
    y = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return p1, p2, p3, feat, y


def main() -> None:
    set_seed()
    seq_paths = sorted(SEQ_DIR.glob("seq_*.npz"))
    if len(seq_paths) < 16:
        raise SystemExit(f"Expected 16 sequences, found {len(seq_paths)} in {SEQ_DIR}")
    print(f"Found {len(seq_paths)} synthetic sequences in {SEQ_DIR}")

    train_paths = [seq_paths[i] for i in TRAIN_SEQS]
    val_paths = [seq_paths[i] for i in VAL_SEQS]

    print("Building train samples ...")
    train_samples = build_samples(train_paths)
    print("Building val samples ...")
    val_samples = build_samples(val_paths)
    n_pos_train = sum(1 for s in train_samples if s[4] == 1)
    n_neg_train = len(train_samples) - n_pos_train
    n_pos_val = sum(1 for s in val_samples if s[4] == 1)
    n_neg_val = len(val_samples) - n_pos_val
    print(f"Train: {len(train_samples)} ({n_pos_train} pos / {n_neg_train} neg)")
    print(f"Val:   {len(val_samples)} ({n_pos_val} pos / {n_neg_val} neg)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = DivisionVerifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = torch.utils.data.DataLoader(
        train_samples, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_samples, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
    )

    best_auc = 0.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for p1, p2, p3, feat, y in train_loader:
            p1, p2, p3, feat, y = p1.to(device), p2.to(device), p3.to(device), feat.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(p1, p2, p3, feat)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(train_samples)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for p1, p2, p3, feat, y in val_loader:
                p1, p2, p3, feat = p1.to(device), p2.to(device), p3.to(device), feat.to(device)
                logits = model(p1, p2, p3, feat)
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
            torch.save(model.state_dict(), MODEL_DIR / "division_verifier_synth.pth")

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
        "model_path": str(MODEL_DIR / "division_verifier_synth.pth"),
    }
    (OUT_DIR / "synth_division_train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
