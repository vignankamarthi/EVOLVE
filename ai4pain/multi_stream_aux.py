"""Multi-stream classifier with an HRV auxiliary side-input.

Family: `multi_stream_aux`. A `MultiStreamNet` body over the raw 4-channel
sequence, fused with a small MLP over the 26-dim HRV feature vector
(ai4pain.hrv.compute_per_trial_features). The `add_aux_stream` architectural
mutation produces children of this family.

Rationale: the deep multi_stream families ingest only the raw signal. HRV
features (RMSSD, SDNN, LF/HF, ...) are the canonical autonomic-state summary
and carry class signal the raw BVP buries. This family gives the network both.

`forward(x, hrv)` takes the sequence tensor AND the precomputed HRV vector.
Because `run_pytorch_model` only passes one tensor, this family has its own
training function `train_multi_stream_aux` mirroring that pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

from ai4pain.data import load_split
from ai4pain.metrics import full_metric_suite
from ai4pain.baselines import (pad_trials_to_max, per_channel_zscore,
                                _device, _atomic_write_json)
from ai4pain.hrv import compute_per_trial_features, HRV_FEATURE_DIM
from ai4pain.multi_stream import MultiStreamNet


class MultiStreamAuxNet(nn.Module):
    """MultiStreamNet body + HRV-feature side-MLP, late-fused before FC."""

    def __init__(self, in_channels: int = 4, per_channel_hidden: int = 32,
                 per_channel_layers: int = 1,
                 encoder_type: str = "gru", fusion: str = "late_concat",
                 fusion_dropout: float = 0.2,
                 hrv_dim: int = HRV_FEATURE_DIM, hrv_hidden: int = 32,
                 num_classes: int = 3):
        super().__init__()
        self.body = MultiStreamNet(
            in_channels=in_channels, per_channel_hidden=per_channel_hidden,
            per_channel_layers=per_channel_layers, encoder_type=encoder_type,
            fusion=fusion, fusion_dropout=fusion_dropout,
            num_classes=num_classes)
        body_dim = self.body.fc.in_features
        self.hrv_mlp = nn.Sequential(
            nn.Linear(hrv_dim, hrv_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
        )
        self.dropout = nn.Dropout(fusion_dropout)
        self.fc = nn.Linear(body_dim + hrv_hidden, num_classes)

    def forward(self, x: torch.Tensor, hrv: torch.Tensor) -> torch.Tensor:
        body_emb = self.body.embed(x)        # (B, body_dim)
        hrv_emb = self.hrv_mlp(hrv)          # (B, hrv_hidden)
        fused = torch.cat([body_emb, hrv_emb], dim=1)
        return self.fc(self.dropout(fused))


def _hrv_matrix(trials: list[np.ndarray], fs: int) -> np.ndarray:
    return np.stack([compute_per_trial_features(t, fs=fs)
                     for t in trials]).astype(np.float32)


def train_multi_stream_aux(spec: dict, data_root: Path, out_dir: Path) -> dict:
    """End-to-end train of the multi_stream_aux family."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = spec.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    signals = tuple(spec.get("data", {}).get(
        "signals", ["Bvp", "Eda", "Resp", "SpO2"]))
    fe = spec.get("feature_extraction", {}) or {}
    fs = int(fe.get("fs", 100))

    print(f"[multi_stream_aux] loading train from {data_root}", flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    print(f"[multi_stream_aux] {len(X_train)} train / {len(X_val)} val trials",
          flush=True)

    # HRV features on the RAW (pre-padding) trials.
    print("[multi_stream_aux] computing HRV features...", flush=True)
    Htr = _hrv_matrix(X_train, fs)
    Hv = _hrv_matrix(X_val, fs)
    hmu = Htr.mean(axis=0, keepdims=True)
    hsig = Htr.std(axis=0, keepdims=True)
    hsig[hsig < 1e-6] = 1.0
    Htr = ((Htr - hmu) / hsig).astype(np.float32)
    Hv = ((Hv - hmu) / hsig).astype(np.float32)

    # Sequence tensors: pad + per-channel zscore (fit on train).
    Xtr = pad_trials_to_max(X_train)
    Xv = pad_trials_to_max(X_val)
    T_max = max(Xtr.shape[1], Xv.shape[1])
    if Xtr.shape[1] < T_max:
        pad = np.zeros((Xtr.shape[0], T_max - Xtr.shape[1], Xtr.shape[2]),
                       dtype=np.float32)
        Xtr = np.concatenate([Xtr, pad], axis=1)
    if Xv.shape[1] < T_max:
        pad = np.zeros((Xv.shape[0], T_max - Xv.shape[1], Xv.shape[2]),
                       dtype=np.float32)
        Xv = np.concatenate([Xv, pad], axis=1)
    Xtr, Xv, _, _ = per_channel_zscore(Xtr, Xv)

    device = _device()
    print(f"[multi_stream_aux] device: {device}", flush=True)
    model_cfg = spec.get("model", {})
    model = MultiStreamAuxNet(
        in_channels=len(signals),
        per_channel_hidden=int(model_cfg.get("per_channel_hidden", 32)),
        per_channel_layers=int(model_cfg.get("per_channel_layers", 1)),
        encoder_type=model_cfg.get("encoder_type", "gru"),
        fusion=model_cfg.get("fusion", "late_concat"),
        fusion_dropout=float(model_cfg.get("fusion_dropout", 0.2)),
        hrv_dim=HRV_FEATURE_DIM,
        hrv_hidden=int(model_cfg.get("hrv_hidden", 32)),
        num_classes=3,
    ).to(device)

    epochs = int(train_cfg.get("epochs", 20))
    bs = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    optim_name = train_cfg.get("optimizer", "adam").lower()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Htr_t = torch.from_numpy(Htr).to(device)
    ytr_t = torch.from_numpy(y_train).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    Hv_t = torch.from_numpy(Hv).to(device)

    counts = np.bincount(y_train, minlength=3)
    base_weights = (counts.sum() / (3 * counts)).astype(np.float32)
    hp_boost = float(train_cfg.get("hp_boost", 1.0))
    if hp_boost != 1.0:
        base_weights = base_weights.copy()
        base_weights[2] *= hp_boost
    class_weights = torch.tensor(base_weights, device=device)

    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    if focal_gamma > 0.0:
        ce_per = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        def loss_fn(logits, y):
            ce = ce_per(logits, y)
            p_correct = torch.softmax(logits, dim=1).gather(
                1, y.unsqueeze(1)).squeeze(1)
            return ((1.0 - p_correct) ** focal_gamma * ce).mean()
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    if optim_name == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(Xtr_t, Htr_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

    history: list[dict] = []
    best_val_bal_acc = -math.inf
    best_metrics: dict = {}
    final_metrics: dict = {}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, hb, yb in train_loader:
            optim.zero_grad()
            loss = loss_fn(model(xb, hb), yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        model.eval()
        with torch.no_grad():
            train_preds = model(Xtr_t, Htr_t).argmax(dim=1).cpu().numpy()
            train_bal = float(balanced_accuracy_score(y_train, train_preds))
            val_logits = model(Xv_t, Hv_t)
            val_proba = val_logits.softmax(dim=1).cpu().numpy()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            final_metrics = full_metric_suite(y_val, val_preds, val_proba)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_bal_acc": train_bal,
            "val_bal_acc": final_metrics["balanced_acc"],
            "val_macro_f1": final_metrics["macro_f1"],
        })
        if final_metrics["balanced_acc"] > best_val_bal_acc:
            best_val_bal_acc = final_metrics["balanced_acc"]
            best_metrics = final_metrics
        print(f"[multi_stream_aux] ep {epoch}: "
              f"train_loss={epoch_loss/max(n_batches,1):.4f} "
              f"train_bal={train_bal:.3f} "
              f"val_bal={final_metrics['balanced_acc']:.3f} "
              f"val_f1={final_metrics['macro_f1']:.3f}", flush=True)

    train_seconds = time.time() - t0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        _ = model(Xv_t, Hv_t)
    inference_seconds = time.time() - t1

    result = {
        "name": spec.get("name", "multi_stream_aux"),
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "history": history,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "device": str(device),
        "spec": spec,
    }
    _atomic_write_json(out_dir / "result.json", result)
    print(f"[multi_stream_aux] best val bal_acc: {best_val_bal_acc:.3f}",
          flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    from ai4pain.multiseed import run_multiseed
    return run_multiseed(train_multi_stream_aux, spec, Path(data_root), run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
