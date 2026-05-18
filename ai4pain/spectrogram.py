"""Spectrogram + 2D-CNN classifier (family: `spectrogram_cnn2d`).

Pipeline:
  Raw trial (T, C) -> per-channel STFT spectrogram via scipy.signal.spectrogram
                    -> stack to (C, F, T') float32 tensor
                    -> optional log-scale (log(eps + |Sxx|))
                    -> per-channel z-score (fit on train)
                    -> pad along time axis to global max T'
                    -> small 2D CNN (conv -> bn -> relu -> maxpool) x depth
                    -> global avg pool + linear -> logits

Why: First 2D pipeline in the population. Sriram Kumar et al. 2024 reported
86% multimodal emotion classification with CWT + VGG16; we use STFT
(cheaper, no extra dep) with a small from-scratch CNN.
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
from scipy.signal import spectrogram as _scipy_spectrogram

from ai4pain.data import load_split
from ai4pain.metrics import full_metric_suite
from sklearn.metrics import balanced_accuracy_score


def compute_spectrogram_stack(trial: np.ndarray, fs: int = 100,
                                nperseg: int = 64, noverlap: int = 32,
                                log_scale: bool = True) -> np.ndarray:
    """Compute per-channel STFT spectrograms and stack to (C, F, T').

    Args:
        trial: (T, C) float array.
        fs: sampling rate (Hz).
        nperseg: STFT window length.
        noverlap: STFT overlap (samples).
        log_scale: if True, return log(eps + Sxx).

    Returns:
        float32 ndarray of shape (C, F, T').
    """
    trial = np.asarray(trial, dtype=np.float32)
    if trial.ndim != 2:
        raise ValueError(f"expected (T, C) trial, got shape {trial.shape}")
    T, C = trial.shape
    nperseg = min(nperseg, T)
    if nperseg < 2:
        nperseg = 2
    noverlap = min(noverlap, nperseg - 1)
    per_channel = []
    for c in range(C):
        _, _, Sxx = _scipy_spectrogram(trial[:, c], fs=fs,
                                          nperseg=nperseg, noverlap=noverlap,
                                          mode="magnitude")
        if log_scale:
            Sxx = np.log1p(Sxx)
        per_channel.append(Sxx.astype(np.float32))
    return np.stack(per_channel, axis=0)


def pad_spectrograms_to_max(specs: list[np.ndarray]) -> np.ndarray:
    """Pad along the time axis to the global max T'. Output (N, C, F, T'_max)."""
    if not specs:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    C = specs[0].shape[0]
    F = specs[0].shape[1]
    T_max = max(s.shape[2] for s in specs)
    out = np.zeros((len(specs), C, F, T_max), dtype=np.float32)
    for i, s in enumerate(specs):
        out[i, :, :, :s.shape[2]] = s
    return out


class _SpecResBlock(nn.Module):
    """2D residual block: conv-bn-relu-conv-bn + projected skip, then relu.
    A 1x1 projection handles the channel change so the skip stays valid."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.proj = (nn.Conv2d(c_in, c_out, kernel_size=1)
                     if c_in != c_out else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class SpectrogramCNN2D(nn.Module):
    """Small 2D CNN over (C, F, T) spectrogram stacks.

    `use_residual=True` swaps each plain conv block for a `_SpecResBlock`
    (ResNet-style skip). Residual connections counter the depth-degradation
    seen in iter_0017 (depth=5 regressed vs depth=3).
    """

    def __init__(self, in_channels: int = 4, F: int = 33,
                 base_channels: int = 16, depth: int = 2,
                 dropout: float = 0.2, use_residual: bool = False,
                 num_classes: int = 3):
        super().__init__()
        self.use_residual = use_residual
        layers = []
        c_in = in_channels
        c_out = base_channels
        for d in range(depth):
            if use_residual:
                layers.append(_SpecResBlock(c_in, c_out))
            else:
                layers += [
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2,
                                        ceil_mode=True))
            c_in = c_out
            c_out = min(c_out * 2, 128)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def train_spectrogram(spec: dict, data_root: Path, out_dir: Path) -> dict:
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
    nperseg = int(fe.get("nperseg", 64))
    noverlap = int(fe.get("noverlap", 32))
    log_scale = bool(fe.get("log_scale", True))

    print(f"[spec_cnn2d] loading train from {data_root}", flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    print(f"[spec_cnn2d] {len(X_train)} train / {len(X_val)} val trials",
          flush=True)

    print(f"[spec_cnn2d] STFT (nperseg={nperseg}, noverlap={noverlap}, "
          f"log_scale={log_scale})...", flush=True)
    Str = [compute_spectrogram_stack(x, fs=fs, nperseg=nperseg,
                                       noverlap=noverlap, log_scale=log_scale)
           for x in X_train]
    Sv = [compute_spectrogram_stack(x, fs=fs, nperseg=nperseg,
                                       noverlap=noverlap, log_scale=log_scale)
          for x in X_val]
    Str_p = pad_spectrograms_to_max(Str)
    Sv_p = pad_spectrograms_to_max(Sv)
    # Align time dims (use larger of the two)
    T_max = max(Str_p.shape[-1], Sv_p.shape[-1])
    if Str_p.shape[-1] < T_max:
        pad = np.zeros((Str_p.shape[0], Str_p.shape[1], Str_p.shape[2],
                         T_max - Str_p.shape[-1]), dtype=np.float32)
        Str_p = np.concatenate([Str_p, pad], axis=-1)
    if Sv_p.shape[-1] < T_max:
        pad = np.zeros((Sv_p.shape[0], Sv_p.shape[1], Sv_p.shape[2],
                         T_max - Sv_p.shape[-1]), dtype=np.float32)
        Sv_p = np.concatenate([Sv_p, pad], axis=-1)

    # Per-channel z-score over (F, T') flattened, fit on train
    mu = Str_p.reshape(Str_p.shape[0], Str_p.shape[1], -1).mean(
        axis=(0, 2), keepdims=True)
    sigma = Str_p.reshape(Str_p.shape[0], Str_p.shape[1], -1).std(
        axis=(0, 2), keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    mu = mu[..., np.newaxis]  # (1, C, 1, 1)
    sigma = sigma[..., np.newaxis]
    Str_n = ((Str_p - mu) / sigma).astype(np.float32)
    Sv_n = ((Sv_p - mu) / sigma).astype(np.float32)

    F = Str_n.shape[2]

    device = _device()
    print(f"[spec_cnn2d] device: {device}, shape (B,C,F,T)={Str_n.shape}",
          flush=True)
    model_cfg = spec.get("model", {})
    model = SpectrogramCNN2D(
        in_channels=len(signals), F=F,
        base_channels=int(model_cfg.get("base_channels", 16)),
        depth=int(model_cfg.get("depth", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        use_residual=bool(model_cfg.get("use_residual", False)),
        num_classes=3,
    ).to(device)

    epochs = int(train_cfg.get("epochs", 20))
    bs = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    optim_name = train_cfg.get("optimizer", "adam").lower()

    Str_t = torch.from_numpy(Str_n).to(device)
    ytr_t = torch.from_numpy(y_train).to(device)
    Sv_t = torch.from_numpy(Sv_n).to(device)

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

    train_ds = TensorDataset(Str_t, ytr_t)
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
        for xb, yb in train_loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        model.eval()
        with torch.no_grad():
            train_preds = model(Str_t).argmax(dim=1).cpu().numpy()
            train_bal = float(balanced_accuracy_score(y_train, train_preds))
            val_logits = model(Sv_t)
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
        print(f"[spec_cnn2d] ep {epoch}: "
              f"train_loss={epoch_loss/max(n_batches,1):.4f} "
              f"train_bal={train_bal:.3f} "
              f"val_bal={final_metrics['balanced_acc']:.3f} "
              f"val_f1={final_metrics['macro_f1']:.3f}", flush=True)

    train_seconds = time.time() - t0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        _ = model(Sv_t)
    inference_seconds = time.time() - t1

    result = {
        "name": spec.get("name", "spec_cnn2d"),
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
    print(f"[spec_cnn2d] best val bal_acc: {best_val_bal_acc:.3f}", flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    from ai4pain.multiseed import run_multiseed
    return run_multiseed(train_spectrogram, spec, Path(data_root), run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
