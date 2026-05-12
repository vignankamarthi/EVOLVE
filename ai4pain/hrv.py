"""HRV-features-based classifier (family: `hrv_features_mlp`).

Pipeline:
  Raw trial (T, 4) -> BVP peak detection (scipy.signal.find_peaks) on channel 0
                   -> RR intervals -> HRV time + frequency features
                   -> auxiliary stats on EDA / RESP / SpO2 (mean, std, min, max,
                      range, energy)
                   -> concatenate to fixed-dim float32 vector (HRV_FEATURE_DIM)
                   -> per-feature z-score (fit on train) -> small MLP -> logits

Why: HRV features (RMSSD, SDNN, pNN50, LF, HF, LF/HF) are the canonical
autonomic-state representation. Xia et al. 2024 hit 98%+ on stress using
HRV features + CNN-LSTM-Transformer; raw-signal models plateaued lower.
Pain is sympathetic-mediated, so HRV is high-signal for this task.

Fixed-dim feature vector (HRV_FEATURE_DIM = 8 HRV + 6 stats * 3 aux channels
= 8 + 18 = 26):
  HRV (from BVP, channel 0):
    [rmssd, sdnn, pnn50, mean_hr, std_hr, lf_power, hf_power, lf_hf_ratio]
  Aux stats per (EDA, RESP, SpO2):
    [mean, std, min, max, range, energy]
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
from scipy.signal import find_peaks, welch

from ai4pain.data import load_split
from ai4pain.metrics import full_metric_suite
from sklearn.metrics import balanced_accuracy_score


HRV_FEATURE_DIM = 8 + 6 * 3  # 8 HRV + 6 aux stats * 3 aux channels = 26


def compute_hrv_features(bvp: np.ndarray, fs: int = 100) -> dict:
    """HRV features from a BVP signal.

    Args:
        bvp: 1D float array of one BVP trial.
        fs: sampling rate (Hz). Default 100.

    Returns: dict with keys rmssd, sdnn, pnn50, mean_hr, std_hr, lf_power,
        hf_power, lf_hf_ratio, n_peaks. All values are floats. When no peaks
        are detectable, all values are 0.0 (and n_peaks = 0).
    """
    bvp = np.asarray(bvp, dtype=np.float64).ravel()
    if bvp.size < fs:  # less than 1 second
        return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0,
                "mean_hr": 0.0, "std_hr": 0.0,
                "lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0,
                "n_peaks": 0}
    # Peak detection. min distance ~ 0.4s (max ~150 BPM)
    min_dist = max(1, int(fs * 0.4))
    peaks, _ = find_peaks(bvp, distance=min_dist)
    if peaks.size < 3:
        return {"rmssd": 0.0, "sdnn": 0.0, "pnn50": 0.0,
                "mean_hr": 0.0, "std_hr": 0.0,
                "lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0,
                "n_peaks": int(peaks.size)}

    rr_intervals = np.diff(peaks) / float(fs)  # seconds
    rr_diff = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(rr_diff ** 2))) if rr_diff.size else 0.0
    sdnn = float(np.std(rr_intervals, ddof=1)) if rr_intervals.size > 1 else 0.0
    pnn50 = float(np.mean(np.abs(rr_diff) > 0.05)) if rr_diff.size else 0.0
    hr = 60.0 / rr_intervals
    mean_hr = float(np.mean(hr))
    std_hr = float(np.std(hr, ddof=1)) if hr.size > 1 else 0.0

    # Frequency-domain HRV: interpolate RR series to uniform 4 Hz, Welch PSD.
    lf_power = 0.0
    hf_power = 0.0
    lf_hf_ratio = 0.0
    if rr_intervals.size >= 4:
        # peaks[1:] gives time of each RR interval (in samples / fs = seconds)
        t_rr = peaks[1:] / float(fs)
        t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / 4.0)  # 4 Hz resample
        if t_uniform.size >= 8:
            rr_uniform = np.interp(t_uniform, t_rr, rr_intervals)
            nperseg = min(64, len(rr_uniform))
            freqs, psd = welch(rr_uniform, fs=4.0, nperseg=nperseg)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask]))
            hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask]))
            if hf_power > 1e-12:
                lf_hf_ratio = lf_power / hf_power

    return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50,
            "mean_hr": mean_hr, "std_hr": std_hr,
            "lf_power": lf_power, "hf_power": hf_power,
            "lf_hf_ratio": lf_hf_ratio, "n_peaks": int(peaks.size)}


def _aux_channel_stats(channel: np.ndarray) -> np.ndarray:
    """6 stats from one auxiliary channel: mean, std, min, max, range, energy."""
    c = np.asarray(channel, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros(6, dtype=np.float32)
    mean = float(np.mean(c))
    std = float(np.std(c, ddof=1)) if c.size > 1 else 0.0
    cmin = float(np.min(c))
    cmax = float(np.max(c))
    rng = cmax - cmin
    energy = float(np.mean(c ** 2))
    return np.array([mean, std, cmin, cmax, rng, energy], dtype=np.float32)


def compute_per_trial_features(trial: np.ndarray, fs: int = 100) -> np.ndarray:
    """Compute the fixed-dim per-trial feature vector.

    Args:
        trial: (T, 4) float array. Channel order: [BVP, EDA, RESP, SpO2].
        fs: sampling rate Hz.

    Returns: float32 ndarray of shape (HRV_FEATURE_DIM,).
    """
    trial = np.asarray(trial, dtype=np.float32)
    if trial.ndim != 2 or trial.shape[1] < 4:
        raise ValueError(f"expected (T, 4) trial, got shape {trial.shape}")
    bvp = trial[:, 0]
    hrv_feats = compute_hrv_features(bvp, fs=fs)
    hrv_vec = np.array([
        hrv_feats["rmssd"], hrv_feats["sdnn"], hrv_feats["pnn50"],
        hrv_feats["mean_hr"], hrv_feats["std_hr"],
        hrv_feats["lf_power"], hrv_feats["hf_power"],
        hrv_feats["lf_hf_ratio"],
    ], dtype=np.float32)
    aux = np.concatenate([
        _aux_channel_stats(trial[:, 1]),  # EDA
        _aux_channel_stats(trial[:, 2]),  # RESP
        _aux_channel_stats(trial[:, 3]),  # SpO2
    ])
    out = np.concatenate([hrv_vec, aux]).astype(np.float32)
    # Clean up NaN/inf from any pathological trial (zero-padded all-zero
    # signals can give std=0 -> std_hr nan after a divide). Replace with 0.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


class HRVFeaturesMLP(nn.Module):
    """Fixed-dim feature vector -> 2-layer MLP -> logits."""

    def __init__(self, n_features: int = HRV_FEATURE_DIM,
                 hidden: int = 64, dropout: float = 0.2,
                 num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def _featurize_split(X: list[np.ndarray], fs: int) -> np.ndarray:
    feats = np.stack([compute_per_trial_features(x, fs=fs) for x in X])
    return feats.astype(np.float32)


def train_hrv(spec: dict, data_root: Path, out_dir: Path) -> dict:
    """End-to-end: load -> HRV-featurize -> MLP train -> result.json."""
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

    print(f"[hrv_mlp] loading train from {data_root}", flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    print(f"[hrv_mlp] {len(X_train)} train / {len(X_val)} val trials",
          flush=True)

    print(f"[hrv_mlp] featurizing (HRV+aux, fs={fs})...", flush=True)
    Ftr = _featurize_split(X_train, fs=fs)
    Fv = _featurize_split(X_val, fs=fs)

    # Per-feature z-score, fit on train only.
    mu = Ftr.mean(axis=0, keepdims=True)
    sigma = Ftr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Ftr_n = ((Ftr - mu) / sigma).astype(np.float32)
    Fv_n = ((Fv - mu) / sigma).astype(np.float32)

    device = _device()
    print(f"[hrv_mlp] device: {device}", flush=True)
    model_cfg = spec.get("model", {})
    model = HRVFeaturesMLP(
        n_features=HRV_FEATURE_DIM,
        hidden=int(model_cfg.get("hidden", 64)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        num_classes=3,
    ).to(device)

    epochs = int(train_cfg.get("epochs", 20))
    bs = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    optim_name = train_cfg.get("optimizer", "adam").lower()

    Ftr_t = torch.from_numpy(Ftr_n).to(device)
    ytr_t = torch.from_numpy(y_train).to(device)
    Fv_t = torch.from_numpy(Fv_n).to(device)

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

    train_ds = TensorDataset(Ftr_t, ytr_t)
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
            train_preds = model(Ftr_t).argmax(dim=1).cpu().numpy()
            train_bal = float(balanced_accuracy_score(y_train, train_preds))
            val_logits = model(Fv_t)
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
        print(f"[hrv_mlp] ep {epoch}: "
              f"train_loss={epoch_loss/max(n_batches,1):.4f} "
              f"train_bal={train_bal:.3f} "
              f"val_bal={final_metrics['balanced_acc']:.3f} "
              f"val_f1={final_metrics['macro_f1']:.3f}", flush=True)

    train_seconds = time.time() - t0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        _ = model(Fv_t)
    inference_seconds = time.time() - t1

    result = {
        "name": spec.get("name", "hrv_mlp"),
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
    print(f"[hrv_mlp] best val bal_acc: {best_val_bal_acc:.3f}", flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    return train_hrv(spec, data_root=Path(data_root), out_dir=run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
