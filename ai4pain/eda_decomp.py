"""cvxEDA tonic+phasic decomposition + MLP classifier (family: `eda_decomp_mlp`).

Implements the Greco et al. 2016 cvxEDA convex-optimization decomposition
of the EDA channel into:
  - tonic: slow-varying skin conductance level (SCL); sympathetic baseline
  - phasic: sparse skin-conductance-response (SCR) bursts; event-driven sympathetic

Per-trial feature vector concatenates:
  - cvxEDA stats: tonic mean, tonic slope, phasic peak count, phasic max,
    phasic AUC, phasic sparsity (fraction zero), phasic mean
  - BVP HRV stats (RMSSD, SDNN, pnn50, mean_HR)
  - RESP stats (mean, std, energy)
  - SpO2 stats (mean, std, energy)

Total: 7 cvxEDA + 4 HRV + 3 RESP + 3 SpO2 = 17 features (EDA_FEATURE_DIM).

cvxEDA reference: Greco et al., "cvxEDA: A Convex Optimization Approach to
Electrodermal Activity Processing", IEEE TBME 2016.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import sparse
from scipy.signal import find_peaks

from ai4pain.data import load_split
from ai4pain.metrics import full_metric_suite
from ai4pain.hrv import compute_hrv_features
from sklearn.metrics import balanced_accuracy_score


EDA_FEATURE_DIM = 7 + 4 + 3 + 3  # 17


def cvx_eda_decompose(y: np.ndarray, fs: int = 100,
                       tau0: float = 2.0, tau1: float = 0.7,
                       delta_knot: float = 10.0,
                       alpha: float = 8e-4, gamma: float = 1e-2,
                       solver: str = "SCS") -> tuple[np.ndarray, np.ndarray]:
    """cvxEDA tonic+phasic decomposition.

    Args:
        y: 1D EDA signal (float).
        fs: sampling rate (Hz).
        tau0, tau1: bateman IRF time constants (rise/decay, seconds).
        delta_knot: knot interval for cubic B-spline tonic basis (seconds).
        alpha: L1 sparsity weight on phasic driver.
        gamma: L2 smoothness on spline coefficients.
        solver: cvxpy solver name. ECOS default; SCS as fallback.

    Returns:
        (tonic, phasic): each float32 ndarray of shape y.shape.
    """
    import cvxpy as cp
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < fs:
        return y.astype(np.float32), np.zeros_like(y, dtype=np.float32)

    delta = 1.0 / fs
    # Discrete-time biexponential IRF (Greco 2016 eq. 6)
    a1 = 1.0 / min(tau1, tau0)
    a0 = 1.0 / max(tau1, tau0)
    denom = (a1 - a0) * delta ** 2
    ar = np.array([
        (a1 * delta + 2.0) * (a0 * delta + 2.0),
        2.0 * a1 * a0 * delta ** 2 - 8.0,
        (a1 * delta - 2.0) * (a0 * delta - 2.0),
    ]) / denom
    ma = np.array([1.0, 2.0, 1.0])

    # Sparse banded M = ar Toeplitz; phasic_obs = M @ q (q is SMNA driver)
    M = sparse.diags([ar[2] * np.ones(n - 2),
                       ar[1] * np.ones(n - 1),
                       ar[0] * np.ones(n)],
                      offsets=[-2, -1, 0], shape=(n, n), format="csc")
    Mt = sparse.diags([ma[0] * np.ones(n),
                        ma[1] * np.ones(n - 1),
                        ma[2] * np.ones(n - 2)],
                       offsets=[0, -1, -2], shape=(n, n), format="csc")

    # Tonic basis: triangular hat functions every delta_knot seconds,
    # plus a linear trend column (offset + slope).
    knot_samples = max(2, int(delta_knot * fs))
    n_knots = max(2, n // knot_samples + 1)
    Cb = np.zeros((n, n_knots), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    for k_idx in range(n_knots):
        center = min(n - 1, k_idx * knot_samples)
        Cb[:, k_idx] = np.maximum(0.0, 1.0 - np.abs(idx - center) / knot_samples)
    Ct = np.column_stack([np.ones(n), idx / n])
    C = np.hstack([Cb, Ct])  # (n, n_knots + 2)

    q = cp.Variable(n, nonneg=True)
    l = cp.Variable(C.shape[1])
    tonic = C @ l
    phasic_obs = M @ q  # observed phasic component

    residual = y - tonic - phasic_obs
    obj = cp.Minimize(
        0.5 * cp.sum_squares(residual)
        + alpha * cp.norm(q, 1)
        + 0.5 * gamma * cp.sum_squares(l[:n_knots])
    )
    prob = cp.Problem(obj)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve(solver=solver, verbose=False, max_iters=1500)
    except Exception:
        return y.astype(np.float32), np.zeros_like(y, dtype=np.float32)

    if l.value is None or q.value is None:
        return y.astype(np.float32), np.zeros_like(y, dtype=np.float32)

    tonic_arr = (C @ l.value).astype(np.float32)
    phasic_arr = (M @ q.value).astype(np.float32)
    # Sanitize
    tonic_arr = np.nan_to_num(tonic_arr, nan=0.0, posinf=0.0, neginf=0.0)
    phasic_arr = np.nan_to_num(phasic_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return tonic_arr, phasic_arr


def _cvx_eda_stats(eda: np.ndarray, fs: int, tau0: float, tau1: float) -> np.ndarray:
    """7 stats from the tonic+phasic decomposition."""
    tonic, phasic = cvx_eda_decompose(eda, fs=fs, tau0=tau0, tau1=tau1)
    if tonic.size < 2:
        return np.zeros(7, dtype=np.float32)
    tonic_mean = float(tonic.mean())
    tonic_slope = float((tonic[-1] - tonic[0]) / max(1, tonic.size))
    # Phasic peaks (event count + amplitude)
    peaks, _ = find_peaks(phasic, height=max(1e-3, 0.1 * float(phasic.std())))
    peak_count = float(peaks.size)
    phasic_max = float(phasic.max())
    phasic_auc = float(phasic.sum() / fs)
    phasic_sparsity = float(np.mean(phasic < 1e-4))
    phasic_mean = float(phasic.mean())
    out = np.array([tonic_mean, tonic_slope, peak_count, phasic_max,
                     phasic_auc, phasic_sparsity, phasic_mean],
                    dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _channel_stats_3(channel: np.ndarray) -> np.ndarray:
    c = np.asarray(channel, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros(3, dtype=np.float32)
    mean = float(c.mean())
    std = float(c.std(ddof=1)) if c.size > 1 else 0.0
    energy = float((c ** 2).mean())
    return np.array([mean, std, energy], dtype=np.float32)


def compute_per_trial_features(trial: np.ndarray, fs: int = 100,
                                  tau0: float = 2.0,
                                  tau1: float = 0.7) -> np.ndarray:
    """Fixed-dim feature vector per trial.

    trial: (T, C>=4) array with channels [BVP, EDA, RESP, SpO2].
    Returns: float32 ndarray of shape (EDA_FEATURE_DIM,).
    """
    trial = np.asarray(trial, dtype=np.float32)
    if trial.ndim != 2 or trial.shape[1] < 4:
        raise ValueError(f"expected (T, >=4) trial, got shape {trial.shape}")
    eda_stats = _cvx_eda_stats(trial[:, 1], fs=fs, tau0=tau0, tau1=tau1)
    bvp_hrv = compute_hrv_features(trial[:, 0], fs=fs)
    hrv_vec = np.array([bvp_hrv["rmssd"], bvp_hrv["sdnn"],
                          bvp_hrv["pnn50"], bvp_hrv["mean_hr"]],
                         dtype=np.float32)
    resp_stats = _channel_stats_3(trial[:, 2])
    spo2_stats = _channel_stats_3(trial[:, 3])
    out = np.concatenate([eda_stats, hrv_vec, resp_stats, spo2_stats])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class EDADecompMLP(nn.Module):
    """Fixed-dim feature vector -> 2-layer MLP -> logits."""

    def __init__(self, n_features: int = EDA_FEATURE_DIM,
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


def _featurize_split(X: list[np.ndarray], fs: int, tau0: float,
                       tau1: float, tag: str) -> np.ndarray:
    feats = []
    for i, x in enumerate(X):
        f = compute_per_trial_features(x, fs=fs, tau0=tau0, tau1=tau1)
        feats.append(f)
        if (i + 1) % 50 == 0:
            print(f"[eda_decomp_mlp] {tag} cvxEDA progress: "
                  f"{i + 1}/{len(X)}", flush=True)
    return np.stack(feats).astype(np.float32)


def train_eda_decomp(spec: dict, data_root: Path, out_dir: Path) -> dict:
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
    tau0 = float(fe.get("tau0", 2.0))
    tau1 = float(fe.get("tau1", 0.7))

    print(f"[eda_decomp_mlp] loading train from {data_root}", flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    print(f"[eda_decomp_mlp] {len(X_train)} train / {len(X_val)} val trials",
          flush=True)

    print(f"[eda_decomp_mlp] cvxEDA featurize (fs={fs}, tau0={tau0}, "
          f"tau1={tau1})...", flush=True)
    Ftr = _featurize_split(X_train, fs=fs, tau0=tau0, tau1=tau1, tag="train")
    Fv = _featurize_split(X_val, fs=fs, tau0=tau0, tau1=tau1, tag="val")

    mu = Ftr.mean(axis=0, keepdims=True)
    sigma = Ftr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Ftr_n = ((Ftr - mu) / sigma).astype(np.float32)
    Fv_n = ((Fv - mu) / sigma).astype(np.float32)

    device = _device()
    print(f"[eda_decomp_mlp] device: {device}", flush=True)
    model_cfg = spec.get("model", {})
    model = EDADecompMLP(
        n_features=EDA_FEATURE_DIM,
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
        print(f"[eda_decomp_mlp] ep {epoch}: "
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
        "name": spec.get("name", "eda_decomp_mlp"),
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
    print(f"[eda_decomp_mlp] best val bal_acc: {best_val_bal_acc:.3f}",
          flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    return train_eda_decomp(spec, data_root=Path(data_root), out_dir=run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
