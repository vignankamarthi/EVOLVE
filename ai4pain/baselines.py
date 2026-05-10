"""BiGRU baseline for the AI4Pain 2026 case study.

One-off baseline used to validate the manual cluster trio (HIP-D / HIP-E /
HIP-F) and the ai4pain.data, ai4pain.splits, ai4pain.metrics, framework.ledger
modules end-to-end. NOT part of the EVOLVE search space; framework.render will
eventually generate richer programs. Kept as `framework/seeds.py` candidate.

Architecture:
  Input: (B, T, C=4)
  Bidirectional GRU x num_layers
  Mean pool over T
  Dropout
  Linear -> num_classes

Pipeline:
  1. Load train and val via ai4pain.data.load_split (real data)
  2. Pad variable-length trials with zeros to global max
  3. Per-channel z-score, fit on train only (ANTIPATTERNS rule 3)
  4. Train with class-balanced cross-entropy
  5. Track full metric suite per epoch
  6. Atomic-write result.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai4pain.data import load_split
from ai4pain.metrics import full_metric_suite


class BiGRUClassifier(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2,
                 num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.fc(self.dropout(pooled))


def pad_trials_to_max(trials: list[np.ndarray]) -> np.ndarray:
    """Right-pad list of (T_i, C) float32 arrays with zeros to (N, T_max, C)."""
    if not trials:
        return np.zeros((0, 0, 0), dtype=np.float32)
    n = len(trials)
    T_max = max(t.shape[0] for t in trials)
    C = trials[0].shape[1]
    out = np.zeros((n, T_max, C), dtype=np.float32)
    for i, t in enumerate(trials):
        out[i, :t.shape[0]] = t
    return out


def per_channel_zscore(train_padded: np.ndarray,
                        val_padded: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit (mean, std) on train (per channel, across N and T), apply to both.

    ANTIPATTERNS rule 3: scaler fits on train only.
    """
    mean = train_padded.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train_padded.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    train_norm = ((train_padded - mean) / std).astype(np.float32)
    val_norm = ((val_padded - mean) / std).astype(np.float32)
    return train_norm, val_norm, mean.squeeze(), std.squeeze()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: write to .tmp then rename."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def train_baseline(spec: dict, data_root: Path, out_dir: Path) -> dict:
    """End-to-end train of the BiGRU baseline.

    Args:
        spec: dict with optional sections 'model', 'training', 'data'.
        data_root: project's data/raw/ root.
        out_dir: directory to write result.json into.

    Returns:
        result dict (also written to out_dir/result.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = spec.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    signals = tuple(spec.get("data", {}).get("signals",
                                              ["Bvp", "Eda", "Resp", "SpO2"]))

    print(f"[baseline_bigru] loading train from {data_root}", flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    print(f"[baseline_bigru] {len(X_train)} train trials", flush=True)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    print(f"[baseline_bigru] {len(X_val)} val trials", flush=True)

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
    print(f"[baseline_bigru] device: {device}", flush=True)
    model_cfg = spec.get("model", {})
    model = BiGRUClassifier(
        in_channels=len(signals),
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        num_classes=3,
    ).to(device)

    epochs = int(train_cfg.get("epochs", 20))
    bs = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(y_train).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)

    counts = np.bincount(y_train, minlength=3)
    class_weights = torch.tensor((counts.sum() / (3 * counts)).astype(np.float32),
                                 device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(Xtr_t, ytr_t)
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
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            train_logits = model(Xtr_t)
            train_preds = train_logits.argmax(dim=1).cpu().numpy()
            train_bal = float(balanced_accuracy_score(y_train, train_preds))

            val_logits = model(Xv_t)
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
        print(f"[baseline_bigru] ep {epoch}: "
              f"train_loss={epoch_loss/max(n_batches,1):.4f} "
              f"train_bal={train_bal:.3f} "
              f"val_bal={final_metrics['balanced_acc']:.3f} "
              f"val_f1={final_metrics['macro_f1']:.3f}", flush=True)

    train_seconds = time.time() - t0

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        _ = model(Xv_t)
    inference_seconds = time.time() - t1

    result = {
        "name": spec.get("name", "baseline_bigru"),
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
    print(f"[baseline_bigru] best val bal_acc: {best_val_bal_acc:.3f}", flush=True)
    print(f"[baseline_bigru] result -> {out_dir / 'result.json'}", flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    """Read spec.json from run_dir, train, write result.json into run_dir."""
    run_dir = Path(run_dir)
    spec_path = run_dir / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"missing spec.json at {spec_path}")
    spec = json.loads(spec_path.read_text())
    return train_baseline(spec, data_root=Path(data_root), out_dir=run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
