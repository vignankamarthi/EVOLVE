"""Dual-architecture ensemble (family: `dual_ensemble`).

iter_0015-0017 produced two co-champions that plateaued at ~0.536:
  - a 1D multi_stream GRU (per-class strength: AP)
  - a 2D spectrogram CNN  (per-class strength: HP)
Different inductive biases -> decorrelated errors. `DualEnsembleNet` runs both
sub-models and fuses their logits with a LEARNED per-class blend weight:

    logits[c] = w[c] * cnn_logits[c] + (1 - w[c]) * gru_logits[c]

w = sigmoid(blend), blend a learnable 3-vector. Per-class (not scalar) so the
ensemble can lean on the spectrogram for HP and the GRU for AP -- the exact
complementarity the iter_0017 per-class profiles showed.

Both sub-models are trained jointly from scratch in one run (we do not persist
champion checkpoints). `forward(x_seq, x_spec)` takes the raw (B,T,C) sequence
for the GRU and the (B,C,F,T') spectrogram stack for the CNN; the training
function builds both representations.
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
from ai4pain.multi_stream import MultiStreamNet
from ai4pain.spectrogram import (SpectrogramCNN2D, compute_spectrogram_stack,
                                  pad_spectrograms_to_max)


class DualEnsembleNet(nn.Module):
    """MultiStreamNet (1D) + SpectrogramCNN2D (2D), learned per-class blend."""

    def __init__(self, in_channels: int = 4, spec_F: int = 33,
                 num_classes: int = 3,
                 gru_cfg: dict | None = None,
                 cnn_cfg: dict | None = None):
        super().__init__()
        gru_cfg = gru_cfg or {}
        cnn_cfg = cnn_cfg or {}
        self.gru = MultiStreamNet(
            in_channels=in_channels,
            per_channel_hidden=int(gru_cfg.get("per_channel_hidden", 40)),
            per_channel_layers=int(gru_cfg.get("per_channel_layers", 2)),
            encoder_type=gru_cfg.get("encoder_type", "gru"),
            fusion=gru_cfg.get("fusion", "late_concat"),
            fusion_dropout=float(gru_cfg.get("fusion_dropout", 0.2)),
            num_classes=num_classes)
        self.cnn = SpectrogramCNN2D(
            in_channels=in_channels, F=spec_F,
            base_channels=int(cnn_cfg.get("base_channels", 24)),
            depth=int(cnn_cfg.get("depth", 3)),
            dropout=float(cnn_cfg.get("dropout", 0.25)),
            num_classes=num_classes)
        # Per-class blend. sigmoid(blend) in (0,1); init 0 -> start at 0.5/0.5.
        self.blend = nn.Parameter(torch.zeros(num_classes))

    def blend_weights(self) -> torch.Tensor:
        """Effective per-class CNN weight, sigmoid-bounded to (0, 1)."""
        return torch.sigmoid(self.blend)

    def forward(self, x_seq: torch.Tensor,
                x_spec: torch.Tensor) -> torch.Tensor:
        gru_logits = self.gru(x_seq)
        cnn_logits = self.cnn(x_spec)
        w = torch.sigmoid(self.blend)            # (num_classes,)
        return w * cnn_logits + (1.0 - w) * gru_logits


def _build_spectrograms(X: list[np.ndarray], tf_kwargs: dict
                         ) -> list[np.ndarray]:
    return [compute_spectrogram_stack(x, **tf_kwargs) for x in X]


def train_dual_ensemble(spec: dict, data_root: Path, out_dir: Path) -> dict:
    """End-to-end train of the dual_ensemble family."""
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
    tf_kwargs = dict(
        fs=fs,
        nperseg=int(fe.get("nperseg", 64)),
        noverlap=int(fe.get("noverlap", 32)),
        log_scale=bool(fe.get("log_scale", True)),
        transform=fe.get("transform", "stft"),
        cwt_n_scales=int(fe.get("cwt_n_scales", 48)),
        cwt_time_decim=int(fe.get("cwt_time_decim", 24)),
        cwt_w0=float(fe.get("cwt_w0", 6.0)),
    )

    print(f"[dual_ensemble] loading train from {data_root}", flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    X_val, y_val, _ = load_split(data_root, "validation", signals=signals)
    print(f"[dual_ensemble] {len(X_train)} train / {len(X_val)} val trials",
          flush=True)

    # --- sequence representation (GRU input): pad + per-channel zscore ---
    Xtr = pad_trials_to_max(X_train)
    Xv = pad_trials_to_max(X_val)
    T_max = max(Xtr.shape[1], Xv.shape[1])
    if Xtr.shape[1] < T_max:
        Xtr = np.concatenate([Xtr, np.zeros(
            (Xtr.shape[0], T_max - Xtr.shape[1], Xtr.shape[2]),
            dtype=np.float32)], axis=1)
    if Xv.shape[1] < T_max:
        Xv = np.concatenate([Xv, np.zeros(
            (Xv.shape[0], T_max - Xv.shape[1], Xv.shape[2]),
            dtype=np.float32)], axis=1)
    Xtr, Xv, _, _ = per_channel_zscore(Xtr, Xv)

    # --- spectrogram representation (CNN input) ---
    print(f"[dual_ensemble] transform={tf_kwargs['transform']} ...", flush=True)
    Str = pad_spectrograms_to_max(_build_spectrograms(X_train, tf_kwargs))
    Sv = pad_spectrograms_to_max(_build_spectrograms(X_val, tf_kwargs))
    St_max = max(Str.shape[-1], Sv.shape[-1])
    if Str.shape[-1] < St_max:
        Str = np.concatenate([Str, np.zeros(
            (*Str.shape[:-1], St_max - Str.shape[-1]), dtype=np.float32)],
            axis=-1)
    if Sv.shape[-1] < St_max:
        Sv = np.concatenate([Sv, np.zeros(
            (*Sv.shape[:-1], St_max - Sv.shape[-1]), dtype=np.float32)],
            axis=-1)
    # per-channel zscore over (F, T'), fit on train
    smu = Str.reshape(Str.shape[0], Str.shape[1], -1).mean(axis=(0, 2))
    ssig = Str.reshape(Str.shape[0], Str.shape[1], -1).std(axis=(0, 2))
    ssig[ssig < 1e-6] = 1.0
    smu = smu[None, :, None, None]
    ssig = ssig[None, :, None, None]
    Str = ((Str - smu) / ssig).astype(np.float32)
    Sv = ((Sv - smu) / ssig).astype(np.float32)
    spec_F = Str.shape[2]

    device = _device()
    print(f"[dual_ensemble] device: {device}, spec stack {Str.shape}",
          flush=True)
    model_cfg = spec.get("model", {})
    model = DualEnsembleNet(
        in_channels=len(signals), spec_F=spec_F, num_classes=3,
        gru_cfg=model_cfg.get("gru_cfg", {}),
        cnn_cfg=model_cfg.get("cnn_cfg", {}),
    ).to(device)

    epochs = int(train_cfg.get("epochs", 20))
    bs = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    optim_name = train_cfg.get("optimizer", "adam").lower()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Str_t = torch.from_numpy(Str).to(device)
    ytr_t = torch.from_numpy(y_train).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    Sv_t = torch.from_numpy(Sv).to(device)

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

    train_loader = DataLoader(TensorDataset(Xtr_t, Str_t, ytr_t),
                              batch_size=bs, shuffle=True)

    history: list[dict] = []
    best_val_bal_acc = -math.inf
    best_metrics: dict = {}
    final_metrics: dict = {}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, sb, yb in train_loader:
            optim.zero_grad()
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        model.eval()
        with torch.no_grad():
            tr_preds = model(Xtr_t, Str_t).argmax(dim=1).cpu().numpy()
            train_bal = float(balanced_accuracy_score(y_train, tr_preds))
            v_logits = model(Xv_t, Sv_t)
            v_proba = v_logits.softmax(dim=1).cpu().numpy()
            v_preds = v_logits.argmax(dim=1).cpu().numpy()
            final_metrics = full_metric_suite(y_val, v_preds, v_proba)
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
        bw = model.blend_weights().detach().cpu().numpy()
        print(f"[dual_ensemble] ep {epoch}: "
              f"train_bal={train_bal:.3f} "
              f"val_bal={final_metrics['balanced_acc']:.3f} "
              f"blend(cnn)={bw.round(2).tolist()}", flush=True)

    train_seconds = time.time() - t0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        _ = model(Xv_t, Sv_t)
    inference_seconds = time.time() - t1

    result = {
        "name": spec.get("name", "dual_ensemble"),
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "history": history,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "blend_weights_cnn": model.blend_weights().detach().cpu().tolist(),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "device": str(device),
        "spec": spec,
    }
    _atomic_write_json(out_dir / "result.json", result)
    print(f"[dual_ensemble] best val bal_acc: {best_val_bal_acc:.3f}",
          flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    from ai4pain.multiseed import run_multiseed
    return run_multiseed(train_dual_ensemble, spec, Path(data_root), run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
