"""HIP-G test-set submission runner.

The evolutionary loop only ever evaluates on the 12-subject validation split.
A challenge submission needs predictions on the BLINDED 12-subject test split.
`run_submission` trains a chosen spec on the 41 train subjects, early-stops on
the validation split (same protocol the loop used), and at the best-val epoch
runs inference on the test split -- writing `test_predictions.csv`.

Supported families: spectrogram_cnn2d, multi_stream_bigru, dual_ensemble.
Each provides its own input representation; a single generic training+predict
loop (`_train_and_predict`) is shared -- `model(*inputs)` handles the 1-tensor
(spectrogram / multi_stream) and 2-tensor (dual) forward signatures uniformly.

Submission budget is hard-capped at 5 (HIP-G); each requires Vignan's explicit
approval and is logged in SUBMISSIONS.md.
"""
from __future__ import annotations

import argparse
import csv
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
from ai4pain.baselines import (pad_trials_to_max, _device, _atomic_write_json)
from ai4pain.spectrogram import (SpectrogramCNN2D, compute_spectrogram_stack,
                                  pad_spectrograms_to_max)
from ai4pain.multi_stream import _multi_stream_factory
from ai4pain.dual_ensemble import DualEnsembleNet

SUPPORTED_FAMILIES = ("spectrogram_cnn2d", "multi_stream_bigru",
                       "dual_ensemble")
LABEL_NAMES = ["NP", "AP", "HP"]


def _align_time(*stacks: np.ndarray) -> list[np.ndarray]:
    """Right-zero-pad arrays to a common max size along the last axis."""
    t_max = max(s.shape[-1] for s in stacks)
    out = []
    for s in stacks:
        if s.shape[-1] < t_max:
            pad = np.zeros((*s.shape[:-1], t_max - s.shape[-1]),
                           dtype=np.float32)
            s = np.concatenate([s, pad], axis=-1)
        out.append(s)
    return out


def _prep_sequence(X_train, X_val, X_test):
    """Pad the three splits to a common T and per-channel z-score (fit on
    train). Returns (Xtr, Xv, Xte) float32 arrays of shape (N, T, C)."""
    Xtr = pad_trials_to_max(X_train)
    Xv = pad_trials_to_max(X_val)
    Xte = pad_trials_to_max(X_test)
    Xtr, Xv, Xte = _align_time(
        Xtr.transpose(0, 2, 1), Xv.transpose(0, 2, 1), Xte.transpose(0, 2, 1))
    Xtr, Xv, Xte = (a.transpose(0, 2, 1) for a in (Xtr, Xv, Xte))
    # per-channel z-score, fit on train
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True)
    sd[sd < 1e-6] = 1.0
    return tuple(((a - mu) / sd).astype(np.float32) for a in (Xtr, Xv, Xte))


def _prep_spectrogram(X_train, X_val, X_test, tf_kwargs):
    """Compute spectrogram stacks for the three splits, pad+align, per-channel
    z-score (fit on train). Returns (Str, Sv, Ste, F)."""
    def stacks(X):
        return pad_spectrograms_to_max(
            [compute_spectrogram_stack(x, **tf_kwargs) for x in X])
    Str, Sv, Ste = stacks(X_train), stacks(X_val), stacks(X_test)
    Str, Sv, Ste = _align_time(Str, Sv, Ste)
    flat = Str.reshape(Str.shape[0], Str.shape[1], -1)
    mu = flat.mean(axis=(0, 2))[None, :, None, None]
    sd = flat.std(axis=(0, 2))[None, :, None, None]
    sd[sd < 1e-6] = 1.0
    Str, Sv, Ste = (((a - mu) / sd).astype(np.float32)
                    for a in (Str, Sv, Ste))
    return Str, Sv, Ste, Str.shape[2]


def _make_loss(train_cfg: dict, y_train: np.ndarray, device):
    """Class-balanced CE, optionally focal (matches the family train loops)."""
    counts = np.bincount(y_train, minlength=3)
    w = (counts.sum() / (3 * counts)).astype(np.float32)
    hp_boost = float(train_cfg.get("hp_boost", 1.0))
    if hp_boost != 1.0:
        w = w.copy()
        w[2] *= hp_boost
    cw = torch.tensor(w, device=device)
    gamma = float(train_cfg.get("focal_gamma", 0.0))
    if gamma > 0.0:
        ce_per = nn.CrossEntropyLoss(weight=cw, reduction="none")
        def loss_fn(logits, y):
            ce = ce_per(logits, y)
            p = torch.softmax(logits, 1).gather(1, y.unsqueeze(1)).squeeze(1)
            return ((1.0 - p) ** gamma * ce).mean()
        return loss_fn
    return nn.CrossEntropyLoss(weight=cw)


def _write_predictions_csv(path, subjects, preds, probas, true_labels=None):
    """Write a per-trial predictions CSV.

    Without `true_labels` -> blinded-test format (subject, trial_index,
    pred_*, p_*). With `true_labels` (the validation split, labels known) ->
    extra true_label/true_name columns, so the file is self-contained for
    post-hoc ensemble scoring. trial_index is the row index.
    """
    path = Path(path)
    has_true = true_labels is not None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["subject", "trial_index"]
        if has_true:
            head += ["true_label", "true_name"]
        head += ["pred_label", "pred_name", "p_NP", "p_AP", "p_HP"]
        w.writerow(head)
        for i in range(len(preds)):
            row = [int(subjects[i]), i]
            if has_true:
                t = int(true_labels[i])
                row += [t, LABEL_NAMES[t]]
            p = int(preds[i])
            row += [p, LABEL_NAMES[p], f"{probas[i, 0]:.4f}",
                    f"{probas[i, 1]:.4f}", f"{probas[i, 2]:.4f}"]
            w.writerow(row)


def _train_and_predict(model, train_inputs, y_train, val_inputs, y_val,
                       subjects_val, test_inputs, subjects_test,
                       train_cfg, run_dir, spec):
    """Generic train (early-stop on val) + predict loop.

    At the best-val epoch, runs inference on BOTH the val split (->
    val_predictions.csv, with true labels, for post-hoc ensembling) and the
    blinded test split (-> test_predictions.csv).

    *_inputs are lists of np.float32 arrays; the model is called as
    model(*tensors), so 1-input (spectrogram/multi_stream) and 2-input (dual)
    families share this loop.
    """
    device = _device()
    model = model.to(device)
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_t = [torch.from_numpy(a).to(device) for a in train_inputs]
    va_t = [torch.from_numpy(a).to(device) for a in val_inputs]
    te_t = [torch.from_numpy(a).to(device) for a in test_inputs]
    ytr_t = torch.from_numpy(y_train).to(device)

    epochs = int(train_cfg.get("epochs", 90))
    bs = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    optim = (torch.optim.AdamW if train_cfg.get("optimizer") == "adamw"
             else torch.optim.Adam)(model.parameters(), lr=lr)
    loss_fn = _make_loss(train_cfg, y_train, device)

    loader = DataLoader(TensorDataset(*tr_t, ytr_t), batch_size=bs,
                        shuffle=True)
    best_val = -math.inf
    best_state = None
    best_val_metrics: dict = {}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            *xb, yb = batch
            optim.zero_grad()
            loss_fn(model(*xb), yb).backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(*va_t)
            vp = vlogits.argmax(1).cpu().numpy()
            vproba = vlogits.softmax(1).cpu().numpy()
            vm = full_metric_suite(y_val, vp, vproba)
        if vm["balanced_acc"] > best_val:
            best_val = vm["balanced_acc"]
            best_val_metrics = vm
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[submission] ep {epoch}: val_bal={vm['balanced_acc']:.4f}",
              flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(*te_t)
        te_proba = te_logits.softmax(1).cpu().numpy()
        te_pred = te_logits.argmax(1).cpu().numpy()
        # val predictions at the best-val epoch -- for post-hoc ensembling
        va_logits = model(*va_t)
        va_proba = va_logits.softmax(1).cpu().numpy()
        va_pred = va_logits.argmax(1).cpu().numpy()

    run_dir = Path(run_dir)
    pred_path = run_dir / "test_predictions.csv"
    _write_predictions_csv(pred_path, subjects_test, te_pred, te_proba)
    val_path = run_dir / "val_predictions.csv"
    _write_predictions_csv(val_path, subjects_val, va_pred, va_proba,
                           true_labels=y_val)

    result = {
        "name": spec.get("name", "submission"),
        "submission": True,
        "best_val_metrics": best_val_metrics,
        "test_n_trials": int(len(te_pred)),
        "test_pred_class_counts": {LABEL_NAMES[c]: int((te_pred == c).sum())
                                    for c in range(3)},
        "test_predictions_csv": str(pred_path),
        "val_predictions_csv": str(val_path),
        "train_seconds": time.time() - t0,
        "device": str(device),
        "spec": spec,
    }
    _atomic_write_json(run_dir / "result.json", result)
    print(f"[submission] best val bal_acc: {best_val:.4f}", flush=True)
    print(f"[submission] val predictions -> {val_path} "
          f"({len(va_pred)} trials)", flush=True)
    print(f"[submission] test predictions -> {pred_path} "
          f"({len(te_pred)} trials)", flush=True)
    print(f"[submission] test class counts: "
          f"{result['test_pred_class_counts']}", flush=True)
    return result


def _spectrogram_tf_kwargs(fe: dict) -> dict:
    return dict(fs=int(fe.get("fs", 100)),
                nperseg=int(fe.get("nperseg", 64)),
                noverlap=int(fe.get("noverlap", 32)),
                log_scale=bool(fe.get("log_scale", True)),
                transform=fe.get("transform", "stft"),
                cwt_n_scales=int(fe.get("cwt_n_scales", 48)),
                cwt_time_decim=int(fe.get("cwt_time_decim", 24)),
                cwt_w0=float(fe.get("cwt_w0", 6.0)))


def run_submission(run_dir: Path, data_root: Path) -> dict:
    """Train spec.json's model on train, early-stop on val, predict test."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads((run_dir / "spec.json").read_text())

    family = spec.get("model", {}).get("family")
    if family not in SUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"submission runner supports {SUPPORTED_FAMILIES}, got {family!r}")

    signals = tuple(spec.get("data", {}).get(
        "signals", ["Bvp", "Eda", "Resp", "SpO2"]))
    fe = spec.get("feature_extraction", {}) or {}
    mc = spec.get("model", {})
    train_cfg = spec.get("training", {})

    print(f"[submission] family={family}, loading splits from {data_root}",
          flush=True)
    X_train, y_train, _ = load_split(data_root, "train", signals=signals)
    X_val, y_val, subj_val = load_split(data_root, "validation",
                                        signals=signals)
    X_test, _, subj_test = load_split(data_root, "test", signals=signals)
    print(f"[submission] {len(X_train)} train / {len(X_val)} val / "
          f"{len(X_test)} test trials", flush=True)

    if family == "spectrogram_cnn2d":
        Str, Sv, Ste, F = _prep_spectrogram(
            X_train, X_val, X_test, _spectrogram_tf_kwargs(fe))
        model = SpectrogramCNN2D(
            in_channels=len(signals), F=F,
            base_channels=int(mc.get("base_channels", 16)),
            depth=int(mc.get("depth", 2)),
            dropout=float(mc.get("dropout", 0.2)),
            use_residual=bool(mc.get("use_residual", False)),
            num_classes=3)
        return _train_and_predict(model, [Str], y_train, [Sv], y_val,
                                   subj_val, [Ste], subj_test, train_cfg,
                                   run_dir, spec)

    if family == "multi_stream_bigru":
        Xtr, Xv, Xte = _prep_sequence(X_train, X_val, X_test)
        model = _multi_stream_factory(
            in_channels=len(signals), T_max=Xtr.shape[1],
            model_cfg=mc, num_classes=3)
        return _train_and_predict(model, [Xtr], y_train, [Xv], y_val,
                                   subj_val, [Xte], subj_test, train_cfg,
                                   run_dir, spec)

    # dual_ensemble
    Xtr, Xv, Xte = _prep_sequence(X_train, X_val, X_test)
    Str, Sv, Ste, F = _prep_spectrogram(
        X_train, X_val, X_test, _spectrogram_tf_kwargs(fe))
    model = DualEnsembleNet(
        in_channels=len(signals), spec_F=F, num_classes=3,
        gru_cfg=mc.get("gru_cfg", {}), cnn_cfg=mc.get("cnn_cfg", {}))
    return _train_and_predict(model, [Xtr, Str], y_train, [Xv, Sv], y_val,
                               subj_val, [Xte, Ste], subj_test, train_cfg,
                               run_dir, spec)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return run_submission(Path(run_dir), Path(data_root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
