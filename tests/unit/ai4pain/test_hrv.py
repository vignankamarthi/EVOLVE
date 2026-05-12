"""Tests for ai4pain.hrv (HRV feature extraction + MLP classifier).

Family: `hrv_features_mlp`. BVP peak detection -> RMSSD/SDNN/pNN50/mean_HR/
LF/HF features per trial -> small MLP. Per-trial features are channel-aware
auxiliary stats (mean, std, min, max, range) on the other physiological
channels (EDA, RESP, SpO2). Total per-trial feature vector is a fixed-size
1D numpy array, no time dimension.

Why this family: HRV time + frequency features are the canonical input for
autonomic-state classification (Xia et al. 2024 hit 98%+ on stress with
CNN-LSTM-Transformer over HRV features, while raw signals plateau at lower
accuracy). The deep models we've been running ingest raw BVP, which contains
LESS class signal than its derived HRV decomposition. This family closes
that gap.
"""
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ai4pain import hrv


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()


def test_module_imports():
    assert hasattr(hrv, "compute_hrv_features")
    assert hasattr(hrv, "compute_per_trial_features")
    assert hasattr(hrv, "HRVFeaturesMLP")
    assert callable(hrv.train_hrv)
    assert callable(hrv.run_from_dir)


def test_compute_hrv_features_returns_expected_keys():
    """A 10-second synthetic BVP at ~60 BPM (fs=100Hz) should produce
    a full feature dict with all standard HRV keys."""
    fs = 100
    t = np.arange(0, 10, 1.0 / fs)
    # Synthetic BVP at 1 Hz (60 BPM) with small jitter
    # Clean synthetic signal: a 1 Hz peak train (60 BPM) with a sharp
    # narrowband peak shape so find_peaks unambiguously locks on once per period.
    bvp = np.maximum(0, np.sin(2 * np.pi * 1.0 * t)) ** 4
    feats = hrv.compute_hrv_features(bvp.astype(np.float32), fs=fs)
    expected = {"rmssd", "sdnn", "pnn50", "mean_hr", "lf_power", "hf_power",
                "lf_hf_ratio", "n_peaks"}
    assert expected.issubset(set(feats.keys()))
    # Mean HR for a 1 Hz beat = 60 BPM (within tolerance)
    assert 50 < feats["mean_hr"] < 70, f"got {feats['mean_hr']}"


def test_compute_hrv_features_handles_short_signal():
    """A signal too short for peak detection should still return a dict,
    not crash. Values default to 0 (no peaks)."""
    bvp = np.zeros(50, dtype=np.float32)
    feats = hrv.compute_hrv_features(bvp, fs=100)
    assert feats["n_peaks"] == 0
    assert feats["rmssd"] == 0.0


def test_compute_per_trial_features_returns_fixed_dim_vector():
    """Per-trial feature vector should be a fixed-length 1D float32 ndarray
    regardless of trial length T."""
    fs = 100
    rng = np.random.default_rng(0)
    short = rng.standard_normal((300, 4)).astype(np.float32)
    long = rng.standard_normal((1500, 4)).astype(np.float32)
    f_short = hrv.compute_per_trial_features(short, fs=fs)
    f_long = hrv.compute_per_trial_features(long, fs=fs)
    assert f_short.ndim == 1
    assert f_long.ndim == 1
    assert f_short.shape == f_long.shape  # fixed dim regardless of T
    assert f_short.dtype == np.float32
    assert hrv.HRV_FEATURE_DIM == f_short.shape[0]


def test_hrv_features_mlp_forward_shape():
    n = hrv.HRV_FEATURE_DIM
    model = hrv.HRVFeaturesMLP(n_features=n, hidden=32, num_classes=3)
    x = torch.randn(8, n)
    out = model(x)
    assert out.shape == (8, 3)


def test_hrv_features_mlp_backward_no_nan():
    n = hrv.HRV_FEATURE_DIM
    model = hrv.HRVFeaturesMLP(n_features=n, hidden=32, num_classes=3)
    x = torch.randn(4, n)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_smoke_train_hrv_writes_result(tmp_path: Path):
    spec = {
        "name": "smoke_hrv",
        "preprocessing": {"normalize": "per_feature_zscore"},
        "feature_extraction": {"family": "hrv_features", "fs": 100},
        "model": {"family": "hrv_features_mlp", "hidden": 16, "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0,
                     "loss": "ce_class_balanced", "optimizer": "adam"},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
        "decode": {"strategy": "argmax"},
    }
    result = hrv.train_hrv(spec, data_root=DATA_ROOT, out_dir=tmp_path)
    assert (tmp_path / "result.json").exists()
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert 0.0 <= persisted["best_val_metrics"]["balanced_acc"] <= 1.0
    assert "confusion_3x3" in persisted["best_val_metrics"]
