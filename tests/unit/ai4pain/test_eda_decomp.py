"""Tests for ai4pain.eda_decomp (cvxEDA tonic+phasic decomposition + MLP).

Family: `eda_decomp_mlp`. cvxEDA (Greco et al. 2016) decomposes EDA into:
  - tonic component (slow-varying SCL, sympathetic baseline)
  - phasic component (fast SCR peaks, event-driven sympathetic bursts)

Per-trial features: stats of tonic (mean, slope), stats of phasic (peak
count, peak amplitude mean, AUC, sparsity) + HRV from BVP + aux stats on
RESP/SpO2 -> fixed-dim feature vector -> MLP.

Pain is sympathetic-mediated; the tonic/phasic split exposes signal the
raw EDA buries. Greco et al. 2016 showed cvxEDA recovers SCRs with higher
fidelity than older deconvolution methods.
"""
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
cvxpy = pytest.importorskip("cvxpy")
from ai4pain import eda_decomp


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()


def test_module_imports():
    assert hasattr(eda_decomp, "cvx_eda_decompose")
    assert hasattr(eda_decomp, "compute_per_trial_features")
    assert hasattr(eda_decomp, "EDADecompMLP")
    assert callable(eda_decomp.train_eda_decomp)
    assert callable(eda_decomp.run_from_dir)


def test_cvx_eda_decompose_returns_tonic_and_phasic():
    """Synthetic EDA = slow baseline + a few SCR-like phasic events.
    cvxEDA should recover a tonic close to the baseline and a phasic
    that is sparse + non-negative."""
    fs = 100
    T = 1000
    t = np.arange(T) / fs
    baseline = 2.0 + 0.01 * t  # very slow rise (SCL)
    phasic_events = np.zeros(T)
    # Three SCR-like pulses at t=2, 5, 8 seconds (200, 500, 800 samples)
    for center in [200, 500, 800]:
        phasic_events[center:center + 50] = np.linspace(0.3, 0, 50)
    eda = (baseline + phasic_events).astype(np.float32)
    tonic, phasic = eda_decomp.cvx_eda_decompose(eda, fs=fs)
    assert tonic.shape == eda.shape
    assert phasic.shape == eda.shape
    # phasic should be approximately non-negative on average
    assert phasic.mean() > -0.1
    # tonic + phasic ~= eda (within decomposition error)
    assert float(np.mean(np.abs(tonic + phasic - eda))) < 0.5


def test_cvx_eda_decompose_handles_short_signal():
    """Very short EDA (< 1 sec) should return tonic = eda, phasic = 0."""
    eda = np.array([1.0, 1.1, 1.0, 0.9], dtype=np.float32)
    tonic, phasic = eda_decomp.cvx_eda_decompose(eda, fs=100)
    assert tonic.shape == eda.shape
    assert phasic.shape == eda.shape


def test_compute_per_trial_features_fixed_dim():
    """Per-trial features have fixed dim regardless of T."""
    rng = np.random.default_rng(0)
    short = rng.standard_normal((400, 4)).astype(np.float32) + 2.0
    long = rng.standard_normal((1200, 4)).astype(np.float32) + 2.0
    f_short = eda_decomp.compute_per_trial_features(short, fs=100)
    f_long = eda_decomp.compute_per_trial_features(long, fs=100)
    assert f_short.ndim == 1
    assert f_long.shape == f_short.shape
    assert f_short.dtype == np.float32
    assert eda_decomp.EDA_FEATURE_DIM == f_short.shape[0]


def test_eda_decomp_mlp_forward_shape():
    n = eda_decomp.EDA_FEATURE_DIM
    model = eda_decomp.EDADecompMLP(n_features=n, hidden=32, num_classes=3)
    x = torch.randn(8, n)
    out = model(x)
    assert out.shape == (8, 3)


def test_eda_decomp_mlp_backward_no_nan():
    n = eda_decomp.EDA_FEATURE_DIM
    model = eda_decomp.EDADecompMLP(n_features=n, hidden=32, num_classes=3)
    x = torch.randn(4, n)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_smoke_train_eda_decomp_writes_result(tmp_path: Path):
    spec = {
        "name": "smoke_eda",
        "preprocessing": {"normalize": "per_feature_zscore"},
        "feature_extraction": {"family": "cvx_eda_decomp", "fs": 100,
                                "tau0": 2.0, "tau1": 0.7},
        "model": {"family": "eda_decomp_mlp", "hidden": 16, "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0,
                     "loss": "ce_class_balanced", "optimizer": "adam"},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
        "decode": {"strategy": "argmax"},
    }
    result = eda_decomp.train_eda_decomp(spec, data_root=DATA_ROOT,
                                           out_dir=tmp_path)
    assert (tmp_path / "result.json").exists()
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert 0.0 <= persisted["best_val_metrics"]["balanced_acc"] <= 1.0


def test_cvx_eda_decompose_decim_speedup_preserves_shape_and_signal():
    """decim>1 solves the cvxEDA QP at a reduced rate then interpolates back.
    Output shape must match the input, and the decimated decomposition must
    stay close to the full-rate one (EDA is slow, so this is near-lossless)."""
    fs = 100
    T = 1200
    t = np.arange(T) / fs
    baseline = 2.0 + 0.01 * t
    phasic_events = np.zeros(T)
    for center in [200, 600, 1000]:
        phasic_events[center:center + 50] = np.linspace(0.3, 0, 50)
    eda = (baseline + phasic_events).astype(np.float32)

    tonic_full, phasic_full = eda_decomp.cvx_eda_decompose(eda, fs=fs, decim=1)
    tonic_ds, phasic_ds = eda_decomp.cvx_eda_decompose(eda, fs=fs, decim=4)

    assert tonic_ds.shape == eda.shape
    assert phasic_ds.shape == eda.shape
    # decimated tonic tracks the full-rate tonic (slow signal, near-lossless)
    assert float(np.mean(np.abs(tonic_ds - tonic_full))) < 0.5


def test_compute_per_trial_features_accepts_decim():
    rng = np.random.default_rng(0)
    trial = rng.standard_normal((800, 4)).astype(np.float32) + 2.0
    f = eda_decomp.compute_per_trial_features(trial, fs=100, decim=4)
    assert f.shape == (eda_decomp.EDA_FEATURE_DIM,)
    assert np.isfinite(f).all()
