"""Tests for ai4pain.baselines (BiGRU baseline)."""
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ai4pain import baselines


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()


def test_module_imports():
    assert hasattr(baselines, "BiGRUClassifier")
    assert callable(baselines.train_baseline)
    assert callable(baselines.run_from_dir)


def test_pad_trials_to_max_length():
    trials = [
        np.random.randn(100, 4).astype(np.float32),
        np.random.randn(150, 4).astype(np.float32),
        np.random.randn(120, 4).astype(np.float32),
    ]
    padded = baselines.pad_trials_to_max(trials)
    assert padded.shape == (3, 150, 4)
    assert np.allclose(padded[0, :100], trials[0])
    assert np.allclose(padded[0, 100:], 0.0)
    assert np.allclose(padded[2, 120:], 0.0)


def test_pad_trials_empty_list():
    out = baselines.pad_trials_to_max([])
    assert out.shape == (0, 0, 0)


def test_per_channel_zscore_normalizes_train():
    rng = np.random.default_rng(0)
    train = (rng.standard_normal((20, 100, 4)).astype(np.float32) * 5 + 3)
    val = (rng.standard_normal((5, 100, 4)).astype(np.float32) * 5 + 3)
    train_n, val_n, mean, std = baselines.per_channel_zscore(train, val)
    for c in range(4):
        assert abs(float(train_n[..., c].mean())) < 0.1
        assert abs(float(train_n[..., c].std()) - 1.0) < 0.1


def test_per_channel_zscore_fit_only_on_train():
    """ANTIPATTERNS rule 3: scaler fits on train only. Same val passed twice
    with different train should yield different val_n."""
    rng = np.random.default_rng(0)
    val = rng.standard_normal((5, 100, 4)).astype(np.float32)
    train_a = rng.standard_normal((20, 100, 4)).astype(np.float32) * 1.0
    train_b = rng.standard_normal((20, 100, 4)).astype(np.float32) * 10.0
    _, val_a, _, _ = baselines.per_channel_zscore(train_a, val)
    _, val_b, _, _ = baselines.per_channel_zscore(train_b, val)
    assert not np.allclose(val_a, val_b)


def test_bigru_forward_shape():
    model = baselines.BiGRUClassifier(in_channels=4, hidden_size=16,
                                       num_layers=1, num_classes=3)
    x = torch.randn(8, 200, 4)
    out = model(x)
    assert out.shape == (8, 3)


def test_bigru_backward_pass_no_nan():
    model = baselines.BiGRUClassifier(in_channels=4, hidden_size=16,
                                       num_layers=1, num_classes=3)
    x = torch.randn(4, 50, 4)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_bigru_param_count_grows_with_hidden():
    small = baselines.BiGRUClassifier(in_channels=4, hidden_size=8, num_layers=1)
    big = baselines.BiGRUClassifier(in_channels=4, hidden_size=64, num_layers=1)
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    assert n_big > n_small


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_smoke_train_baseline_writes_result(tmp_path: Path):
    """End-to-end smoke: 1-epoch train on the real data, expect result.json."""
    spec = {
        "name": "smoke_test",
        "model": {"family": "bigru", "hidden_size": 8, "num_layers": 1, "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
    }
    result = baselines.train_baseline(spec, data_root=DATA_ROOT, out_dir=tmp_path)
    assert (tmp_path / "result.json").exists()
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert persisted["best_val_metrics"]["balanced_acc"] >= 0.0
    assert persisted["best_val_metrics"]["balanced_acc"] <= 1.0
    assert "confusion_3x3" in persisted["best_val_metrics"]
    assert persisted["param_count"] > 0
