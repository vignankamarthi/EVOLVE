"""Tests for ai4pain.spectrogram (STFT + 2D CNN classifier).

Family: `spectrogram_cnn2d`. STFT spectrogram per channel -> stack to
(C, F, T') -> small 2D CNN -> logits.

Why: Sriram Kumar et al. 2024 hit 86% multimodal emotion classification
with CWT + VGG16. We have zero 2D pipelines in our population; this is
an orthogonal axis of variation that migration can exploit later.
"""
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ai4pain import spectrogram as spec_mod


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()


def test_module_imports():
    assert hasattr(spec_mod, "compute_spectrogram_stack")
    assert hasattr(spec_mod, "SpectrogramCNN2D")
    assert callable(spec_mod.train_spectrogram)
    assert callable(spec_mod.run_from_dir)


def test_compute_spectrogram_stack_shape_4channel():
    """For a 1000-sample, 4-channel trial at fs=100, the spectrogram stack
    should be (C=4, F, T'). F and T' depend on nperseg / noverlap."""
    trial = np.random.default_rng(0).standard_normal((1000, 4)).astype(np.float32)
    out = spec_mod.compute_spectrogram_stack(trial, fs=100, nperseg=64, noverlap=32)
    assert out.ndim == 3
    assert out.shape[0] == 4  # channels first
    F, T_prime = out.shape[1], out.shape[2]
    assert F == 64 // 2 + 1  # rfft bins
    assert T_prime > 1


def test_compute_spectrogram_stack_dtype():
    trial = np.random.default_rng(0).standard_normal((500, 4)).astype(np.float32)
    out = spec_mod.compute_spectrogram_stack(trial, fs=100, nperseg=64, noverlap=32)
    assert out.dtype == np.float32


def test_compute_spectrogram_stack_log_scale_finite():
    """Log-scaled spectrograms should be finite (no -inf from log(0))."""
    trial = np.zeros((500, 4), dtype=np.float32)
    out = spec_mod.compute_spectrogram_stack(trial, fs=100, nperseg=64,
                                              noverlap=32, log_scale=True)
    assert np.isfinite(out).all()


def test_pad_spectrograms_to_max_time():
    """A batch of variable-T' spectrograms should pad along the time axis
    to the global maximum (zero-pad on the right)."""
    rng = np.random.default_rng(0)
    s1 = rng.standard_normal((4, 33, 10)).astype(np.float32)
    s2 = rng.standard_normal((4, 33, 15)).astype(np.float32)
    s3 = rng.standard_normal((4, 33, 12)).astype(np.float32)
    batch = spec_mod.pad_spectrograms_to_max([s1, s2, s3])
    assert batch.shape == (3, 4, 33, 15)
    assert np.allclose(batch[0, :, :, :10], s1)
    assert np.allclose(batch[0, :, :, 10:], 0.0)


def test_spectrogram_cnn2d_forward_shape():
    model = spec_mod.SpectrogramCNN2D(in_channels=4, F=33, base_channels=16,
                                       depth=2, num_classes=3)
    # batch of 8, 4 channels, F=33 freq bins, T'=20 time bins
    x = torch.randn(8, 4, 33, 20)
    out = model(x)
    assert out.shape == (8, 3)


def test_spectrogram_cnn2d_backward_no_nan():
    model = spec_mod.SpectrogramCNN2D(in_channels=4, F=33, base_channels=16,
                                       depth=2, num_classes=3)
    x = torch.randn(4, 4, 33, 15)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_smoke_train_spectrogram_writes_result(tmp_path: Path):
    spec = {
        "name": "smoke_spec",
        "preprocessing": {"normalize": "per_channel_zscore"},
        "feature_extraction": {"family": "spectrogram", "fs": 100,
                                "nperseg": 64, "noverlap": 32, "log_scale": True},
        "model": {"family": "spectrogram_cnn2d", "base_channels": 8,
                   "depth": 1, "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0,
                     "loss": "ce_class_balanced", "optimizer": "adam"},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
        "decode": {"strategy": "argmax"},
    }
    result = spec_mod.train_spectrogram(spec, data_root=DATA_ROOT, out_dir=tmp_path)
    assert (tmp_path / "result.json").exists()
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert 0.0 <= persisted["best_val_metrics"]["balanced_acc"] <= 1.0


def test_spectrogram_cnn2d_residual_forward_shape():
    """use_residual=True swaps plain conv blocks for ResNet-style _SpecResBlock.
    Forward shape must be unchanged."""
    model = spec_mod.SpectrogramCNN2D(in_channels=4, F=33, base_channels=16,
                                       depth=3, use_residual=True,
                                       num_classes=3)
    out = model(torch.randn(8, 4, 33, 20))
    assert out.shape == (8, 3)
    assert model.use_residual is True


def test_spectrogram_cnn2d_residual_backward_no_nan():
    model = spec_mod.SpectrogramCNN2D(in_channels=4, F=33, base_channels=16,
                                       depth=4, use_residual=True,
                                       num_classes=3)
    x = torch.randn(4, 4, 33, 15)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None or torch.isfinite(p.grad).all()


def test_spectrogram_residual_default_off():
    """Backward compat: use_residual defaults False so existing specs are
    unaffected."""
    model = spec_mod.SpectrogramCNN2D(in_channels=4, F=33)
    assert model.use_residual is False
