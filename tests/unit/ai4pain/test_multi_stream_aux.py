"""Tests for ai4pain.multi_stream_aux (family: multi_stream_aux).

A MultiStreamNet body fused with a 26-dim HRV feature side-input. The
`add_aux_stream` architectural mutation produces children of this family.
The HRV vector is computed per trial via ai4pain.hrv.compute_per_trial_features.
"""
import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
from ai4pain import multi_stream_aux
from ai4pain.hrv import HRV_FEATURE_DIM


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()


def test_module_imports():
    assert hasattr(multi_stream_aux, "MultiStreamAuxNet")
    assert callable(multi_stream_aux.train_multi_stream_aux)
    assert callable(multi_stream_aux.run_from_dir)


def test_aux_forward_shape():
    model = multi_stream_aux.MultiStreamAuxNet(
        in_channels=4, per_channel_hidden=16, encoder_type="gru",
        fusion="late_concat", hrv_dim=HRV_FEATURE_DIM, num_classes=3)
    x = torch.randn(8, 120, 4)
    hrv = torch.randn(8, HRV_FEATURE_DIM)
    out = model(x, hrv)
    assert out.shape == (8, 3)


def test_aux_backward_no_nan():
    model = multi_stream_aux.MultiStreamAuxNet(
        in_channels=4, per_channel_hidden=16, hrv_dim=HRV_FEATURE_DIM,
        num_classes=3)
    x = torch.randn(4, 80, 4)
    hrv = torch.randn(4, HRV_FEATURE_DIM)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x, hrv), y)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None or torch.isfinite(p.grad).all()


def test_aux_has_hrv_branch():
    """The HRV side-MLP must exist and add parameters beyond the body."""
    model = multi_stream_aux.MultiStreamAuxNet(
        in_channels=4, per_channel_hidden=16, hrv_dim=HRV_FEATURE_DIM)
    assert hasattr(model, "hrv_mlp")
    hrv_params = sum(p.numel() for p in model.hrv_mlp.parameters())
    assert hrv_params > 0


def test_aux_honors_encoder_and_fusion_swaps():
    """add_aux_stream stacks on top of swap_encoder / swap_fusion."""
    model = multi_stream_aux.MultiStreamAuxNet(
        in_channels=4, per_channel_hidden=16, encoder_type="lstm",
        fusion="attention_pool", hrv_dim=HRV_FEATURE_DIM)
    assert model.body.encoder_type == "lstm"
    assert model.body.fusion == "attention_pool"


def test_aux_uses_all_four_channels():
    """Hard guard: the aux family must ingest all 4 physiological channels."""
    model = multi_stream_aux.MultiStreamAuxNet(in_channels=4,
                                                 per_channel_hidden=8)
    assert model.body.in_channels == 4


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain data not present")
def test_smoke_train_aux_writes_result(tmp_path: Path):
    spec = {
        "name": "smoke_aux",
        "preprocessing": {"normalize": "per_channel_zscore",
                           "padding": "right_zero_to_global_max"},
        "feature_extraction": {"family": "hrv_aux", "fs": 100},
        "model": {"family": "multi_stream_aux", "per_channel_hidden": 8,
                   "encoder_type": "gru", "fusion": "late_concat"},
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0,
                     "loss": "ce_class_balanced", "optimizer": "adam"},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
        "decode": {"strategy": "argmax"},
    }
    result = multi_stream_aux.train_multi_stream_aux(
        spec, data_root=DATA_ROOT, out_dir=tmp_path)
    assert (tmp_path / "result.json").exists()
    persisted = json.loads((tmp_path / "result.json").read_text())
    assert 0.0 <= persisted["best_val_metrics"]["balanced_acc"] <= 1.0
