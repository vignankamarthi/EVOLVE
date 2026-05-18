"""Tests for ai4pain.multi_stream (MultiStreamNet + legacy MultiStreamBiGRU)."""
import torch
import pytest

from ai4pain import multi_stream


# ---------- legacy MultiStreamBiGRU (backward compat) ----------

def test_module_imports():
    assert hasattr(multi_stream, "MultiStreamBiGRU")
    assert hasattr(multi_stream, "MultiStreamNet")
    assert callable(multi_stream.run_from_dir)


def test_multi_stream_forward_shape():
    model = multi_stream.MultiStreamBiGRU(
        in_channels=4, per_channel_hidden=16, per_channel_layers=1,
        fusion="late_concat", fusion_dropout=0.0, num_classes=3,
    )
    x = torch.randn(8, 200, 4)
    out = model(x)
    assert out.shape == (8, 3)


def test_multi_stream_has_one_encoder_per_channel():
    model = multi_stream.MultiStreamBiGRU(in_channels=4, per_channel_hidden=8)
    assert len(model.encoders) == 4


def test_multi_stream_rejects_unknown_fusion():
    with pytest.raises(ValueError):
        multi_stream.MultiStreamBiGRU(in_channels=4, fusion="early_concat")


def test_multi_stream_handles_short_sequence():
    model = multi_stream.MultiStreamBiGRU(
        in_channels=4, per_channel_hidden=8, per_channel_layers=1,
        fusion="late_concat", fusion_dropout=0.0, num_classes=3,
    )
    out = model(torch.randn(2, 20, 4))
    assert out.shape == (2, 3)


def test_legacy_bigru_is_gru_late_concat():
    """MultiStreamBiGRU is a MultiStreamNet with encoder_type=gru, fusion=late_concat."""
    m = multi_stream.MultiStreamBiGRU(in_channels=4, per_channel_hidden=8)
    assert isinstance(m, multi_stream.MultiStreamNet)
    assert m.encoder_type == "gru"
    assert m.fusion == "late_concat"


# ---------- MultiStreamNet: encoder_type sweep ----------

ENCODER_TYPES = ["gru", "lstm", "bilstm", "transformer", "conv1d"]
FUSION_MODES = ["late_concat", "mean_pool", "max_pool", "attention_pool",
                "mid_fusion"]


@pytest.mark.parametrize("enc", ENCODER_TYPES)
def test_net_forward_shape_per_encoder(enc):
    model = multi_stream.MultiStreamNet(
        in_channels=4, per_channel_hidden=16, per_channel_layers=1,
        encoder_type=enc, fusion="late_concat", fusion_dropout=0.0,
        num_classes=3,
    )
    out = model(torch.randn(6, 128, 4))
    assert out.shape == (6, 3)


@pytest.mark.parametrize("enc", ENCODER_TYPES)
def test_net_backward_no_nan_per_encoder(enc):
    model = multi_stream.MultiStreamNet(
        in_channels=4, per_channel_hidden=16, per_channel_layers=1,
        encoder_type=enc, fusion="late_concat", num_classes=3,
    )
    x = torch.randn(4, 64, 4)
    y = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None or torch.isfinite(p.grad).all()


# ---------- MultiStreamNet: fusion sweep ----------

@pytest.mark.parametrize("fusion", FUSION_MODES)
def test_net_forward_shape_per_fusion(fusion):
    model = multi_stream.MultiStreamNet(
        in_channels=4, per_channel_hidden=16, per_channel_layers=1,
        encoder_type="gru", fusion=fusion, fusion_dropout=0.0, num_classes=3,
    )
    out = model(torch.randn(6, 96, 4))
    assert out.shape == (6, 3)


def test_attention_pool_weights_sum_to_one():
    model = multi_stream.MultiStreamNet(
        in_channels=4, per_channel_hidden=16, encoder_type="gru",
        fusion="attention_pool", num_classes=3,
    )
    model.eval()
    _ = model(torch.randn(5, 64, 4))
    w = model.last_attn_weights  # (B, C)
    assert w is not None
    assert w.shape == (5, 4)
    sums = w.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_net_rejects_unknown_encoder():
    with pytest.raises(ValueError):
        multi_stream.MultiStreamNet(in_channels=4, encoder_type="quantum_rnn")


def test_net_rejects_unknown_fusion():
    with pytest.raises(ValueError):
        multi_stream.MultiStreamNet(in_channels=4, fusion="early_concat")


def test_net_param_count_grows_with_hidden():
    small = multi_stream.MultiStreamNet(in_channels=4, per_channel_hidden=8,
                                         encoder_type="gru")
    big = multi_stream.MultiStreamNet(in_channels=4, per_channel_hidden=48,
                                       encoder_type="gru")
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    assert n_big > n_small


def test_factory_reads_encoder_type_and_fusion():
    """_multi_stream_factory honors encoder_type + fusion from model_cfg,
    defaulting to gru / late_concat for legacy specs."""
    cfg = {"per_channel_hidden": 16, "encoder_type": "lstm",
           "fusion": "attention_pool"}
    m = multi_stream._multi_stream_factory(
        in_channels=4, T_max=100, model_cfg=cfg, num_classes=3)
    assert m.encoder_type == "lstm"
    assert m.fusion == "attention_pool"
    # legacy spec with no encoder_type -> gru / late_concat
    legacy = multi_stream._multi_stream_factory(
        in_channels=4, T_max=100, model_cfg={"per_channel_hidden": 16},
        num_classes=3)
    assert legacy.encoder_type == "gru"
    assert legacy.fusion == "late_concat"
