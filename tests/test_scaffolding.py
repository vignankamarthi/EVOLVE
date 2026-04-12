"""Smoke tests for the AI4Pain-2026 scaffolding.

These tests verify that every Python module imports without syntax errors and
that the three config files parse. They do NOT require any data: that is the
whole point of scaffolding-time tests. Full unit tests land in later phases as
the stubs are replaced with implementations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_src_package_imports():
    """Every module in src/ imports without raising."""
    from src import data_loader, models, tuning, evaluation  # noqa: F401
    from src import dl_models, dl_training, dl_data, utils  # noqa: F401


def test_resnet1d_forward_shape():
    """ResNet1D produces (batch, 3) logits for arbitrary in_channels."""
    import torch
    from src.dl_models import ResNet1D

    for in_channels in (2, 3, 4):
        model = ResNet1D(in_channels=in_channels, num_classes=3)
        model.eval()
        x = torch.randn(4, in_channels, 1000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 3), (
            f"ResNet1D with in_channels={in_channels} returned {out.shape}, expected (4, 3)"
        )


def test_nn2_and_nn3_are_stubs():
    """NN2SOTA and NN3Novel raise NotImplementedError at construction time.

    Raising early (in __init__ rather than forward) gives a cleaner failure
    mode when the training script tries to instantiate them before Phases 4/5
    have populated the architectures.
    """
    from src.dl_models import NN2SOTA, NN3Novel

    for cls in (NN2SOTA, NN3Novel):
        with pytest.raises(NotImplementedError, match="TBD Phase"):
            cls(in_channels=2, num_classes=3)


def test_build_model_registry():
    """build_model resolves cnn1d (real) and rejects unknowns. nn2/nn3 raise
    NotImplementedError per the stub contract."""
    from src.dl_models import build_model

    cnn = build_model(arch="cnn1d", in_channels=2, num_classes=3)
    assert cnn is not None

    for arch in ("nn2", "nn3"):
        with pytest.raises(NotImplementedError):
            build_model(arch=arch, in_channels=2, num_classes=3)

    with pytest.raises(ValueError):
        build_model(arch="unknown", in_channels=2)


def test_ablation_config_parses():
    """configs/ablation_configs.json loads and has the three expected keys."""
    path = REPO_ROOT / "configs" / "ablation_configs.json"
    with open(path) as f:
        cfg = json.load(f)
    assert set(cfg.keys()) == {"bvp_eda", "bvp_eda_resp", "all_four"}
    assert cfg["bvp_eda"]["dl_channels"] == 2
    assert cfg["bvp_eda_resp"]["dl_channels"] == 3
    assert cfg["all_four"]["dl_channels"] == 4


def test_model_config_parses():
    path = REPO_ROOT / "configs" / "model_configs.json"
    with open(path) as f:
        cfg = json.load(f)
    assert "rf" in cfg
    assert "xgb" in cfg
    assert "lgbm" in cfg
    assert "logistic_regression" in cfg


def test_feature_config_parses():
    path = REPO_ROOT / "configs" / "feature_configs.json"
    with open(path) as f:
        cfg = json.load(f)
    assert "catch22" in cfg["frameworks"]
    assert "entropy" in cfg["frameworks"]
    assert "stats" in cfg["frameworks"]
