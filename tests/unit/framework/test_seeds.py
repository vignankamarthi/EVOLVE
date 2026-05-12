"""Tests for framework.seeds. Spec: FRAMEWORK.md Section 9 decision 4."""
import pytest
from framework import seeds


def test_module_imports():
    assert callable(seeds.default_seed_specs)
    assert callable(seeds.diversify_population)


def test_default_seed_specs_returns_eight_seeds():
    """iter_0012 expanded the seed pool from 5 to 8 via literature sprawl:
    cvxEDA decomp, spectrogram 2D CNN, HRV features. HIP-C cut (Catch22+gbm)
    remains permanent.
    """
    specs = seeds.default_seed_specs()
    assert len(specs) == 8


def test_default_seed_families_match_post_iter_0012():
    specs = seeds.default_seed_specs()
    families = {s["model"]["family"] for s in specs}
    expected = {"1d_cnn", "bigru", "transformer",
                "multi_stream_bigru", "ridge_classifier_cv",
                "eda_decomp_mlp", "spectrogram_cnn2d", "hrv_features_mlp"}
    assert families == expected


def test_catch22_seeds_no_longer_present():
    """HIP-C cut: Catch22+xgb and Catch22+lightgbm are no longer in the pool."""
    names = {s["name"] for s in seeds.default_seed_specs()}
    assert "seed_catch22_xgb" not in names
    assert "seed_catch22_lightgbm" not in names


def test_each_seed_has_required_top_level_keys():
    for spec in seeds.default_seed_specs():
        for key in ("name", "preprocessing", "feature_extraction", "model",
                    "training", "decode"):
            assert key in spec


def test_minirocket_seed_uses_random_kernel_features():
    specs = {s["name"]: s for s in seeds.default_seed_specs()}
    s = specs["seed_minirocket"]
    assert s["feature_extraction"]["family"] == "minirocket"
    # Canonical MINIROCKET feature count is ~9996 (84 dilations x 119 kernels)
    assert s["feature_extraction"]["num_features"] == 9996
    assert s["model"]["family"] == "ridge_classifier_cv"
    # RidgeClassifierCV alpha grid spans 6 decades
    assert len(s["model"]["alphas"]) >= 4


def test_neural_seeds_have_no_feature_extraction():
    specs = {s["name"]: s for s in seeds.default_seed_specs()}
    for name in ("seed_bigru", "seed_1d_cnn_resnet",
                 "seed_lightweight_transformer", "seed_multi_stream_bigru"):
        assert specs[name]["feature_extraction"] is None


# ---------- diversify_population ----------

def test_diversify_population_returns_one_list_per_island():
    specs = seeds.default_seed_specs()
    distributed = seeds.diversify_population(specs, island_count=8)
    assert len(distributed) == 8


def test_diversify_population_each_island_nonempty():
    specs = seeds.default_seed_specs()
    distributed = seeds.diversify_population(specs, island_count=8)
    for island in distributed:
        assert len(island) >= 1


def test_diversify_population_rejects_invalid_island_count():
    with pytest.raises(ValueError):
        seeds.diversify_population(seeds.default_seed_specs(), island_count=0)


def test_diversify_population_handles_more_islands_than_seeds():
    specs = seeds.default_seed_specs()  # 8 seeds
    distributed = seeds.diversify_population(specs, island_count=12)
    assert len(distributed) == 12


def test_diversify_population_handles_fewer_islands_than_seeds():
    specs = seeds.default_seed_specs()  # 8 seeds
    distributed = seeds.diversify_population(specs, island_count=3)
    assert len(distributed) == 3
    total = sum(len(i) for i in distributed)
    assert total == 8


def test_diversify_population_empty_seeds():
    distributed = seeds.diversify_population([], island_count=3)
    assert len(distributed) == 3
    assert all(len(i) == 0 for i in distributed)
