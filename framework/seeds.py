"""Multi-seed initialization.

Returns the 8 seed program specs:

HIP-C originals (ratified 2026-05-11):
  1. 1D-CNN ResNet-style
  2. BiGRU baseline (existing ai4pain.baselines)
  3. Lightweight Transformer encoder
  4. Multi-stream BiGRU (per-channel encoder + late fusion)
  5. MINIROCKET + RidgeClassifierCV (Dempster, Schmidt, Webb 2020,
     arxiv:2012.08791).

iter_0012 expansion (literature sprawl, 2026-05-11 late):
  6. EDA decomposition MLP (cvxEDA tonic+phasic + HRV stats + aux stats -> MLP).
     Greco et al. 2016 cvxEDA + Xia et al. 2024 HRV-feature approach.
  7. Spectrogram 2D-CNN (STFT per channel -> stack -> small 2D CNN).
     Sriram Kumar et al. 2024 (CWT + VGG16, 86% multimodal).
  8. HRV features MLP (BVP peaks -> RMSSD/SDNN/LF/HF + aux stats -> MLP).
     Xia et al. 2024 stress detection on HRV features.

HIP-C decision (Vignan, 2026-05-11): dropped the original Catch22+XGB and
Catch22+LightGBM seeds. Rationale: hand-crafted-feature + tree-boosting
pipelines have repeatedly underperformed on this peripheral-signal task in
prior work; the search should bias toward end-to-end learned features.
MINIROCKET stays because its kernels are random-conv (neural-adjacent) and
the ridge classifier is a counter-baseline for the otherwise-neural pool.

NOTE: Only the bigru family is currently runnable end-to-end (its render
entry point lives in framework.render.FAMILY_ENTRY_POINTS). The other 4
specs will be rejected by render.render_spec_to_code until their entry
points are added. That's intentional: the search starts narrow and widens
as we implement more model families.

`diversify_population` distributes seeds across N islands, filling islands
with fewer seeds by replicating the assigned seed (the loop's first round
of mutation will diversify them in-place).
"""


def default_seed_specs() -> list[dict]:
    """The 6 locked seed specs per FRAMEWORK.md Section 9 decision 4."""
    return [
        {
            "name": "seed_1d_cnn_resnet",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "1d_cnn", "depth": 4,
                      "base_channels": 32, "kernel_size": 7,
                      "use_residual": True},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_bigru",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "bigru", "hidden_size": 64,
                      "num_layers": 1, "dropout": 0.2},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_lightweight_transformer",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "transformer", "d_model": 64,
                      "num_heads": 4, "num_layers": 2, "ff_dim": 128,
                      "dropout": 0.1},
            "training": {"loss": "ce_class_balanced", "optimizer": "adamw",
                         "lr": 5e-4, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            "name": "seed_multi_stream_bigru",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": None,
            "model": {"family": "multi_stream_bigru",
                      "per_channel_hidden": 32, "per_channel_layers": 1,
                      "fusion": "late_concat", "fusion_dropout": 0.2},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 20, "batch_size": 32, "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            # MINIROCKET (Dempster, Schmidt, Webb 2020, arxiv:2012.08791).
            # Fixed random convolutional kernels (~9996 features) transform the
            # multivariate time series; RidgeClassifierCV picks alpha via CV.
            # Multivariate inputs handled by per-channel kernel application
            # then concatenation. No learned features, no SGD.
            "name": "seed_minirocket",
            "preprocessing": {"normalize": "per_channel_zscore",
                              "padding": "right_zero_to_global_max"},
            "feature_extraction": {"family": "minirocket",
                                    "num_features": 9996,
                                    "per_channel": True,
                                    "random_state": 42},
            "model": {"family": "ridge_classifier_cv",
                      "alphas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                      "class_weight": "balanced"},
            "training": {"loss": "ridge_regression_cv",
                         "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            # cvxEDA tonic+phasic decomposition + HRV-from-BVP + aux stats.
            # Greco et al. 2016 (cvxEDA) + Xia et al. 2024 (HRV features).
            "name": "seed_eda_decomp_mlp",
            "preprocessing": {"normalize": "per_feature_zscore"},
            "feature_extraction": {"family": "cvx_eda_decomp", "fs": 100,
                                    "tau0": 2.0, "tau1": 0.7},
            "model": {"family": "eda_decomp_mlp", "hidden": 64,
                      "dropout": 0.2},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 50, "batch_size": 32,
                         "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            # STFT per channel -> stack to (C, F, T') -> small 2D CNN.
            # Sriram Kumar et al. 2024 multimodal pain/affect (CWT + VGG16).
            "name": "seed_spectrogram_cnn2d",
            "preprocessing": {"normalize": "per_channel_zscore"},
            "feature_extraction": {"family": "spectrogram", "fs": 100,
                                    "nperseg": 64, "noverlap": 32,
                                    "log_scale": True},
            "model": {"family": "spectrogram_cnn2d", "base_channels": 16,
                      "depth": 2, "dropout": 0.2},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 30, "batch_size": 32,
                         "seed": 42},
            "decode": {"strategy": "argmax"},
        },
        {
            # BVP peak detection -> HRV time+freq features -> MLP.
            # Xia et al. 2024 hit 98%+ on stress with HRV features.
            "name": "seed_hrv_features_mlp",
            "preprocessing": {"normalize": "per_feature_zscore"},
            "feature_extraction": {"family": "hrv_features", "fs": 100},
            "model": {"family": "hrv_features_mlp", "hidden": 64,
                      "dropout": 0.2},
            "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                         "lr": 1e-3, "epochs": 50, "batch_size": 32,
                         "seed": 42},
            "decode": {"strategy": "argmax"},
        },
    ]


def diversify_population(seed_specs: list[dict],
                          island_count: int) -> list[list[dict]]:
    """Distribute the seeds across `island_count` islands.

    If island_count >= len(seed_specs): one seed per island, remaining islands
    get a copy of a randomly cycled seed.
    If island_count < len(seed_specs): pack multiple seeds per island in a
    round-robin.

    Returns a list of length island_count; each element is a list of seed
    spec dicts assigned to that island.
    """
    if island_count < 1:
        raise ValueError(f"island_count must be >= 1, got {island_count}")
    if not seed_specs:
        return [[] for _ in range(island_count)]

    islands: list[list[dict]] = [[] for _ in range(island_count)]
    for i, spec in enumerate(seed_specs):
        islands[i % island_count].append(dict(spec))

    # Fill empty islands by copying from the most-populated ones (deterministic).
    n_seeds = len(seed_specs)
    if island_count > n_seeds:
        for j in range(n_seeds, island_count):
            source_idx = j % n_seeds
            islands[j].append(dict(seed_specs[source_idx]))

    return islands
