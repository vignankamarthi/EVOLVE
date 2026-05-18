"""Tests for framework.mutation. Spec: FRAMEWORK.md Section 2 + 6."""
import pytest
from framework import mutation


def test_module_imports():
    assert mutation.MetaState is not None
    assert callable(mutation.assemble_mutation_prompt)


def test_metastate_dataclass_fields():
    m = mutation.MetaState(p_lit=0.5, novelty_alpha=0.3, temperature=0.7,
                           failure_boost_active=False)
    assert m.p_lit == 0.5
    assert m.failure_boost_active is False


def _baseline_meta(**overrides) -> mutation.MetaState:
    defaults = dict(p_lit=0.5, novelty_alpha=0.3, temperature=0.7,
                    failure_boost_active=False)
    defaults.update(overrides)
    return mutation.MetaState(**defaults)


def test_prompt_contains_all_required_sections(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[sample_program_spec],
        recent_failures=[],
        meta=_baseline_meta(),
    )
    assert isinstance(out, str)
    for section in (
        "## Parent program",
        "## Best in island",
        "## Recent rejected programs",
        "## Meta-stochastic state",
        "## Mutation directive",
    ):
        assert section in out


def test_prompt_serializes_parent_spec_as_json(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(),
    )
    assert '"family": "bigru"' in out


def test_directive_aggressive_when_failure_boost_active(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(failure_boost_active=True),
    )
    assert "AGGRESSIVE" in out


def test_directive_literature_bias_when_p_lit_high(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(p_lit=0.8),
    )
    assert "literature-derived" in out


def test_directive_novel_when_p_lit_low(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(p_lit=0.25),
    )
    assert "NOVEL" in out


def test_empty_inputs_do_not_crash(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(),
    )
    assert "(empty)" in out
    assert "(none recorded)" in out


def test_meta_state_values_appear_in_prompt(sample_program_spec):
    out = mutation.assemble_mutation_prompt(
        parent_spec=sample_program_spec,
        island_best_specs=[],
        recent_failures=[],
        meta=_baseline_meta(p_lit=0.42, novelty_alpha=0.61, temperature=1.1),
    )
    assert "0.420" in out
    assert "0.610" in out
    assert "1.100" in out


# ---------- graft_family crossover primitive (iter_0015 rebuild) ----------

import random as _random


def _seq_parent():
    return {
        "name": "parent_seq",
        "preprocessing": {"normalize": "per_channel_zscore",
                           "padding": "right_zero_to_global_max"},
        "feature_extraction": None,
        "model": {"family": "multi_stream_bigru", "per_channel_hidden": 40},
        "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                      "lr": 1e-3, "epochs": 110, "seed": 1},
        "decode": {"strategy": "argmax"},
    }


def _spectrogram_parent():
    return {
        "name": "parent_spec",
        "preprocessing": {"normalize": "per_channel_zscore"},
        "feature_extraction": {"family": "spectrogram", "fs": 100,
                                "nperseg": 64},
        "model": {"family": "spectrogram_cnn2d", "base_channels": 16},
        "training": {"loss": "ce_class_balanced", "optimizer": "adam",
                      "lr": 1e-3, "epochs": 30, "seed": 2},
        "decode": {"strategy": "argmax"},
    }


def _ridge_parent():
    return {
        "name": "parent_ridge",
        "preprocessing": {"normalize": "per_channel_zscore"},
        "feature_extraction": {"family": "minirocket", "num_features": 9996},
        "model": {"family": "ridge_classifier_cv", "alphas": [0.1, 1.0]},
        "training": {"loss": "ridge_regression_cv", "seed": 3},
        "decode": {"strategy": "argmax"},
    }


def test_graft_family_exists():
    assert callable(mutation.graft_family)
    assert callable(mutation.predict_graft_coherence)


def test_graft_family_picks_genes_from_both_parents():
    """Across many seeds, each gene should sometimes come from A, sometimes B."""
    a, b = _seq_parent(), _spectrogram_parent()
    model_families = set()
    for s in range(40):
        child = mutation.graft_family(a, b, _random.Random(s))
        model_families.add(child["model"]["family"])
    # both parents' model families should appear
    assert "multi_stream_bigru" in model_families
    assert "spectrogram_cnn2d" in model_families


def test_graft_family_deterministic_given_rng():
    a, b = _seq_parent(), _spectrogram_parent()
    c1 = mutation.graft_family(a, b, _random.Random(123))
    c2 = mutation.graft_family(a, b, _random.Random(123))
    assert c1["model"] == c2["model"]
    assert c1["training"] == c2["training"]


def test_graft_family_tags_coherence():
    a, b = _seq_parent(), _spectrogram_parent()
    child = mutation.graft_family(a, b, _random.Random(7))
    assert "graft_coherence" in child
    assert isinstance(child["graft_coherence"], bool)
    assert "reasoning_summary" in child


def test_coherence_flags_spectrogram_prep_with_sequence_model():
    """spectrogram feature_extraction + a sequence model = incoherent."""
    incoherent = {
        "preprocessing": {"normalize": "per_channel_zscore"},
        "feature_extraction": {"family": "spectrogram", "fs": 100},
        "model": {"family": "multi_stream_bigru"},
        "training": {"loss": "ce_class_balanced"},
        "decode": {"strategy": "argmax"},
    }
    assert mutation.predict_graft_coherence(incoherent) is False


def test_coherence_flags_ridge_training_with_neural_model():
    """ridge_regression_cv training + a neural model = incoherent."""
    incoherent = {
        "preprocessing": {"normalize": "per_channel_zscore"},
        "feature_extraction": None,
        "model": {"family": "multi_stream_bigru"},
        "training": {"loss": "ridge_regression_cv"},
        "decode": {"strategy": "argmax"},
    }
    assert mutation.predict_graft_coherence(incoherent) is False


def test_coherence_accepts_well_formed_sequence_spec():
    assert mutation.predict_graft_coherence(_seq_parent()) is True


def test_coherence_accepts_well_formed_ridge_spec():
    assert mutation.predict_graft_coherence(_ridge_parent()) is True
