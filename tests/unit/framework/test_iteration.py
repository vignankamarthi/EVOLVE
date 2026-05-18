"""Tests for framework.iteration (batch orchestrator).

Wraps framework.population + framework.mutation + framework.meta into a
single per-batch helper that the Claude Code session calls. Replaces the
ad-hoc per-iteration scripts.
"""
import json
from pathlib import Path

import pytest

from framework import iteration as it, ledger, render


@pytest.fixture
def seed_specs():
    """Five trivial specs representing the 5 seed families."""
    return [
        {"name": "seed_1d_cnn_resnet",
         "model": {"family": "1d_cnn"}, "training": {"seed": 42},
         "preprocessing": {}, "feature_extraction": None,
         "decode": {"strategy": "argmax"}},
        {"name": "seed_bigru",
         "model": {"family": "bigru"}, "training": {"seed": 42},
         "preprocessing": {}, "feature_extraction": None,
         "decode": {"strategy": "argmax"}},
        {"name": "seed_lightweight_transformer",
         "model": {"family": "transformer"}, "training": {"seed": 42},
         "preprocessing": {}, "feature_extraction": None,
         "decode": {"strategy": "argmax"}},
        {"name": "seed_multi_stream_bigru",
         "model": {"family": "multi_stream_bigru"}, "training": {"seed": 42},
         "preprocessing": {}, "feature_extraction": None,
         "decode": {"strategy": "argmax"}},
        {"name": "seed_minirocket",
         "model": {"family": "ridge_classifier_cv"}, "training": {"seed": 42},
         "preprocessing": {}, "feature_extraction": {"family": "minirocket"},
         "decode": {"strategy": "argmax"}},
    ]


def test_module_exports():
    assert callable(it.seed_population)
    assert callable(it.prepare_batch)
    assert callable(it.global_child_count)


def test_seed_population_one_seed_per_island(tmp_db_path, seed_specs):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    run_ids = it.seed_population(led, seed_specs, island_count=5)
    assert len(run_ids) == 5
    # Each seed got a unique run_id
    assert len(set(run_ids)) == 5
    # Each lives on its own island
    for i, rid in enumerate(run_ids):
        members = led.get_island_members(i)
        assert len(members) == 1
        assert members[0]["run_id"] == rid
    led.close()


def test_seed_population_assigns_dummy_fitness(tmp_db_path, seed_specs):
    """Seeds get a small placeholder fitness so tournament_select works."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    run_ids = it.seed_population(led, seed_specs, island_count=5)
    for rid in run_ids:
        members = sum(
            (led.get_island_members(i) for i in range(5)),
            start=[]
        )
        for m in members:
            if m["run_id"] == rid:
                assert m["fitness"] is not None
                assert "balanced_acc" in m["fitness"]


def test_prepare_batch_returns_one_entry_per_island(tmp_db_path, seed_specs):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    it.seed_population(led, seed_specs, island_count=5)
    batch = it.prepare_batch(led, island_count=5,
                              tournament_size=3, rng_seed=42)
    assert len(batch) == 5
    for entry in batch:
        assert "island_id" in entry
        assert "parent_run_id" in entry
        assert "parent_spec" in entry
        assert "prompt" in entry
    led.close()


def test_prepare_batch_parents_drawn_from_correct_islands(tmp_db_path, seed_specs):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    seed_run_ids = it.seed_population(led, seed_specs, island_count=5)
    batch = it.prepare_batch(led, island_count=5, tournament_size=3,
                              rng_seed=42)
    for entry in batch:
        # With only 1 member per island, the parent must be that member
        assert entry["parent_run_id"] == seed_run_ids[entry["island_id"]]
    led.close()


def test_global_child_count_increments(tmp_db_path, seed_specs):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    assert it.global_child_count(led) == 0
    it.seed_population(led, seed_specs, island_count=5)
    # Seeds count too (they're the first 5 mutation_traces)
    assert it.global_child_count(led) == 5
    led.close()


def test_prepare_batch_prompt_is_markdown(tmp_db_path, seed_specs):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    it.seed_population(led, seed_specs, island_count=5)
    batch = it.prepare_batch(led, island_count=5, tournament_size=3,
                              rng_seed=42)
    for entry in batch:
        prompt = entry["prompt"]
        assert isinstance(prompt, str)
        assert "## Parent program" in prompt
        assert "## Meta-stochastic state" in prompt
    led.close()


def test_prepare_batch_composite_scoring_uses_novelty_alpha(tmp_db_path):
    """When meta_state.novelty_alpha is set, prepare_batch should pick parents
    via fitness.scalar_score (composite of pareto + novelty + accuracy + ECE),
    not raw balanced_acc. With alpha low (heavy novelty weight), the winner
    can differ from the raw-accuracy winner."""
    import numpy as np
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    spec = {"model": {"family": "bigru"}}

    # Two members on island 0: A has higher acc but identical confusion to others
    # (so low novelty); B has slightly lower acc but a different confusion.
    rid_a = led.allocate_run_id()
    led.write_experiment(rid_a, spec, parent_id=None, island_id=0)
    led.write_result(rid_a, {
        "balanced_acc": 0.45,
        "confusion_3x3": [[10, 1, 1], [1, 10, 1], [1, 1, 10]],
        "ece": 0.05, "param_count": 1000, "generalization_gap": 0.0,
    })

    rid_b = led.allocate_run_id()
    led.write_experiment(rid_b, spec, parent_id=None, island_id=0)
    led.write_result(rid_b, {
        "balanced_acc": 0.40,
        "confusion_3x3": [[5, 5, 2], [4, 6, 2], [3, 4, 5]],
        "ece": 0.05, "param_count": 1000, "generalization_gap": 0.0,
    })

    # Add a third "reference" member that's similar to A (so A has low novelty
    # vs the population, B has high novelty).
    rid_c = led.allocate_run_id()
    led.write_experiment(rid_c, spec, parent_id=None, island_id=0)
    led.write_result(rid_c, {
        "balanced_acc": 0.44,
        "confusion_3x3": [[10, 1, 1], [1, 10, 1], [1, 1, 10]],
        "ece": 0.05, "param_count": 1000, "generalization_gap": 0.0,
    })

    # With composite scoring + heavy novelty weight (alpha=0.1), B can win
    # despite lower accuracy.
    meta_low_alpha = {"p_lit": 0.5, "novelty_alpha": 0.1,
                      "temperature": 0.7, "failure_boost_active": False}
    won_b_at_least_once = False
    for seed in range(20):
        batch = it.prepare_batch(led, island_count=1, tournament_size=3,
                                  rng_seed=seed, meta_state=meta_low_alpha,
                                  composite_scoring=True)
        if batch[0]["parent_run_id"] == rid_b:
            won_b_at_least_once = True
            break
    assert won_b_at_least_once, "novelty-weighted tournament never picked B"
    led.close()


def test_prepare_batch_composite_off_uses_raw_accuracy(tmp_db_path):
    """When composite_scoring=False (default), highest balanced_acc always wins."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    spec = {"model": {"family": "bigru"}}
    rid_high = led.allocate_run_id()
    led.write_experiment(rid_high, spec, parent_id=None, island_id=0)
    led.write_result(rid_high, {"balanced_acc": 0.50,
                                "confusion_3x3": [[10, 0, 0]] * 3})
    rid_low = led.allocate_run_id()
    led.write_experiment(rid_low, spec, parent_id=None, island_id=0)
    led.write_result(rid_low, {"balanced_acc": 0.40,
                               "confusion_3x3": [[10, 0, 0]] * 3})
    # tournament_size >= island size => deterministic
    batch = it.prepare_batch(led, island_count=1, tournament_size=2,
                              rng_seed=0)
    assert batch[0]["parent_run_id"] == rid_high
    led.close()


# --- Phase X: wired breakdown + meta-stochastic mechanisms ---


@pytest.fixture
def minimal_specs():
    """5 minimal specs for tests that don't care about details."""
    return [
        {"name": f"seed_{i}",
         "model": {"family": "bigru"},
         "training": {"seed": 42},
         "preprocessing": {}, "feature_extraction": None,
         "decode": {"strategy": "argmax"}}
        for i in range(5)
    ]


def test_step_meta_state_with_genome_defaults_persists_l2_mutation(tmp_db_path,
                                                                     minimal_specs):
    """Level 2 mutates genome novelty_alpha to 0.45; relaxation must pull
    toward 0.45, not the hardcoded 0.3 baseline."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    it.seed_population(led, minimal_specs, island_count=5)
    # Seed meta_state at the genome's post-L2 values
    led.write_meta_state(iteration=1, p_lit=0.5, novelty_alpha=0.45,
                          temperature=0.7,
                          failure_boost={"failure_boost_active": False})
    # Now run prepare_batch with evolve_meta=True; novelty_alpha should
    # relax toward 0.45 (genome default), NOT toward 0.3.
    it.prepare_batch(led, island_count=5, tournament_size=3, rng_seed=42,
                      evolve_meta=True)
    state = led.read_latest_meta_state()
    # Relaxation rate is 0.1, so new = 0.45 + 0.1*(0.45 - 0.45) = 0.45 exactly
    assert abs(state["novelty_alpha"] - 0.45) < 0.01
    led.close()


def test_prepare_batch_persists_meta_state(tmp_db_path, minimal_specs):
    """Each prepare_batch call writes a new meta_state row reflecting the
    stepped (drifted p_lit + failure_boost-updated) state."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    it.seed_population(led, minimal_specs, island_count=5)
    assert led.read_latest_meta_state() is None
    it.prepare_batch(led, island_count=5, tournament_size=3,
                      rng_seed=42, evolve_meta=True)
    state1 = led.read_latest_meta_state()
    assert state1 is not None
    assert "p_lit" in state1
    assert "novelty_alpha" in state1
    # Second batch produces a new (possibly drifted) row at higher iteration
    it.prepare_batch(led, island_count=5, tournament_size=3,
                      rng_seed=43, evolve_meta=True)
    state2 = led.read_latest_meta_state()
    assert state2["iteration"] > state1["iteration"]
    led.close()


def test_prepare_batch_escalates_per_island_stagnation(tmp_db_path):
    """When an island's last_improvement gap exceeds patience, the entry's
    local `meta` has bumped novelty_alpha + temperature."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    spec = {"model": {"family": "bigru"}, "training": {"seed": 42},
            "preprocessing": {}, "feature_extraction": None,
            "decode": {"strategy": "argmax"}}
    # Island 0 has a "stale" winner (no improvement for many iters)
    rid = led.allocate_run_id()
    led.write_experiment(rid, spec, parent_id=None, island_id=0)
    led.write_result(rid, {"balanced_acc": 0.42,
                           "confusion_3x3": [[10, 0, 0]] * 3})
    led.write_mutation_trace(iteration=1, run_id=rid, parent_run_ids=[],
                              prompt_context="", child_spec=spec,
                              fingerprint="fp_seed_0",
                              reasoning_summary="", accepted=True)
    # 30 children later, no improvement on island 0
    for i in range(2, 32):
        rid_i = led.allocate_run_id()
        led.write_experiment(rid_i, spec, parent_id=rid, island_id=0)
        led.write_result(rid_i, {"balanced_acc": 0.40,
                                 "confusion_3x3": [[10, 0, 0]] * 3})
        led.write_mutation_trace(iteration=i, run_id=rid_i,
                                  parent_run_ids=[rid],
                                  prompt_context="", child_spec=spec,
                                  fingerprint=f"fp_{i}", reasoning_summary="",
                                  accepted=True)

    base_meta = {"p_lit": 0.5, "novelty_alpha": 0.3, "temperature": 0.7,
                 "failure_boost_active": False}
    batch = it.prepare_batch(led, island_count=1, tournament_size=2,
                              rng_seed=42, meta_state=base_meta,
                              stagnation_patience=5, evolve_meta=False)
    entry = batch[0]
    # novelty_alpha should be raised; temperature too
    assert entry["meta"].novelty_alpha > 0.3
    assert entry["meta"].temperature > 0.7
    assert entry.get("stagnant") is True
    led.close()


def test_prepare_batch_triggers_migration_on_long_stagnation(tmp_db_path):
    """When island stagnant past migration_patience, entry carries a
    foreign_parent_run_id from a different island."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    spec = {"model": {"family": "bigru"}, "training": {"seed": 42},
            "preprocessing": {}, "feature_extraction": None,
            "decode": {"strategy": "argmax"}}
    # Island 0: stagnant
    rid_a = led.allocate_run_id()
    led.write_experiment(rid_a, spec, parent_id=None, island_id=0)
    led.write_result(rid_a, {"balanced_acc": 0.40,
                             "confusion_3x3": [[10, 0, 0]] * 3})
    led.write_mutation_trace(iteration=1, run_id=rid_a, parent_run_ids=[],
                              prompt_context="", child_spec=spec,
                              fingerprint="fp_a", reasoning_summary="",
                              accepted=True)
    # Add many flat-fitness children to island 0
    for i in range(2, 30):
        rid_i = led.allocate_run_id()
        led.write_experiment(rid_i, spec, parent_id=rid_a, island_id=0)
        led.write_result(rid_i, {"balanced_acc": 0.40,
                                 "confusion_3x3": [[10, 0, 0]] * 3})
        led.write_mutation_trace(iteration=i, run_id=rid_i, parent_run_ids=[rid_a],
                                  prompt_context="", child_spec=spec,
                                  fingerprint=f"fp_island0_{i}",
                                  reasoning_summary="", accepted=True)
    # Island 1: healthy
    rid_b = led.allocate_run_id()
    led.write_experiment(rid_b, spec, parent_id=None, island_id=1)
    led.write_result(rid_b, {"balanced_acc": 0.55,
                             "confusion_3x3": [[15, 0, 0]] * 3})
    led.write_mutation_trace(iteration=30, run_id=rid_b, parent_run_ids=[],
                              prompt_context="", child_spec=spec,
                              fingerprint="fp_b", reasoning_summary="",
                              accepted=True)

    batch = it.prepare_batch(led, island_count=2, tournament_size=2,
                              rng_seed=42,
                              migration_patience=20, evolve_meta=False)
    island0 = next(e for e in batch if e["island_id"] == 0)
    assert island0.get("migrated_from_island") == 1
    assert island0.get("foreign_parent_run_id") == rid_b
    assert island0.get("foreign_parent_spec") is not None
    led.close()


def test_prepare_batch_evolves_critics(tmp_db_path, minimal_specs):
    """Each prepare_batch with evolve_critics=True writes critic_population
    rows; first call seeds, subsequent calls add 1 critic each."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    it.seed_population(led, minimal_specs, island_count=5)
    assert len(led.read_critic_population()) == 0
    it.prepare_batch(led, island_count=5, tournament_size=3,
                      rng_seed=42, evolve_critics=True,
                      critic_pop_size=5)
    pop1 = led.read_critic_population()
    assert len(pop1) >= 1  # at least seeded
    it.prepare_batch(led, island_count=5, tournament_size=3,
                      rng_seed=43, evolve_critics=True,
                      critic_pop_size=5)
    pop2 = led.read_critic_population()
    assert len(pop2) >= len(pop1)  # population grew or evolved
    led.close()


def test_prepare_batch_prompt_includes_critic_when_present(tmp_db_path, minimal_specs):
    """When critic population is non-empty, mutation prompt should include
    a 'Hard cases' or 'Critic' section so Claude knows what to target."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    it.seed_population(led, minimal_specs, island_count=5)
    # Seed a critic manually
    led.write_critic(critic_id="c_test", parent_id=None,
                      genome={"subject_subset": [11, 15],
                              "signal_perturbation": {},
                              "channel_permutation": [0, 1, 2, 3]},
                      fitness=0.5)
    batch = it.prepare_batch(led, island_count=5, tournament_size=3,
                              rng_seed=42, include_critic_in_prompt=True)
    # At least one entry's prompt should mention critics
    found = any("Hard case" in e["prompt"] or "critic" in e["prompt"].lower()
                for e in batch)
    assert found, "no critic context in prompts"
    led.close()


# ---------- family_quota: hard diversity constraint (iter_0015 rebuild) ----------

import random as _random
from framework.constraints import ConstraintViolation as _CV

_ALL_FAMILIES = ["bigru", "1d_cnn", "transformer", "multi_stream_bigru",
                 "multi_stream_aux", "ridge_classifier_cv", "eda_decomp_mlp",
                 "spectrogram_cnn2d", "hrv_features_mlp"]


def test_allocate_family_slots_respects_max_per_family():
    slots = it.allocate_family_slots(
        n_children=8, available_families=_ALL_FAMILIES,
        recent_family_counts={}, rng=_random.Random(0),
        max_per_family=3, min_families=4)
    assert len(slots) == 8
    from collections import Counter
    for fam, n in Counter(slots).items():
        assert n <= 3, f"{fam} has {n} > max_per_family"


def test_allocate_family_slots_guarantees_min_families():
    slots = it.allocate_family_slots(
        n_children=8, available_families=_ALL_FAMILIES,
        recent_family_counts={}, rng=_random.Random(1),
        max_per_family=3, min_families=4)
    assert len(set(slots)) >= 4


def test_allocate_family_slots_under_weights_recent_dominant():
    """multi_stream_bigru dominated recent batches -> should be rare/absent now."""
    recent = {"multi_stream_bigru": 24}  # monoculture history
    counts_over_runs = {}
    for s in range(20):
        slots = it.allocate_family_slots(
            n_children=8, available_families=_ALL_FAMILIES,
            recent_family_counts=recent, rng=_random.Random(s),
            max_per_family=3, min_families=4)
        for f in slots:
            counts_over_runs[f] = counts_over_runs.get(f, 0) + 1
    ms = counts_over_runs.get("multi_stream_bigru", 0)
    # a never-used family should be picked more often than the dominant one
    fresh = counts_over_runs.get("eda_decomp_mlp", 0)
    assert fresh > ms


def test_allocate_family_slots_rejects_impossible_quota():
    with pytest.raises(ValueError):
        it.allocate_family_slots(
            n_children=8, available_families=["bigru", "1d_cnn"],
            recent_family_counts={}, rng=_random.Random(0),
            max_per_family=3, min_families=4)


def test_validate_batch_family_quota_passes_legal_manifest():
    manifest = {"experiments": [
        {"family": "bigru"}, {"family": "bigru"}, {"family": "bigru"},
        {"family": "transformer"}, {"family": "transformer"},
        {"family": "spectrogram_cnn2d"}, {"family": "hrv_features_mlp"},
        {"family": "multi_stream_aux"},
    ]}
    assert it.validate_batch_family_quota(manifest, max_per_family=3,
                                           min_families=4) is None


def test_validate_batch_family_quota_rejects_over_cap():
    manifest = {"experiments": [
        {"family": "multi_stream_bigru"}] * 4 + [
        {"family": "bigru"}, {"family": "transformer"},
        {"family": "hrv_features_mlp"}, {"family": "eda_decomp_mlp"}]}
    v = it.validate_batch_family_quota(manifest, max_per_family=3,
                                        min_families=4)
    assert isinstance(v, _CV)


def test_validate_batch_family_quota_rejects_too_few_families():
    manifest = {"experiments": [
        {"family": "bigru"}, {"family": "bigru"}, {"family": "bigru"},
        {"family": "transformer"}, {"family": "transformer"},
        {"family": "transformer"}, {"family": "1d_cnn"}, {"family": "1d_cnn"}]}
    v = it.validate_batch_family_quota(manifest, max_per_family=3,
                                        min_families=4)
    assert isinstance(v, _CV)


def test_assign_blender_flags_marks_repeated_families():
    slots = ["bigru", "bigru", "bigru", "transformer", "transformer",
             "spectrogram_cnn2d", "hrv_features_mlp", "multi_stream_aux"]
    flags = it.assign_blender_flags(slots)
    assert flags == [True, True, True, True, True, False, False, False]
