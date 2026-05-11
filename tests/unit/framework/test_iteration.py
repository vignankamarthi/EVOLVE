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
