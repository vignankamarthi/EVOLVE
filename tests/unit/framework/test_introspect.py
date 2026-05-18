"""Tests for framework.introspect (Level 2 self-modification).

Spec: FRAMEWORK.md Section 7 (operators + compound detector) and Section 11
(observability over ledger). Synthetic ledger fixtures drive propose_mutation
through each pathology branch.
"""
import pytest

from framework import introspect, ledger
from framework.introspect import (
    DetectorConfig,
    GenomeConstraintViolation,
    GenomeMutation,
    add_island,
    apply_genome_mutation,
    assemble_introspection_prompt,
    compute_genome_hash,
    drop_island,
    propose_mutation,
    scale_param,
    set_axis_weight,
    set_threshold,
    should_fire,
    swap_operator,
    toggle_operator,
    validate_genome_mutation,
)


DEFAULT_GENOME = {
    "island_count": 8,
    "island_size": 12,
    "reset_cadence": 50,
    "novelty_alpha": 0.3,
    "ece_lambda": 0.5,
    "max_params": 10_000_000,
    "max_train_seconds": 1800,
    "ast_tabu_k": 50,
    "lineage_cap": 5,
    "migration_patience": 10,
    "critic_pop_size": 30,
    "stagnation_patience": 10,
    "p_lit_drift_sigma": 0.05,
    "failure_boost_gain": 1.0,
    "introspection_cadence_M": 50,
    "max_per_family": 3,
    "min_families": 4,
    "axis_weights": {"balanced_acc": 1.0, "novelty": 0.3, "ece": 0.5, "gap": 0.2},
    "operators_enabled": {"failure_boost": True, "critic": True, "migration": True},
    "operator_slots": {"replacement_rule": "GENITOR"},
}


# --- Genome rule guard sanity ---


def test_module_imports():
    assert "island_count" in introspect.GENOME_RULE_GUARDS


def test_genome_rule_guards_have_sane_bounds():
    lo, hi = introspect.GENOME_RULE_GUARDS["island_count"]
    assert 1 <= lo < hi <= 64
    lo, hi = introspect.GENOME_RULE_GUARDS["introspection_cadence_M"]
    assert lo >= 10


# --- Compound detector branches ---


def test_should_fire_bootstrap_floor():
    cfg = DetectorConfig()
    assert should_fire(
        current_iter=19, last_fire_iter=0,
        median_delta=0.0, rejection_rate=0.9, entropy_ratio=0.3,
        config=cfg,
    ) is False


def test_should_fire_min_gap_chatter_guard():
    cfg = DetectorConfig(min_gap=10)
    assert should_fire(
        current_iter=25, last_fire_iter=20,
        median_delta=0.0, rejection_rate=0.9, entropy_ratio=0.3,
        config=cfg,
    ) is False


def test_should_fire_max_gap_silence_cap():
    cfg = DetectorConfig(min_gap=10, max_gap=100)
    assert should_fire(
        current_iter=121, last_fire_iter=20,
        median_delta=0.05, rejection_rate=0.1, entropy_ratio=1.0,
        config=cfg,
    ) is True


def test_should_fire_stagnation_branch():
    cfg = DetectorConfig(epsilon=0.005)
    assert should_fire(
        current_iter=50, last_fire_iter=20,
        median_delta=0.001, rejection_rate=0.1, entropy_ratio=1.0,
        config=cfg,
    ) is True


def test_should_fire_over_rejection_branch():
    cfg = DetectorConfig(max_rejection_rate=0.6)
    assert should_fire(
        current_iter=50, last_fire_iter=20,
        median_delta=0.02, rejection_rate=0.7, entropy_ratio=1.0,
        config=cfg,
    ) is True


def test_should_fire_entropy_collapse_branch():
    cfg = DetectorConfig(entropy_drop_ratio=0.7)
    assert should_fire(
        current_iter=50, last_fire_iter=20,
        median_delta=0.02, rejection_rate=0.1, entropy_ratio=0.5,
        config=cfg,
    ) is True


def test_should_fire_healthy_state_no_fire():
    cfg = DetectorConfig()
    assert should_fire(
        current_iter=50, last_fire_iter=20,
        median_delta=0.02, rejection_rate=0.1, entropy_ratio=1.0,
        config=cfg,
    ) is False


# --- Typed operators ---


def test_scale_param_in_range():
    mut = scale_param("hash_p", DEFAULT_GENOME, field="novelty_alpha", factor=1.5)
    assert isinstance(mut, GenomeMutation)
    assert abs(mut.parameter_changes["novelty_alpha"] - 0.45) < 1e-9
    assert "SCALE_PARAM" in mut.description


def test_scale_param_clamped_to_upper_guard():
    mut = scale_param("hash_p", DEFAULT_GENOME, field="novelty_alpha", factor=5.0)
    lo, hi = introspect.GENOME_RULE_GUARDS["novelty_alpha"]
    assert mut.parameter_changes["novelty_alpha"] == hi


def test_scale_param_clamped_to_lower_guard():
    mut = scale_param("hash_p", DEFAULT_GENOME, field="novelty_alpha", factor=0.0)
    lo, hi = introspect.GENOME_RULE_GUARDS["novelty_alpha"]
    assert mut.parameter_changes["novelty_alpha"] == lo


def test_scale_param_rejects_unknown_field():
    with pytest.raises(ValueError):
        scale_param("hash_p", DEFAULT_GENOME, field="not_a_field", factor=1.5)


def test_add_island_increments_count():
    mut = add_island("hash_p", DEFAULT_GENOME)
    assert mut.parameter_changes["island_count"] == 9
    assert "ADD_ISLAND" in mut.description


def test_add_island_blocked_at_cap():
    g = dict(DEFAULT_GENOME, island_count=16)
    with pytest.raises(ValueError):
        add_island("hash_p", g)


def test_drop_island_decrements_count():
    mut = drop_island("hash_p", DEFAULT_GENOME, island_id=0)
    assert mut.parameter_changes["island_count"] == 7
    assert "DROP_ISLAND" in mut.description


def test_drop_island_blocked_at_floor():
    g = dict(DEFAULT_GENOME, island_count=4)
    with pytest.raises(ValueError):
        drop_island("hash_p", g, island_id=0)


def test_set_threshold_in_range():
    # novelty_alpha is a numeric genome field with guard (0.0, 0.8)
    mut = set_threshold("hash_p", DEFAULT_GENOME,
                       name="novelty_alpha", value=0.5)
    assert mut.parameter_changes["novelty_alpha"] == 0.5
    assert "SET_THRESHOLD" in mut.description


def test_set_threshold_rejected_out_of_range():
    with pytest.raises(ValueError):
        set_threshold("hash_p", DEFAULT_GENOME,
                      name="novelty_alpha", value=1.5)


def test_set_axis_weight_updates_pareto_dict():
    mut = set_axis_weight("hash_p", DEFAULT_GENOME, axis="novelty", weight=0.6)
    changes = mut.parameter_changes
    assert changes["axis_weights"]["novelty"] == 0.6
    # other axes preserved
    assert changes["axis_weights"]["balanced_acc"] == 1.0


def test_set_axis_weight_rejected_out_of_range():
    with pytest.raises(ValueError):
        set_axis_weight("hash_p", DEFAULT_GENOME, axis="novelty", weight=1.5)


def test_toggle_operator_off():
    mut = toggle_operator("hash_p", DEFAULT_GENOME,
                          name="failure_boost", enabled=False)
    assert mut.parameter_changes["operators_enabled"]["failure_boost"] is False


def test_swap_operator_changes_slot():
    mut = swap_operator("hash_p", DEFAULT_GENOME,
                       slot="replacement_rule", name="TOURNAMENT")
    assert mut.parameter_changes["operator_slots"]["replacement_rule"] == "TOURNAMENT"


# --- Apply + validate ---


def test_apply_genome_mutation_returns_new_genome_dict():
    mut = scale_param("hash_p", DEFAULT_GENOME, field="novelty_alpha", factor=1.5)
    new = apply_genome_mutation(mut, DEFAULT_GENOME)
    assert new is not DEFAULT_GENOME
    assert abs(new["novelty_alpha"] - 0.45) < 1e-9
    assert DEFAULT_GENOME["novelty_alpha"] == 0.3  # original unchanged


def test_apply_genome_mutation_rejects_out_of_bounds():
    bad = GenomeMutation(
        parent_hash="h", child_hash="h2",
        description="bad island_count",
        parameter_changes={"island_count": 100},
        operator_changes={},
    )
    with pytest.raises(GenomeConstraintViolation):
        apply_genome_mutation(bad, DEFAULT_GENOME)


def test_validate_genome_mutation_returns_none_when_valid():
    assert validate_genome_mutation(
        proposed={"island_count": 10, "novelty_alpha": 0.4},
        current=DEFAULT_GENOME,
    ) is None


def test_validate_genome_mutation_flags_out_of_bounds():
    err = validate_genome_mutation(
        proposed={"island_count": 100}, current=DEFAULT_GENOME,
    )
    assert err is not None
    assert "island_count" in err


# --- propose_mutation against synthetic ledger fixtures ---


_DIVERSE_FAMILIES = ["bigru", "1d_cnn", "transformer", "multi_stream_bigru"]


def _seed_synthetic_ledger(led, n_iters, fitness_fn, fingerprint_fn,
                            constraint_fn, families=None):
    """Populate a Ledger with synthetic history for propose_mutation tests.

    `families` cycles model families across experiments. Defaults to a diverse
    4-family rotation so the family_monoculture detector (Priority 0) does NOT
    fire -- isolating the Priority 1/2/3 branches under test. Monoculture tests
    pass families=["one_family"].
    """
    fams = families or _DIVERSE_FAMILIES
    for i in range(n_iters):
        spec = {"model": {"family": fams[i % len(fams)]}}
        rid = led.allocate_run_id()
        led.write_experiment(rid, spec, parent_id=None, island_id=i % 4)
        led.write_result(rid, {"balanced_acc": fitness_fn(i)})
        led.write_mutation_trace(
            iteration=i, run_id=rid, parent_run_ids=[],
            prompt_context="", child_spec=spec,
            fingerprint=fingerprint_fn(i),
            reasoning_summary="", accepted=True,
        )
        rule, ok = constraint_fn(i)
        led.write_constraint_event(
            iteration=i, child_fingerprint=fingerprint_fn(i),
            rule_name=rule, accepted=ok,
            reason_code=None if ok else "dup",
        )


def test_propose_mutation_stagnation_scales_exploration(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _seed_synthetic_ledger(
        led, n_iters=25,
        fitness_fn=lambda i: 0.420 + 0.0001 * i,        # flat
        fingerprint_fn=lambda i: f"fp_{i}",             # all distinct
        constraint_fn=lambda i: ("rule_guards", True),  # no rejection
    )
    try:
        mut = propose_mutation(led, DEFAULT_GENOME, current_iter=25,
                               config=DetectorConfig())
        assert mut is not None
        # Stagnation branch -> SCALE_PARAM on an exploration knob
        targeted = set(mut.parameter_changes.keys())
        assert targeted & {"novelty_alpha", "p_lit_drift_sigma"}
    finally:
        led.close()


def test_propose_mutation_over_rejection_targets_offending_rule(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()

    def constraint_fn(i):
        # 80% rejection on ast_tabu
        return ("ast_tabu", i % 5 == 0)

    _seed_synthetic_ledger(
        led, n_iters=25,
        fitness_fn=lambda i: 0.42 + 0.01 * i,        # rising, no stagnation
        fingerprint_fn=lambda i: f"fp_{i}",          # diverse
        constraint_fn=constraint_fn,
    )
    try:
        mut = propose_mutation(led, DEFAULT_GENOME, current_iter=25,
                               config=DetectorConfig())
        assert mut is not None
        # Should relax ast_tabu_k (scale DOWN)
        assert "ast_tabu_k" in mut.parameter_changes
        assert mut.parameter_changes["ast_tabu_k"] < DEFAULT_GENOME["ast_tabu_k"]
    finally:
        led.close()


def test_propose_mutation_entropy_collapse_adds_island_or_axis_weight(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _seed_synthetic_ledger(
        led, n_iters=25,
        fitness_fn=lambda i: 0.42 + 0.01 * i,         # rising
        fingerprint_fn=lambda i: f"fp_{i % 2}",       # only 2 unique -> low entropy
        constraint_fn=lambda i: ("rule_guards", True),
    )
    try:
        mut = propose_mutation(led, DEFAULT_GENOME, current_iter=25,
                               config=DetectorConfig())
        assert mut is not None
        # Either ADD_ISLAND or SET_AXIS_WEIGHT(novelty, +)
        keys = set(mut.parameter_changes.keys())
        assert keys & {"island_count", "axis_weights"}
    finally:
        led.close()


def test_propose_mutation_returns_none_when_healthy(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _seed_synthetic_ledger(
        led, n_iters=25,
        fitness_fn=lambda i: 0.42 + 0.02 * i,         # strong gain
        fingerprint_fn=lambda i: f"fp_{i}",           # diverse
        constraint_fn=lambda i: ("rule_guards", True),
    )
    try:
        mut = propose_mutation(led, DEFAULT_GENOME, current_iter=25,
                               config=DetectorConfig())
        assert mut is None
    finally:
        led.close()


# --- Prompt assembly + hash ---


def test_assemble_introspection_prompt_returns_structured_blob():
    out = assemble_introspection_prompt(
        ledger_recent=[],
        current_genome={"island_count": 8},
        m_iter_window=50,
    )
    assert isinstance(out, str)
    assert "## Recent fitness trajectory" in out
    assert "## Current genome" in out
    assert "## Mutation directive" in out


def test_compute_genome_hash_is_deterministic():
    h1 = compute_genome_hash(DEFAULT_GENOME)
    h2 = compute_genome_hash(dict(DEFAULT_GENOME))
    assert h1 == h2
    assert len(h1) >= 16


def test_compute_genome_hash_changes_with_content():
    h1 = compute_genome_hash(DEFAULT_GENOME)
    h2 = compute_genome_hash(dict(DEFAULT_GENOME, island_count=10))
    assert h1 != h2


# --- family_monoculture detector (iter_0015 framework rebuild) ---

from framework.introspect import detect_family_monoculture


def test_detect_family_monoculture_fires_on_dominant_family(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    # 22 of 24 multi_stream_bigru = 91% -> monoculture
    fams = (["multi_stream_bigru"] * 22) + ["bigru", "transformer"]
    _seed_synthetic_ledger(
        led, n_iters=24,
        fitness_fn=lambda i: 0.50,
        fingerprint_fn=lambda i: f"fp_{i}",
        constraint_fn=lambda i: ("rule_guards", True),
        families=fams)
    try:
        assert detect_family_monoculture(led, window=24, threshold=0.7) is True
    finally:
        led.close()


def test_detect_family_monoculture_quiet_on_diverse_history(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _seed_synthetic_ledger(
        led, n_iters=24,
        fitness_fn=lambda i: 0.50,
        fingerprint_fn=lambda i: f"fp_{i}",
        constraint_fn=lambda i: ("rule_guards", True))  # 4-family rotation
    try:
        assert detect_family_monoculture(led, window=24, threshold=0.7) is False
    finally:
        led.close()


def test_propose_mutation_monoculture_tightens_max_per_family(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _seed_synthetic_ledger(
        led, n_iters=25,
        fitness_fn=lambda i: 0.42 + 0.01 * i,            # rising (no stagnation)
        fingerprint_fn=lambda i: f"fp_{i}",              # diverse fingerprints
        constraint_fn=lambda i: ("rule_guards", True),   # no rejection
        families=["multi_stream_bigru"])                 # 100% monoculture
    try:
        mut = propose_mutation(led, DEFAULT_GENOME, current_iter=25,
                               config=DetectorConfig())
        assert mut is not None
        assert "max_per_family" in mut.parameter_changes
        assert mut.parameter_changes["max_per_family"] == 2
    finally:
        led.close()


def test_ledger_recent_family_distribution_counts(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _seed_synthetic_ledger(
        led, n_iters=8,
        fitness_fn=lambda i: 0.5,
        fingerprint_fn=lambda i: f"fp_{i}",
        constraint_fn=lambda i: ("rule_guards", True),
        families=["bigru", "bigru", "transformer"])
    try:
        dist = led.recent_family_distribution(window=8)
        # 8 experiments, pattern bigru,bigru,transformer repeating
        assert dist["bigru"] + dist["transformer"] == 8
        assert dist["bigru"] > dist["transformer"]
    finally:
        led.close()
