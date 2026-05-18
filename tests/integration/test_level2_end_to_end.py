"""Integration: end-to-end Level 2 cycle on a synthetic ledger.

Exercises FRAMEWORK.md Section 7 + Section 11:
  populate ledger with stagnation pattern ->
  loop.check_level2 fires ->
  propose_mutation returns typed mutation ->
  apply_genome_mutation builds new genome ->
  write_framework_mutation persists the cycle

Plus a healthy-state regression (no fire) and a smoke test that
loop.record_mutation_attempt + record_constraint_check write through the
ledger AND optionally mirror to <run_dir>/trace.jsonl when run_dir is passed.
"""
from framework import introspect, ledger, loop


DEFAULT_GENOME = {
    "island_count": 8, "island_size": 12, "reset_cadence": 50,
    "novelty_alpha": 0.3, "ece_lambda": 0.5,
    "max_params": 10_000_000, "max_train_seconds": 1800,
    "ast_tabu_k": 50, "curriculum_threshold": 0.55, "lineage_cap": 5,
    "migration_patience": 10, "critic_pop_size": 30, "stagnation_patience": 10,
    "p_lit_drift_sigma": 0.05, "failure_boost_gain": 1.0,
    "introspection_cadence_M": 50,
    "axis_weights": {"balanced_acc": 1.0, "novelty": 0.3, "ece": 0.5, "gap": 0.2},
    "operators_enabled": {"failure_boost": True, "critic": True, "migration": True},
    "operator_slots": {"replacement_rule": "GENITOR"},
}


# Diverse family rotation so the Level 2 family_monoculture detector
# (Priority 0) does not fire -- isolates the stagnation / rejection branches.
_DIVERSE_FAMILIES = ["bigru", "1d_cnn", "transformer", "multi_stream_bigru"]


def _populate(led, n_iters, fitness_fn, fingerprint_fn, constraint_fn,
              families=None):
    fams = families or _DIVERSE_FAMILIES
    for i in range(n_iters):
        spec = {"model": {"family": fams[i % len(fams)]}}
        rid = led.allocate_run_id()
        led.write_experiment(rid, spec, parent_id=None, island_id=i % 4)
        led.write_result(rid, {"balanced_acc": fitness_fn(i)})
        led.write_mutation_trace(
            iteration=i, run_id=rid, parent_run_ids=[],
            prompt_context="", child_spec=spec,
            fingerprint=fingerprint_fn(i), reasoning_summary="",
            accepted=True,
        )
        rule, ok = constraint_fn(i)
        led.write_constraint_event(
            iteration=i, child_fingerprint=fingerprint_fn(i),
            rule_name=rule, accepted=ok,
            reason_code=None if ok else "dup",
        )


def test_level2_full_cycle_stagnation(tmp_db_path, tmp_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _populate(
        led, n_iters=25,
        fitness_fn=lambda i: 0.420 + 0.0001 * i,
        fingerprint_fn=lambda i: f"fp_{i}",
        constraint_fn=lambda i: ("rule_guards", True),
    )

    fired, proposal, signals = loop.check_level2(
        led, DEFAULT_GENOME, current_iter=25, last_fire_iter=0,
    )
    assert fired is True
    assert proposal is not None
    assert "novelty_alpha" in proposal.parameter_changes
    assert signals["median_fitness_delta"] < 0.005

    new_genome = introspect.apply_genome_mutation(proposal, DEFAULT_GENOME)
    assert new_genome["novelty_alpha"] > DEFAULT_GENOME["novelty_alpha"]

    led.write_framework_mutation(
        proposal.parent_hash, proposal.child_hash, proposal.description,
    )
    rows = led._conn.execute(
        "SELECT description FROM framework_mutations").fetchall()
    assert len(rows) == 1
    assert "SCALE_PARAM" in rows[0]["description"]
    led.close()


def test_level2_no_fire_when_healthy(tmp_db_path, tmp_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _populate(
        led, n_iters=25,
        fitness_fn=lambda i: 0.42 + 0.02 * i,
        fingerprint_fn=lambda i: f"fp_{i}",
        constraint_fn=lambda i: ("rule_guards", True),
    )
    fired, proposal, _ = loop.check_level2(
        led, DEFAULT_GENOME, current_iter=25, last_fire_iter=0,
    )
    assert fired is False
    assert proposal is None
    led.close()


def test_level2_over_rejection_routes_to_offending_rule(tmp_db_path, tmp_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    _populate(
        led, n_iters=25,
        fitness_fn=lambda i: 0.42 + 0.01 * i,
        fingerprint_fn=lambda i: f"fp_{i}",
        constraint_fn=lambda i: ("ast_tabu", i % 5 == 0),  # 80% rejection
    )
    fired, proposal, _ = loop.check_level2(
        led, DEFAULT_GENOME, current_iter=25, last_fire_iter=0,
    )
    assert fired is True
    assert proposal is not None
    assert "ast_tabu_k" in proposal.parameter_changes
    assert proposal.parameter_changes["ast_tabu_k"] < DEFAULT_GENOME["ast_tabu_k"]
    led.close()


def test_loop_record_helpers_write_to_ledger_and_jsonl(tmp_db_path, tmp_path):
    """record_mutation_attempt with run_dir lands JSONL next to spec.json."""
    spec = {"model": {"family": "bigru"}}
    run_dir = tmp_path / "iter_0001" / "child_00"
    run_dir.mkdir(parents=True)
    loop.record_mutation_attempt(
        iteration=0, run_id="r_00000001", parent_run_ids=[],
        prompt_context="(seed)", child_spec=spec,
        fingerprint="fp_seed", reasoning_summary="initial seed",
        accepted=True,
        run_dir=run_dir, ledger_path=tmp_db_path,
    )
    loop.record_constraint_check(
        iteration=0, fingerprint="fp_seed",
        rule_name="rule_guards", accepted=True,
        ledger_path=tmp_db_path,
    )
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    traces = led.recent_mutation_traces(window=10)
    assert len(traces) == 1
    assert traces[0]["fingerprint"] == "fp_seed"
    assert led.constraint_rejection_rate(window=10) == 0.0
    assert (run_dir / "trace.jsonl").exists()
    led.close()
