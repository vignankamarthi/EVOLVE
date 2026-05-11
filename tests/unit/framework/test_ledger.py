"""Tests for framework.ledger. Spec: FRAMEWORK.md Section 10."""
import time as _time
import pytest
from framework import ledger


def test_module_imports():
    assert ledger.DEFAULT_DB_PATH.name == "experiments.db"


def test_init_schema_is_idempotent(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    led.init_schema()
    led.close()


def test_allocate_run_id_returns_unique_strings(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    ids = [led.allocate_run_id() for _ in range(5)]
    assert len(set(ids)) == 5
    assert all(rid.startswith("r_") and len(rid) == 10 for rid in ids)
    led.close()


def test_write_experiment_round_trip(tmp_db_path, sample_program_spec, sample_fitness_vector):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    rid = led.allocate_run_id()
    led.write_experiment(rid, sample_program_spec, parent_id=None, island_id=0)
    led.write_result(rid, sample_fitness_vector)
    members = led.get_island_members(0)
    assert len(members) == 1
    m = members[0]
    assert m["run_id"] == rid
    assert m["spec"] == sample_program_spec
    assert m["fitness"]["balanced_acc"] == sample_fitness_vector["balanced_acc"]
    assert m["completed_at"] is not None
    assert m["completed_at"] >= m["created_at"]
    led.close()


def test_lineage_recorded_when_parent_id_present(tmp_db_path, sample_program_spec):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    parent = led.allocate_run_id()
    child = led.allocate_run_id()
    led.write_experiment(parent, sample_program_spec, parent_id=None, island_id=0)
    led.write_experiment(child, sample_program_spec, parent_id=parent, island_id=0)
    members = led.get_island_members(0)
    parent_row = next(m for m in members if m["run_id"] == parent)
    child_row = next(m for m in members if m["run_id"] == child)
    assert parent_row["parent_id"] is None
    assert child_row["parent_id"] == parent
    led.close()


def test_get_recent_iterations_orders_newest_first(tmp_db_path, sample_program_spec, sample_fitness_vector):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    rids = []
    for _ in range(3):
        rid = led.allocate_run_id()
        led.write_experiment(rid, sample_program_spec, parent_id=None, island_id=0)
        led.write_result(rid, sample_fitness_vector)
        rids.append(rid)
        _time.sleep(0.005)
    recent = led.get_recent_iterations(n=2)
    assert len(recent) == 2
    assert recent[0]["run_id"] == rids[-1]
    assert recent[1]["run_id"] == rids[-2]
    led.close()


def test_get_recent_iterations_excludes_unfinished(tmp_db_path, sample_program_spec, sample_fitness_vector):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    finished = led.allocate_run_id()
    unfinished = led.allocate_run_id()
    led.write_experiment(finished, sample_program_spec, parent_id=None, island_id=0)
    led.write_result(finished, sample_fitness_vector)
    led.write_experiment(unfinished, sample_program_spec, parent_id=None, island_id=0)
    recent = led.get_recent_iterations(n=10)
    rids = [r["run_id"] for r in recent]
    assert finished in rids
    assert unfinished not in rids
    led.close()


def test_framework_mutation_persists(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    led.write_framework_mutation("hash_a", "hash_b",
                                 "raise curriculum_threshold 0.55 -> 0.60")
    cur = led._conn.execute(
        "SELECT parent_hash, child_hash, description FROM framework_mutations")
    row = cur.fetchone()
    assert row["parent_hash"] == "hash_a"
    assert row["child_hash"] == "hash_b"
    assert "curriculum_threshold" in row["description"]
    led.close()


def test_context_manager_closes(tmp_db_path):
    with ledger.Ledger(tmp_db_path) as led:
        led.init_schema()
        rid = led.allocate_run_id()
    # Re-open to confirm persistence
    led2 = ledger.Ledger(tmp_db_path)
    led2.init_schema()
    next_rid = led2.allocate_run_id()
    assert next_rid != rid
    led2.close()


# --- Phase 1: mutation_traces table + JSONL mirror ---


def test_write_mutation_trace_round_trip(tmp_db_path, sample_program_spec):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    led.write_mutation_trace(
        iteration=3,
        run_id="r_00000001",
        parent_run_ids=["r_00000000"],
        prompt_context="(stub prompt)",
        child_spec=sample_program_spec,
        fingerprint="fp_abc",
        reasoning_summary="raised hidden_size 64 -> 96 to break stagnation",
        accepted=True,
    )
    traces = led.recent_mutation_traces(window=10)
    assert len(traces) == 1
    t = traces[0]
    assert t["iteration"] == 3
    assert t["run_id"] == "r_00000001"
    assert t["parent_run_ids"] == ["r_00000000"]
    assert t["child_spec"] == sample_program_spec
    assert t["fingerprint"] == "fp_abc"
    assert t["reasoning_summary"].startswith("raised hidden_size")
    assert t["accepted"] is True
    led.close()


def test_write_mutation_trace_jsonl_mirror(tmp_db_path, tmp_path, sample_program_spec):
    import json as _json
    experiments_root = tmp_path / "experiments"
    led = ledger.Ledger(tmp_db_path, experiments_root=experiments_root)
    led.init_schema()
    rid = "r_00000005"
    led.write_mutation_trace(
        iteration=7,
        run_id=rid,
        parent_run_ids=[],
        prompt_context="(p)",
        child_spec=sample_program_spec,
        fingerprint="fp1",
        reasoning_summary="seed",
        accepted=True,
    )
    jsonl = experiments_root / rid / "trace.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = _json.loads(lines[0])
    assert payload["iteration"] == 7
    assert payload["fingerprint"] == "fp1"
    assert payload["accepted"] is True
    led.close()


def test_recent_mutation_traces_orders_newest_first(tmp_db_path, sample_program_spec):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    for i in range(5):
        led.write_mutation_trace(
            iteration=i, run_id=f"r_{i:08d}", parent_run_ids=[],
            prompt_context="", child_spec=sample_program_spec,
            fingerprint=f"fp_{i}", reasoning_summary="", accepted=True,
        )
        _time.sleep(0.002)
    recent = led.recent_mutation_traces(window=3)
    assert len(recent) == 3
    assert [t["iteration"] for t in recent] == [4, 3, 2]
    led.close()


def test_current_iteration_returns_max(tmp_db_path, sample_program_spec):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    assert led.current_iteration() == 0
    for i in [2, 4, 1, 7, 3]:
        led.write_mutation_trace(
            iteration=i, run_id=f"r_{i:08d}", parent_run_ids=[],
            prompt_context="", child_spec=sample_program_spec,
            fingerprint=f"fp_{i}", reasoning_summary="", accepted=True,
        )
    assert led.current_iteration() == 7
    led.close()


def test_fingerprint_entropy_uniform_distribution(tmp_db_path, sample_program_spec):
    import math
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    for i in range(8):
        led.write_mutation_trace(
            iteration=i, run_id=f"r_{i:08d}", parent_run_ids=[],
            prompt_context="", child_spec=sample_program_spec,
            fingerprint=f"fp_unique_{i}", reasoning_summary="", accepted=True,
        )
    h = led.fingerprint_entropy(window=8)
    # 8 distinct fingerprints, p = 1/8 each, H = log2(8) = 3.0
    assert abs(h - math.log2(8)) < 1e-9
    led.close()


def test_fingerprint_entropy_full_collapse(tmp_db_path, sample_program_spec):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    for i in range(8):
        led.write_mutation_trace(
            iteration=i, run_id=f"r_{i:08d}", parent_run_ids=[],
            prompt_context="", child_spec=sample_program_spec,
            fingerprint="fp_only", reasoning_summary="", accepted=True,
        )
    h = led.fingerprint_entropy(window=8)
    assert h == 0.0
    led.close()


def test_fingerprint_entropy_empty_window(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    assert led.fingerprint_entropy(window=10) == 0.0
    led.close()


# --- Phase 1: constraint_events table ---


def test_write_constraint_event_round_trip(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    led.write_constraint_event(
        iteration=2, child_fingerprint="fp_x", rule_name="ast_tabu",
        accepted=False, reason_code="duplicate",
        reason_detail="fingerprint already in tabu list",
    )
    rate = led.constraint_rejection_rate(window=10)
    assert rate == 1.0
    led.close()


def test_constraint_rejection_rate_mixed(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    for i in range(3):
        led.write_constraint_event(
            iteration=i, child_fingerprint=f"fp_{i}",
            rule_name="rule_guards", accepted=True,
        )
    for i in range(3, 5):
        led.write_constraint_event(
            iteration=i, child_fingerprint=f"fp_{i}",
            rule_name="ast_tabu", accepted=False, reason_code="dup",
        )
    rate = led.constraint_rejection_rate(window=10)
    assert abs(rate - 2 / 5) < 1e-9
    led.close()


def test_constraint_rejection_rate_by_rule(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    # 2 ast_tabu accepted, 3 ast_tabu rejected
    for i in range(2):
        led.write_constraint_event(
            iteration=i, child_fingerprint=f"fp_a{i}",
            rule_name="ast_tabu", accepted=True,
        )
    for i in range(3):
        led.write_constraint_event(
            iteration=10 + i, child_fingerprint=f"fp_b{i}",
            rule_name="ast_tabu", accepted=False, reason_code="dup",
        )
    # 5 lineage_cap accepted
    for i in range(5):
        led.write_constraint_event(
            iteration=20 + i, child_fingerprint=f"fp_c{i}",
            rule_name="lineage_cap", accepted=True,
        )
    rate_all = led.constraint_rejection_rate(window=50)
    rate_tabu = led.constraint_rejection_rate(window=50, rule="ast_tabu")
    rate_lineage = led.constraint_rejection_rate(window=50, rule="lineage_cap")
    assert abs(rate_all - 3 / 10) < 1e-9
    assert abs(rate_tabu - 3 / 5) < 1e-9
    assert rate_lineage == 0.0
    led.close()


def test_constraint_rejection_rate_empty(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    assert led.constraint_rejection_rate(window=10) == 0.0
    assert led.constraint_rejection_rate(window=10, rule="ast_tabu") == 0.0
    led.close()


# --- Phase 1: median_fitness_delta_per_island ---


def test_median_fitness_delta_per_island_empty(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    assert led.median_fitness_delta_per_island(window=10) == 0.0
    led.close()


def test_median_fitness_delta_per_island_basic(tmp_db_path, sample_program_spec):
    """Two islands with different deltas; median across islands is computed."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    # Island 0: 0.40 -> 0.45 -> 0.50 (deltas 0.05, 0.05; per-island median 0.05)
    for acc in (0.40, 0.45, 0.50):
        rid = led.allocate_run_id()
        led.write_experiment(rid, sample_program_spec, parent_id=None, island_id=0)
        led.write_result(rid, {"balanced_acc": acc})
        _time.sleep(0.002)
    # Island 1: 0.42 -> 0.42 -> 0.42 (deltas 0, 0; per-island median 0)
    for acc in (0.42, 0.42, 0.42):
        rid = led.allocate_run_id()
        led.write_experiment(rid, sample_program_spec, parent_id=None, island_id=1)
        led.write_result(rid, {"balanced_acc": acc})
        _time.sleep(0.002)
    delta = led.median_fitness_delta_per_island(window=20)
    # median of per-island medians [0.05, 0.0] = 0.025
    assert abs(delta - 0.025) < 1e-6
    led.close()


def test_framework_mutation_records_iteration_optional(tmp_db_path):
    """write_framework_mutation accepts optional iteration metadata in description."""
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    led.write_framework_mutation("hash_a", "hash_b",
                                 "[iter=42] SCALE_PARAM(novelty_alpha, 1.5)")
    cur = led._conn.execute(
        "SELECT description FROM framework_mutations ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    assert "iter=42" in row["description"]
    led.close()
