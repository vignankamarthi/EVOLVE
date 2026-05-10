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
