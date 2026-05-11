"""Tests for framework.loop. Spec: FRAMEWORK.md Section 8."""
import pytest
from pathlib import Path
from framework import loop


def test_module_imports():
    assert loop.IterationOutcome is not None
    assert loop.IterationPaused is not None
    assert loop.IterationCompleted is not None


def test_iteration_paused_carries_hip_label():
    p = loop.IterationPaused(hip="HIP-D", run_id="r1", action_required="sbatch on cluster")
    assert p.hip == "HIP-D"
    assert p.run_id == "r1"


def test_iteration_completed_carries_fitness(sample_fitness_vector):
    c = loop.IterationCompleted(run_id="r1", fitness_vector=sample_fitness_vector)
    assert c.fitness_vector["balanced_acc"] == 0.55


def test_advance_one_iteration_runs_on_empty_ledger(tmp_path: Path):
    out = loop.advance_one_iteration(
        experiments_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.db",
    )
    assert isinstance(out, loop.IterationOutcome)
    assert isinstance(out, loop.IterationPaused)
    assert out.hip == "HIP-D"


def test_advance_one_iteration_seeds_first_runnable_program(tmp_path: Path):
    """First call on empty ledger should write spec.json + run.py for the bigru seed."""
    out = loop.advance_one_iteration(
        experiments_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.db",
    )
    assert isinstance(out, loop.IterationPaused)
    run_dir = tmp_path / "experiments" / out.run_id
    assert (run_dir / "spec.json").exists()
    assert (run_dir / "run.py").exists()


def test_advance_one_iteration_persists_to_ledger(tmp_path: Path):
    """First call should write a row to experiments table."""
    out = loop.advance_one_iteration(
        experiments_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.db",
    )
    from framework.ledger import Ledger
    led = Ledger(tmp_path / "ledger.db")
    led.init_schema()
    members = led.get_island_members(0)
    led.close()
    assert any(m["run_id"] == out.run_id for m in members)


def test_advance_one_iteration_subsequent_call_returns_paused(tmp_path: Path):
    out1 = loop.advance_one_iteration(
        experiments_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.db",
    )
    out2 = loop.advance_one_iteration(
        experiments_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.db",
    )
    assert isinstance(out1, loop.IterationPaused)
    assert isinstance(out2, loop.IterationPaused)


# ---------- report_result ----------

def test_report_result_returns_completed(tmp_path: Path, sample_fitness_vector):
    out = loop.advance_one_iteration(
        experiments_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.db",
    )
    completed = loop.report_result(
        run_id=out.run_id, fitness_vector=sample_fitness_vector,
        ledger_path=tmp_path / "ledger.db",
    )
    assert isinstance(completed, loop.IterationCompleted)
    assert completed.run_id == out.run_id
    assert completed.fitness_vector == sample_fitness_vector


def test_report_result_persists_fitness_to_ledger(tmp_path: Path, sample_fitness_vector):
    from framework.ledger import Ledger
    out = loop.advance_one_iteration(
        experiments_root=tmp_path / "experiments",
        ledger_path=tmp_path / "ledger.db",
    )
    loop.report_result(out.run_id, sample_fitness_vector, tmp_path / "ledger.db")
    led = Ledger(tmp_path / "ledger.db")
    led.init_schema()
    members = led.get_island_members(0)
    led.close()
    persisted = next(m for m in members if m["run_id"] == out.run_id)
    assert persisted["fitness"]["balanced_acc"] == sample_fitness_vector["balanced_acc"]
