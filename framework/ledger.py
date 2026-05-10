"""SQLite ledger for the EVOLVE framework.

Tables:
  experiments         run_id, parent_id, island_id, spec_json, fitness_json, timestamps
  lineage             child_id, parent_id, mutation_type
  islands             island_id, best_fitness, last_improvement_iter
  critic_population   critic_id, parent_id, critic_genome_json, fitness
  meta_state          iter, p_lit, novelty_alpha, temperature, failure_boost_json
  framework_mutations id, parent_hash, child_hash, description, fitness_after_M_json, created_at
  submissions         submission_id, run_id, slot_used, balanced_acc, created_at
  run_id_seq          single-column rowid sequencer for unique run_id allocation

All writes are atomic (autocommit, WAL journal). Loop is resumable from any iteration.

Spec: FRAMEWORK.md Section 6 (meta_state), Section 7 (framework_mutations).
"""
from pathlib import Path
from typing import Any
import json
import sqlite3
import time


DEFAULT_DB_PATH = Path("ledger/experiments.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id TEXT PRIMARY KEY,
    parent_id TEXT,
    island_id INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    fitness_json TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_experiments_island ON experiments(island_id);
CREATE INDEX IF NOT EXISTS idx_experiments_parent ON experiments(parent_id);
CREATE INDEX IF NOT EXISTS idx_experiments_completed ON experiments(completed_at);

CREATE TABLE IF NOT EXISTS lineage (
    child_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    mutation_type TEXT,
    PRIMARY KEY (child_id, parent_id)
);

CREATE TABLE IF NOT EXISTS islands (
    island_id INTEGER PRIMARY KEY,
    best_fitness REAL,
    last_improvement_iter INTEGER
);

CREATE TABLE IF NOT EXISTS critic_population (
    critic_id TEXT PRIMARY KEY,
    parent_id TEXT,
    critic_genome_json TEXT NOT NULL,
    fitness REAL
);

CREATE TABLE IF NOT EXISTS meta_state (
    iter INTEGER PRIMARY KEY,
    p_lit REAL NOT NULL,
    novelty_alpha REAL NOT NULL,
    temperature REAL NOT NULL,
    failure_boost_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS framework_mutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_hash TEXT NOT NULL,
    child_hash TEXT NOT NULL,
    description TEXT NOT NULL,
    fitness_after_M_json TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    slot_used INTEGER NOT NULL UNIQUE,
    balanced_acc REAL,
    created_at REAL NOT NULL,
    FOREIGN KEY(run_id) REFERENCES experiments(run_id)
);

CREATE TABLE IF NOT EXISTS run_id_seq (
    n INTEGER PRIMARY KEY AUTOINCREMENT
);
"""


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _hydrate_experiment(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "parent_id": row["parent_id"],
        "island_id": row["island_id"],
        "spec": json.loads(row["spec_json"]),
        "fitness": json.loads(row["fitness_json"]) if row["fitness_json"] else None,
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
    }


class Ledger:
    """SQLite-backed experiment ledger. See module docstring for schema."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def init_schema(self) -> None:
        """Create tables and indices. Idempotent."""
        self._conn.executescript(SCHEMA)

    def allocate_run_id(self) -> str:
        """Allocate a unique run_id. Format: 'r_' + 8-digit zero-padded sequence."""
        cur = self._conn.execute("INSERT INTO run_id_seq DEFAULT VALUES")
        n = cur.lastrowid
        return f"r_{n:08d}"

    def write_experiment(self, run_id: str, spec_json: dict,
                         parent_id: str | None, island_id: int) -> None:
        self._conn.execute(
            "INSERT INTO experiments (run_id, parent_id, island_id, spec_json, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, parent_id, island_id, json.dumps(spec_json), time.time()),
        )
        if parent_id is not None:
            self._conn.execute(
                "INSERT OR IGNORE INTO lineage (child_id, parent_id, mutation_type) "
                "VALUES (?, ?, ?)",
                (run_id, parent_id, "mutation"),
            )

    def write_result(self, run_id: str, fitness_vector: dict) -> None:
        self._conn.execute(
            "UPDATE experiments SET fitness_json = ?, completed_at = ? "
            "WHERE run_id = ?",
            (json.dumps(fitness_vector), time.time(), run_id),
        )

    def get_island_members(self, island_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT run_id, parent_id, island_id, spec_json, fitness_json, "
            "created_at, completed_at FROM experiments WHERE island_id = ? "
            "ORDER BY created_at ASC",
            (island_id,),
        ).fetchall()
        return [_hydrate_experiment(r) for r in rows]

    def get_recent_iterations(self, n: int) -> list[dict]:
        """Return up to n most-recently completed experiments, newest first."""
        rows = self._conn.execute(
            "SELECT run_id, parent_id, island_id, spec_json, fitness_json, "
            "created_at, completed_at FROM experiments "
            "WHERE fitness_json IS NOT NULL "
            "ORDER BY completed_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [_hydrate_experiment(r) for r in rows]

    def write_framework_mutation(self, parent_hash: str, child_hash: str,
                                 desc: str) -> None:
        self._conn.execute(
            "INSERT INTO framework_mutations "
            "(parent_hash, child_hash, description, created_at) "
            "VALUES (?, ?, ?, ?)",
            (parent_hash, child_hash, desc, time.time()),
        )
