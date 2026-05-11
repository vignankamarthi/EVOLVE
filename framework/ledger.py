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
  mutation_traces     iteration, run_id, parent_run_ids, prompt_context,
                      child_spec, fingerprint, reasoning_summary, accepted (Section 11)
  constraint_events   iteration, child_fingerprint, rule_name, accepted,
                      reason_code, reason_detail (Section 11)

All writes are atomic (autocommit, WAL journal). Loop is resumable from any iteration.

`mutation_traces` writes optionally mirror to `<run_dir>/trace.jsonl` when
`run_dir` is passed to `write_mutation_trace`. JSONL is the durable write-ahead
log; SQLite is the queryable materialized view for Level 2.

Spec: FRAMEWORK.md Section 6 (meta_state), Section 7 (framework_mutations),
Section 11 (mutation_traces, constraint_events, query helpers).
"""
from pathlib import Path
from typing import Any
import json
import math
import sqlite3
import statistics
import time


DEFAULT_DB_PATH = Path("ledger/experiments.db")
DEFAULT_EXPERIMENTS_ROOT = Path("experiments")


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

CREATE TABLE IF NOT EXISTS mutation_traces (
    iteration INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    parent_run_ids TEXT,
    prompt_context TEXT,
    child_spec TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    reasoning_summary TEXT,
    accepted INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (iteration, run_id)
);
CREATE INDEX IF NOT EXISTS idx_traces_iter ON mutation_traces(iteration);
CREATE INDEX IF NOT EXISTS idx_traces_fingerprint ON mutation_traces(fingerprint);
CREATE INDEX IF NOT EXISTS idx_traces_created ON mutation_traces(created_at);

CREATE TABLE IF NOT EXISTS constraint_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER NOT NULL,
    child_fingerprint TEXT,
    rule_name TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    reason_code TEXT,
    reason_detail TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_iter ON constraint_events(iteration);
CREATE INDEX IF NOT EXISTS idx_events_rule ON constraint_events(rule_name);
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

    # --- Section 11: mutation_traces + JSONL mirror ---

    def write_mutation_trace(self, iteration: int, run_id: str,
                             parent_run_ids: list[str],
                             prompt_context: str,
                             child_spec: dict,
                             fingerprint: str,
                             reasoning_summary: str,
                             accepted: bool,
                             run_dir: Path | None = None) -> None:
        """Persist one mutation trace.

        SQLite write is atomic via the autocommit connection. If `run_dir`
        is provided, the same payload is appended to `<run_dir>/trace.jsonl`
        (alongside spec.json + run.py + result.json). This is the canonical
        Section 11 observability layout: every artifact for one run lives in
        one directory.

        Note: the previous experiments_root-based sentinel naming
        (`experiments_root / run_id / trace.jsonl`) was removed 2026-05-11
        because run_id is not the same as run_dir under the iter_NNNN/child_MM/
        layout. Pass `run_dir` explicitly.
        """
        created_at = time.time()
        payload = {
            "iteration": iteration,
            "run_id": run_id,
            "parent_run_ids": parent_run_ids,
            "prompt_context": prompt_context,
            "child_spec": child_spec,
            "fingerprint": fingerprint,
            "reasoning_summary": reasoning_summary,
            "accepted": bool(accepted),
            "created_at": created_at,
        }
        self._conn.execute(
            "INSERT OR REPLACE INTO mutation_traces "
            "(iteration, run_id, parent_run_ids, prompt_context, child_spec, "
            "fingerprint, reasoning_summary, accepted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                iteration,
                run_id,
                json.dumps(parent_run_ids),
                prompt_context,
                json.dumps(child_spec),
                fingerprint,
                reasoning_summary,
                1 if accepted else 0,
                created_at,
            ),
        )
        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "trace.jsonl").open("a") as f:
                f.write(json.dumps(payload) + "\n")

    def recent_mutation_traces(self, window: int) -> list[dict]:
        """Last `window` traces, newest first."""
        rows = self._conn.execute(
            "SELECT iteration, run_id, parent_run_ids, prompt_context, "
            "child_spec, fingerprint, reasoning_summary, accepted, created_at "
            "FROM mutation_traces ORDER BY created_at DESC LIMIT ?",
            (int(window),),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "iteration": r["iteration"],
                "run_id": r["run_id"],
                "parent_run_ids": json.loads(r["parent_run_ids"]) if r["parent_run_ids"] else [],
                "prompt_context": r["prompt_context"],
                "child_spec": json.loads(r["child_spec"]),
                "fingerprint": r["fingerprint"],
                "reasoning_summary": r["reasoning_summary"],
                "accepted": bool(r["accepted"]),
                "created_at": r["created_at"],
            })
        return out

    def current_iteration(self) -> int:
        """Highest iteration in mutation_traces; 0 if empty."""
        row = self._conn.execute(
            "SELECT MAX(iteration) AS m FROM mutation_traces").fetchone()
        if row is None or row["m"] is None:
            return 0
        return int(row["m"])

    def fingerprint_entropy(self, window: int) -> float:
        """Shannon entropy (bits) of the fingerprint distribution in the
        most recent `window` traces. Returns 0.0 on empty window.
        """
        rows = self._conn.execute(
            "SELECT fingerprint FROM mutation_traces "
            "ORDER BY created_at DESC LIMIT ?",
            (int(window),),
        ).fetchall()
        if not rows:
            return 0.0
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["fingerprint"]] = counts.get(r["fingerprint"], 0) + 1
        n = sum(counts.values())
        h = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                h -= p * math.log2(p)
        return h

    # --- Section 11: constraint_events ---

    def write_constraint_event(self, iteration: int,
                               child_fingerprint: str | None,
                               rule_name: str,
                               accepted: bool,
                               reason_code: str | None = None,
                               reason_detail: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO constraint_events "
            "(iteration, child_fingerprint, rule_name, accepted, "
            "reason_code, reason_detail, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                int(iteration),
                child_fingerprint,
                rule_name,
                1 if accepted else 0,
                reason_code,
                reason_detail,
                time.time(),
            ),
        )

    def constraint_rejection_rate(self, window: int,
                                  rule: str | None = None) -> float:
        """Fraction of the most recent `window` constraint_events (optionally
        filtered by rule) that were rejected. Returns 0.0 on empty window.
        """
        if rule is None:
            rows = self._conn.execute(
                "SELECT accepted FROM constraint_events "
                "ORDER BY created_at DESC LIMIT ?",
                (int(window),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT accepted FROM constraint_events WHERE rule_name = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (rule, int(window)),
            ).fetchall()
        if not rows:
            return 0.0
        rejected = sum(1 for r in rows if not r["accepted"])
        return rejected / len(rows)

    # --- Section 11: fitness-delta query helper ---

    def median_fitness_delta_per_island(self, window: int) -> float:
        """Compute median balanced_acc delta per island over the last
        `window` completed experiments per island, then return the median
        across islands. Returns 0.0 when there is insufficient data.
        """
        rows = self._conn.execute(
            "SELECT island_id, fitness_json, completed_at FROM experiments "
            "WHERE fitness_json IS NOT NULL "
            "ORDER BY completed_at ASC",
        ).fetchall()
        if not rows:
            return 0.0
        per_island: dict[int, list[float]] = {}
        for r in rows:
            try:
                acc = json.loads(r["fitness_json"]).get("balanced_acc")
            except (TypeError, ValueError):
                continue
            if isinstance(acc, (int, float)):
                per_island.setdefault(r["island_id"], []).append(float(acc))
        if not per_island:
            return 0.0
        per_island_medians: list[float] = []
        for accs in per_island.values():
            tail = accs[-int(window):] if len(accs) > window else accs
            if len(tail) < 2:
                continue
            deltas = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
            per_island_medians.append(statistics.median(deltas))
        if not per_island_medians:
            return 0.0
        return float(statistics.median(per_island_medians))
