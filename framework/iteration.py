"""Batch orchestrator: ties population + mutation + meta + ledger into one
helper that the Claude Code session calls each iteration.

Replaces the ad-hoc per-iteration scripts that hardcoded parent_rid choices
and mutations. The framework's stochastic-search APIs now actually run:
FunSearch tournament selection per island, real mutation prompts assembled
from island context and meta-state, ledger seeding for the population, and
a global per-child iteration counter for Level 2's bootstrap floor.

Public API:
  seed_population(ledger, seed_specs, island_count) -> list[run_id]
      One-time call before iter 1. Writes one seed per island (round-robin
      if seeds < islands). Each seed gets a placeholder fitness so
      tournament selection is well-defined.
  prepare_batch(ledger, island_count, tournament_size, rng_seed) -> list[entry]
      For each island, return a dict with island_id, parent_run_id,
      parent_spec, and the assembled mutation prompt (Markdown blob the
      Claude Code session reads).
  global_child_count(ledger) -> int
      Per-child global counter for ledger's iteration field.

Spec: FRAMEWORK.md Sections 2 + 6 + 8.
"""
from __future__ import annotations

import json
from typing import Any

from framework.ledger import Ledger
from framework.mutation import MetaState, assemble_mutation_prompt
from framework.population import Islands


# Placeholder fitness assigned to fresh seeds so tournament selection works.
# A real fitness lands once the seed is trained on cluster.
_SEED_PLACEHOLDER_FITNESS: dict[str, Any] = {
    "balanced_acc": 0.333,  # random baseline for 3-class
    "macro_f1": 0.0,
    "ece": 0.5,
    "param_count": 0,
    "train_seconds": 0.0,
    "generalization_gap": 0.0,
}


def seed_population(ledger: Ledger, seed_specs: list[dict],
                    island_count: int,
                    run_dirs: list[Path] | None = None) -> list[str]:
    """Place one seed per island into the ledger.

    Returns the list of allocated run_ids in island order (run_ids[i] is on
    island i). If `len(seed_specs) < island_count`, cycles the list to fill.

    Each seed gets:
      - An experiments row (parent_id=None, island_id=i)
      - A placeholder fitness (so tournament selection has a basis)
      - A mutation_traces row (so it counts in the global child counter)

    If `run_dirs[i]` is provided, the seed's trace.jsonl is mirrored there
    (alongside spec.json + run.py per the Section 11 canonical layout).
    """
    if island_count < 1:
        raise ValueError(f"island_count must be >= 1, got {island_count}")
    if not seed_specs:
        raise ValueError("seed_specs is empty")
    run_ids = []
    for i in range(island_count):
        spec = dict(seed_specs[i % len(seed_specs)])
        rid = ledger.allocate_run_id()
        ledger.write_experiment(rid, spec, parent_id=None, island_id=i)
        ledger.write_result(rid, _SEED_PLACEHOLDER_FITNESS)
        # Each seed also gets a mutation_trace so the global counter advances
        rd = run_dirs[i] if run_dirs and i < len(run_dirs) else None
        ledger.write_mutation_trace(
            iteration=i + 1, run_id=rid, parent_run_ids=[],
            prompt_context="(seed population, iter 0)",
            child_spec=spec,
            fingerprint=f"seed_island_{i}",
            reasoning_summary=(
                f"Initial seed for island {i}: family="
                f"{spec.get('model', {}).get('family', '?')}. "
                f"Placeholder fitness; real value lands after cluster train."
            ),
            accepted=True,
            run_dir=rd,
        )
        run_ids.append(rid)
    return run_ids


def _build_islands_from_ledger(ledger: Ledger, island_count: int,
                                rng_seed: int | None = None) -> Islands:
    """Reconstruct Islands in-memory state from ledger rows."""
    isl = Islands(m=island_count, k=12, reset_cadence=50, rng_seed=rng_seed)
    for i in range(island_count):
        members = ledger.get_island_members(i)
        for m in members:
            fit = m.get("fitness") or _SEED_PLACEHOLDER_FITNESS
            isl.seed(island_id=i, run_id=m["run_id"], fitness=fit)
    return isl


def _spec_for_run_id(ledger: Ledger, run_id: str) -> dict:
    """Lookup spec via get_island_members across all islands."""
    for i in range(16):  # max islands per genome rule guard
        for m in ledger.get_island_members(i):
            if m["run_id"] == run_id:
                return m["spec"]
    raise LookupError(f"run_id {run_id!r} not found in any island")


def _island_best_specs(ledger: Ledger, island_id: int, k: int = 5) -> list[dict]:
    """Top-k specs in an island by balanced_acc (descending)."""
    members = ledger.get_island_members(island_id)
    members_with_fitness = [m for m in members if m.get("fitness")]
    members_with_fitness.sort(
        key=lambda m: m["fitness"].get("balanced_acc", -1.0),
        reverse=True,
    )
    return [m["spec"] for m in members_with_fitness[:k]]


def prepare_batch(ledger: Ledger, island_count: int,
                  tournament_size: int = 3,
                  rng_seed: int | None = None,
                  meta_state: dict | None = None) -> list[dict]:
    """Build one batch of mutation prompts, one per island.

    Returns a list of length `island_count`. Each entry is a dict with:
      island_id: int
      parent_run_id: str  (selected via tournament)
      parent_spec: dict
      prompt: str  (Markdown blob via assemble_mutation_prompt)
      meta: MetaState  (the meta-stochastic state at this iteration)
    """
    isl = _build_islands_from_ledger(ledger, island_count, rng_seed=rng_seed)
    meta = MetaState(
        p_lit=float((meta_state or {}).get("p_lit", 0.5)),
        novelty_alpha=float((meta_state or {}).get("novelty_alpha", 0.3)),
        temperature=float((meta_state or {}).get("temperature", 0.7)),
        failure_boost_active=bool((meta_state or {}).get(
            "failure_boost_active", False)),
    )
    batch: list[dict] = []
    for island_id in range(island_count):
        try:
            parent_rid = isl.sample_parent(island_id,
                                            tournament_size=tournament_size)
        except ValueError:
            # Empty island; skip
            continue
        parent_spec = _spec_for_run_id(ledger, parent_rid)
        island_bests = _island_best_specs(ledger, island_id, k=5)
        # Recent rejected programs: pulled from constraint_events; for now,
        # pass empty list. Future enhancement: query rejected mutation_traces.
        recent_failures: list[dict] = []
        prompt = assemble_mutation_prompt(
            parent_spec=parent_spec,
            island_best_specs=island_bests,
            recent_failures=recent_failures,
            meta=meta,
        )
        batch.append({
            "island_id": island_id,
            "parent_run_id": parent_rid,
            "parent_spec": parent_spec,
            "prompt": prompt,
            "meta": meta,
        })
    return batch


def global_child_count(ledger: Ledger) -> int:
    """Per-child global counter. Equals current_iteration() in mutation_traces."""
    return ledger.current_iteration()
