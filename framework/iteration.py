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

import json
import math
import random
from pathlib import Path

import numpy as np

# Cluster QOS hard cap: max simultaneous queued jobs per user on the
# NEU Explorer GPU partition. Any batch larger than this fails at
# sbatch with QOSMaxSubmitJobPerUserLimit. Mac-side helpers enforce
# this so the cluster step is always a single `sbatch --array=0-(N-1)%4`.
CLUSTER_BATCH_CAP = 8


from collections import Counter

from framework.breakdown import (
    CriticPopulation,
    stagnation_escalation,
    trigger_migration,
)
from framework.constraints import ConstraintViolation
from framework.fitness import (
    confidence_weighted,
    novelty_score,
    pareto_rank,
    scalar_score,
)
from framework.ledger import Ledger
from framework.meta import step_meta_state
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


def _composite_tournament(ledger: Ledger, island_id: int,
                           tournament_size: int, meta: MetaState,
                           ece_lambda: float, rng: random.Random) -> str:
    """Tournament selection using `framework.fitness.scalar_score`.

    Builds Pareto rank + novelty score per island member, then samples
    `tournament_size` candidates and picks the highest composite. `alpha`
    in scalar_score = meta.novelty_alpha. Falls back to raw-accuracy if any
    member lacks confusion_3x3 (e.g., seed placeholder fitness).
    """
    members = [m for m in ledger.get_island_members(island_id)
               if m.get("fitness")]
    if not members:
        raise ValueError(f"island {island_id} is empty")

    # If any member lacks a real confusion matrix, raw-accuracy fallback
    cms = [m["fitness"].get("confusion_3x3") for m in members]
    if any(cm is None for cm in cms):
        members.sort(key=lambda m: m["fitness"].get("balanced_acc", -math.inf),
                     reverse=True)
        size = min(tournament_size, len(members))
        candidates = rng.sample(members, size)
        return max(candidates,
                   key=lambda m: m["fitness"].get("balanced_acc", -math.inf)
                   )["run_id"]

    cms_np = [np.array(cm, dtype=np.float64) for cm in cms]
    # Pareto: minimize gap, params; maximize bal_acc (we negate it for the
    # minimize-everywhere convention)
    fitness_vectors = []
    for m in members:
        fv = m["fitness"]
        fitness_vectors.append({
            "neg_bal_acc": -float(fv.get("balanced_acc", 0.0)),
            "gap": float(fv.get("generalization_gap", 0.0)),
            "params": float(fv.get("param_count", 0)),
            "ece": float(fv.get("ece", 0.5)),
        })
    ranks = pareto_rank(
        fitness_vectors,
        axes=["neg_bal_acc", "gap", "params", "ece"],
        directions={"neg_bal_acc": "minimize", "gap": "minimize",
                    "params": "minimize", "ece": "minimize"},
    )

    # Composite score per member
    scored = []
    for i, m in enumerate(members):
        fv = m["fitness"]
        # Novelty = mean kNN distance to other members' confusion matrices
        others = [cms_np[j] for j in range(len(members)) if j != i]
        nov = novelty_score(cms_np[i], others, k=min(5, max(1, len(others))))
        s = scalar_score(
            pareto_rank_value=ranks[i],
            novelty=nov,
            accuracy=float(fv.get("balanced_acc", 0.0)),
            ece=float(fv.get("ece", 0.5)),
            alpha=meta.novelty_alpha,
            lam=ece_lambda,
        )
        scored.append((s, m["run_id"]))

    size = min(tournament_size, len(scored))
    candidates = rng.sample(scored, size)
    return max(candidates, key=lambda x: x[0])[1]


def _island_stagnation_gap(ledger: Ledger, island_id: int) -> int:
    """Count of completed children on this island since its last fitness
    improvement. 0 means the most-recent child IS the best."""
    members = [m for m in ledger.get_island_members(island_id)
               if m.get("fitness") and m["fitness"].get("balanced_acc") is not None]
    if len(members) < 2:
        return 0
    members.sort(key=lambda m: m.get("completed_at") or 0.0)
    best_so_far = -float("inf")
    last_improvement_idx = 0
    for i, m in enumerate(members):
        acc = m["fitness"]["balanced_acc"]
        if acc > best_so_far:
            best_so_far = acc
            last_improvement_idx = i
    return len(members) - 1 - last_improvement_idx


def _foreign_champion(ledger: Ledger, island_count: int,
                      origin_island_id: int) -> tuple[int, str, dict] | None:
    """Pick the highest-fitness member across all OTHER islands. Returns
    (foreign_island_id, foreign_run_id, foreign_spec) or None."""
    best_acc = -float("inf")
    best = None
    for j in range(island_count):
        if j == origin_island_id:
            continue
        for m in ledger.get_island_members(j):
            if not m.get("fitness"):
                continue
            acc = m["fitness"].get("balanced_acc", -float("inf"))
            if acc > best_acc:
                best_acc = acc
                best = (j, m["run_id"], m["spec"])
    return best


def _prompt_with_critic(prompt: str, critic_pop: list[dict]) -> str:
    """Append top-3 critic genomes to the mutation prompt."""
    if not critic_pop:
        return prompt
    top = critic_pop[:3]
    lines = ["", "## Hard cases from critic population (Hillis 1990)", ""]
    for c in top:
        g = c.get("genome", {})
        lines.append(
            f"- critic_id={c['critic_id']} fitness={c['fitness']:.3f} "
            f"subjects={g.get('subject_subset')} "
            f"channel_perm={g.get('channel_permutation')}"
        )
    lines.append("")
    lines.append("Prefer mutations that increase robustness on these hard cases.")
    return prompt + "\n" + "\n".join(lines)


def prepare_batch(ledger: Ledger, island_count: int,
                  tournament_size: int = 3,
                  rng_seed: int | None = None,
                  meta_state: dict | None = None,
                  composite_scoring: bool = False,
                  ece_lambda: float = 0.5,
                  evolve_meta: bool = False,
                  stagnation_patience: int | None = None,
                  migration_patience: int | None = None,
                  evolve_critics: bool = False,
                  critic_pop_size: int = 30,
                  include_critic_in_prompt: bool = False) -> list[dict]:
    """Build one batch of mutation prompts, one per island.

    Returns a list of length `island_count`. Each entry is a dict with:
      island_id, parent_run_id, parent_spec, prompt, meta (MetaState)

    Additional keys may be present:
      stagnant: True when island gap >= stagnation_patience
      migrated_from_island: foreign island id when migration triggered
      foreign_parent_run_id, foreign_parent_spec: the migration co-parent

    Wired Level 1 mechanisms:
      composite_scoring=True   -> fitness.scalar_score for tournament
      evolve_meta=True         -> step_meta_state across batches (persists to ledger)
      stagnation_patience=N    -> per-island stagnation_escalation when gap >= N
      migration_patience=N     -> trigger_migration with foreign co-parent when gap >= N
      evolve_critics=True      -> CriticPopulation.evolve_one + persist via ledger
      include_critic_in_prompt -> top critics appended to mutation prompt
    """
    rng = random.Random(rng_seed)

    # ----- Meta-state: load latest from ledger if present, else use input -----
    prior_meta = ledger.read_latest_meta_state() if evolve_meta else None
    if prior_meta is not None:
        base_meta = {
            "p_lit": prior_meta["p_lit"],
            "novelty_alpha": prior_meta["novelty_alpha"],
            "temperature": prior_meta["temperature"],
            "failure_boost_active": prior_meta["failure_boost"].get(
                "failure_boost_active", False),
        }
    else:
        base_meta = dict(meta_state or {})
        base_meta.setdefault("p_lit", 0.5)
        base_meta.setdefault("novelty_alpha", 0.3)
        base_meta.setdefault("temperature", 0.7)
        base_meta.setdefault("failure_boost_active", False)

    # Recent fitness deltas across the pool (for failure-aware boost)
    recent_traces = ledger.recent_mutation_traces(window=10)
    recent_deltas: list[float] = []
    recent_accs = []
    for t in recent_traces:
        # Lookup acc via experiments hydration; cheap enough here
        for i in range(island_count):
            for m in ledger.get_island_members(i):
                if m["run_id"] == t["run_id"] and m.get("fitness"):
                    recent_accs.append(m["fitness"].get("balanced_acc", 0.0))
                    break
    for i in range(1, len(recent_accs)):
        recent_deltas.append(recent_accs[i] - recent_accs[i - 1])

    if evolve_meta:
        # Pass genome's current values as relaxation defaults so Level 2
        # mutations (e.g., novelty_alpha 0.3 -> 0.45 -> 0.675) actually
        # persist instead of washing out to the hardcoded baseline.
        genome_defaults = {
            "temperature": base_meta.get("temperature", 1.0),
            "novelty_alpha": base_meta.get("novelty_alpha", 0.3),
            "tabu_k": 50,
            "lineage_cap": 5,
        }
        new_meta = step_meta_state(base_meta, recent_deltas=recent_deltas,
                                    rng=rng, defaults=genome_defaults)
        # Persist
        next_iter = (prior_meta["iteration"] + 1) if prior_meta else 1
        ledger.write_meta_state(
            iteration=next_iter,
            p_lit=new_meta.get("p_lit", 0.5),
            novelty_alpha=new_meta.get("novelty_alpha", 0.3),
            temperature=new_meta.get("temperature", 0.7),
            failure_boost={
                "failure_boost_active": new_meta.get(
                    "failure_boost_active", False),
            },
        )
        base_meta = new_meta

    global_meta = MetaState(
        p_lit=float(base_meta.get("p_lit", 0.5)),
        novelty_alpha=float(base_meta.get("novelty_alpha", 0.3)),
        temperature=float(base_meta.get("temperature", 0.7)),
        failure_boost_active=bool(base_meta.get("failure_boost_active", False)),
    )

    # ----- Critic population: maintain across batches -----
    critic_pop_data = ledger.read_critic_population() if (
        evolve_critics or include_critic_in_prompt) else []
    if evolve_critics:
        if not critic_pop_data:
            # Seed initial population
            critic_obj = CriticPopulation(size=critic_pop_size,
                                           rng_seed=rng_seed)
            for c in critic_obj._members:
                ledger.write_critic(
                    critic_id=c.critic_id, parent_id=None,
                    genome={
                        "subject_subset": c.subject_subset,
                        "signal_perturbation": c.signal_perturbation,
                        "channel_permutation": c.channel_permutation,
                    },
                    fitness=c.fitness,
                )
        else:
            # Evolve one new critic from a random parent
            parent = rng.choice(critic_pop_data)
            new_id = f"c_{rng.randint(0, 999999):06d}_{ledger.current_iteration()}"
            # Mutate channel permutation: small swap
            perm = list(parent["genome"].get("channel_permutation", [0, 1, 2, 3]))
            if len(perm) >= 2:
                i, j = rng.sample(range(len(perm)), 2)
                perm[i], perm[j] = perm[j], perm[i]
            new_genome = dict(parent["genome"])
            new_genome["channel_permutation"] = perm
            # Fitness placeholder: 1.0 - mean recent program acc (high = hard)
            mean_acc = (sum(recent_accs) / len(recent_accs)) if recent_accs else 0.4
            ledger.write_critic(
                critic_id=new_id, parent_id=parent["critic_id"],
                genome=new_genome, fitness=1.0 - mean_acc,
            )
        critic_pop_data = ledger.read_critic_population()

    # ----- Per-island parent selection + escalation + migration -----
    batch: list[dict] = []
    if not composite_scoring:
        isl = _build_islands_from_ledger(ledger, island_count,
                                          rng_seed=rng_seed)
    for island_id in range(island_count):
        try:
            if composite_scoring:
                parent_rid = _composite_tournament(
                    ledger, island_id, tournament_size,
                    meta=global_meta, ece_lambda=ece_lambda, rng=rng,
                )
            else:
                parent_rid = isl.sample_parent(island_id,
                                                tournament_size=tournament_size)
        except ValueError:
            continue
        parent_spec = _spec_for_run_id(ledger, parent_rid)
        island_bests = _island_best_specs(ledger, island_id, k=5)

        # Per-island local meta starts from global, may be escalated
        local_meta_dict = {
            "p_lit": global_meta.p_lit,
            "novelty_alpha": global_meta.novelty_alpha,
            "temperature": global_meta.temperature,
            "failure_boost_active": global_meta.failure_boost_active,
        }
        gap = _island_stagnation_gap(ledger, island_id)
        stagnant = False
        if stagnation_patience is not None and gap >= stagnation_patience:
            local_meta_dict = stagnation_escalation(
                island_id=island_id, patience=stagnation_patience,
                current_meta=local_meta_dict,
            )
            stagnant = True
        local_meta = MetaState(
            p_lit=float(local_meta_dict.get("p_lit", 0.5)),
            novelty_alpha=float(local_meta_dict.get("novelty_alpha", 0.3)),
            temperature=float(local_meta_dict.get("temperature", 0.7)),
            failure_boost_active=bool(local_meta_dict.get(
                "failure_boost_active", False)),
        )

        # Migration trigger
        foreign_info = None
        if migration_patience is not None and gap >= migration_patience:
            foreign = _foreign_champion(ledger, island_count, island_id)
            if foreign is not None:
                # trigger_migration returns a structured directive dict
                trigger_migration(
                    stagnant_island_id=island_id,
                    champion_run_id=parent_rid,
                    foreign_champion_run_id=foreign[1],
                )
                foreign_info = foreign

        recent_failures: list[dict] = []
        prompt = assemble_mutation_prompt(
            parent_spec=parent_spec,
            island_best_specs=island_bests,
            recent_failures=recent_failures,
            meta=local_meta,
        )
        if include_critic_in_prompt:
            prompt = _prompt_with_critic(prompt, critic_pop_data)

        # crossover_pool: champion specs from OTHER islands, so graft_family
        # always has cross-family material (iter_0015 rebuild).
        crossover_pool: list[dict] = []
        for j in range(island_count):
            if j == island_id:
                continue
            j_bests = _island_best_specs(ledger, j, k=1)
            if j_bests:
                crossover_pool.append(j_bests[0])
            if len(crossover_pool) >= 3:
                break

        entry = {
            "island_id": island_id,
            "parent_run_id": parent_rid,
            "parent_spec": parent_spec,
            "prompt": prompt,
            "meta": local_meta,
            "stagnant": stagnant,
            "island_gap": gap,
            "crossover_pool": crossover_pool,
        }
        if foreign_info is not None:
            entry["migrated_from_island"] = foreign_info[0]
            entry["foreign_parent_run_id"] = foreign_info[1]
            entry["foreign_parent_spec"] = foreign_info[2]
        batch.append(entry)
    return batch


def global_child_count(ledger: Ledger) -> int:
    """Per-child global counter. Equals current_iteration() in mutation_traces."""
    return ledger.current_iteration()


# --------------------------------------------------------------------------
# family_quota: hard architectural-diversity constraint (iter_0015 rebuild)
# --------------------------------------------------------------------------
# Since iter_0012 the loop collapsed into a multi_stream_bigru monoculture.
# These functions force every batch to span >= min_families distinct model
# families with no family exceeding max_per_family. Same-family siblings must
# differ via a structural operator (the "blender" rule) -- see
# assign_blender_flags. The functions are standalone: the batch generator
# calls allocate_family_slots up front, then validate_batch_family_quota as a
# hard gate before the manifest is written.

def allocate_family_slots(n_children: int,
                          available_families: list[str],
                          recent_family_counts: dict[str, int],
                          rng: random.Random,
                          max_per_family: int = 3,
                          min_families: int = 4) -> list[str]:
    """Allocate a target model family to each of `n_children` batch slots.

    Guarantees: no family appears more than `max_per_family` times, and at
    least `min_families` distinct families appear. Families that dominated
    `recent_family_counts` are under-weighted so the loop self-corrects out
    of a monoculture.

    Raises ValueError if the quota is unsatisfiable for the given inputs.
    """
    fams = list(available_families)
    if min_families > len(fams):
        raise ValueError(
            f"min_families={min_families} exceeds available "
            f"families={len(fams)}")
    if min_families > n_children:
        raise ValueError(
            f"min_families={min_families} exceeds n_children={n_children}")
    if max_per_family * len(fams) < n_children:
        raise ValueError(
            f"max_per_family={max_per_family} x {len(fams)} families cannot "
            f"fill {n_children} slots")

    # Order families least-recently-used first; random tie-break.
    rng.shuffle(fams)
    fams.sort(key=lambda f: recent_family_counts.get(f, 0))

    counts: dict[str, int] = {f: 0 for f in fams}
    slots: list[str] = []
    # 1. Seed min_families distinct families (the least-used ones).
    for f in fams[:min_families]:
        slots.append(f)
        counts[f] += 1
    # 2. Fill the rest: least-assigned-so-far first, then least-recently-used,
    #    capped at max_per_family.
    while len(slots) < n_children:
        cand = [f for f in fams if counts[f] < max_per_family]
        cand.sort(key=lambda f: (counts[f], recent_family_counts.get(f, 0)))
        chosen = cand[0]
        slots.append(chosen)
        counts[chosen] += 1
    rng.shuffle(slots)
    return slots


def validate_batch_family_quota(manifest: dict,
                                 max_per_family: int = 3,
                                 min_families: int = 4
                                 ) -> ConstraintViolation | None:
    """Hard gate run BEFORE a manifest ships. Returns a ConstraintViolation if
    the realized batch violates the family quota, else None."""
    families = [e.get("family") for e in manifest.get("experiments", [])]
    counts = Counter(families)
    for fam, n in counts.items():
        if n > max_per_family:
            return ConstraintViolation(
                rule="family_quota",
                detail=(f"family {fam!r} appears {n} times "
                        f"(max_per_family={max_per_family})"))
    if len(counts) < min_families:
        return ConstraintViolation(
            rule="family_quota",
            detail=(f"batch spans {len(counts)} families "
                    f"(min_families={min_families})"))
    return None


def assign_blender_flags(slots: list[str]) -> list[bool]:
    """The "blender" rule: any family with >= 2 slots in the batch must have
    its siblings differ via a structural operator (swap_encoder, swap_fusion,
    add_aux_stream, graft_family) -- not hyperparameter knobs. Returns a
    per-slot bool: True means that slot requires a structural mutation."""
    counts = Counter(slots)
    return [counts[f] >= 2 for f in slots]
