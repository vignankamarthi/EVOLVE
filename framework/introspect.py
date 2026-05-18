"""Level 2 self-modification.

The framework genome is a JSON snapshot of every Level 1 parameter and
operator choice. Level 2 reads the ledger, decides whether to fire a
mutation (compound detector over stagnation / over-rejection / diversity
collapse), proposes a typed genome mutation, validates against the rule
guards, and (after HIP-H approval) applies the mutation and logs it.

Module surface:
  GENOME_RULE_GUARDS                  bound ranges per genome field
  DetectorConfig                      detector thresholds
  GenomeMutation                      dataclass persisted to framework_mutations
  GenomeConstraintViolation           raised when an out-of-bound mutation is applied
  compute_genome_hash(genome)         deterministic short-hash of a genome
  should_fire(...)                    pure compound detector
  propose_mutation(ledger, ...)       ledger -> typed mutation or None
  apply_genome_mutation(mutation, g)  produce new genome, raise on bound violation
  validate_genome_mutation(p, c)      None if valid, else "field=value outside [lo, hi]"
  assemble_introspection_prompt(...)  Markdown blob for Claude Code to read
  scale_param / add_island / drop_island /
  set_threshold / set_axis_weight / toggle_operator / swap_operator
                                      typed constructors (FRAMEWORK.md Section 9.5)

Spec: FRAMEWORK.md Section 7 (compound detector + 7 typed operators) and
Section 11 (observability surfaces this module reads).
Lineage: Schmidhuber Goedel Machines, AlphaEvolve (DeepMind 2025), POET 2019.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import math


GENOME_RULE_GUARDS: dict[str, tuple[float, float]] = {
    "island_count": (4, 16),
    "island_size": (5, 50),
    "reset_cadence": (20, 500),
    "novelty_alpha": (0.0, 0.8),
    "ece_lambda": (0.0, 2.0),
    "max_params": (10_000, 100_000_000),
    "max_train_seconds": (60, 28_800),
    "ast_tabu_k": (5, 200),
    "lineage_cap": (1, 20),
    "migration_patience": (3, 50),
    "critic_pop_size": (5, 100),
    "stagnation_patience": (3, 50),
    "p_lit_drift_sigma": (0.0, 0.2),
    "failure_boost_gain": (0.0, 5.0),
    "introspection_cadence_M": (10, 500),
    # iter_0015 framework rebuild: family-quota knobs. Level 2 tightens
    # max_per_family on family_monoculture.
    "max_per_family": (2, 8),
    "min_families": (2, 8),
}


class GenomeConstraintViolation(Exception):
    """Raised when apply_genome_mutation receives an out-of-bound mutation."""


@dataclass
class DetectorConfig:
    """Compound detector thresholds (FRAMEWORK.md Section 7.1)."""
    min_gap: int = 10
    max_gap: int = 100
    window: int = 20
    epsilon: float = 0.005
    max_rejection_rate: float = 0.6
    entropy_drop_ratio: float = 0.7
    bootstrap_floor: int = 20


@dataclass
class GenomeMutation:
    """Persisted form of a Level 2 mutation. `parameter_changes` carries the
    full field-level diff; `operator_changes` is parallel metadata that
    describes higher-level operator toggles/swaps for human inspection. The
    authoritative change set for `apply_genome_mutation` is
    `parameter_changes`.
    """
    parent_hash: str
    child_hash: str
    description: str
    parameter_changes: dict = field(default_factory=dict)
    operator_changes: dict = field(default_factory=dict)


def compute_genome_hash(genome: dict) -> str:
    """Deterministic 16-char SHA-256 prefix of the genome's canonical JSON."""
    canonical = json.dumps(genome, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def should_fire(current_iter: int, last_fire_iter: int,
                median_delta: float, rejection_rate: float,
                entropy_ratio: float,
                config: DetectorConfig | None = None) -> bool:
    """Compound detector. Returns True iff any pathology signal fires AND
    the gap guards permit it. See FRAMEWORK.md Section 7.1 diagram.
    """
    cfg = config or DetectorConfig()
    if current_iter < cfg.bootstrap_floor:
        return False
    iters_since = current_iter - last_fire_iter
    if iters_since < cfg.min_gap:
        return False
    if iters_since >= cfg.max_gap:
        return True
    if median_delta < cfg.epsilon:
        return True
    if rejection_rate > cfg.max_rejection_rate:
        return True
    if entropy_ratio < cfg.entropy_drop_ratio:
        return True
    return False


def detect_family_monoculture(ledger, window: int = 24,
                              threshold: float = 0.7) -> bool:
    """Family-monoculture detector (iter_0015 rebuild).

    Returns True when a single model family accounts for >= `threshold` of the
    last `window` experiments. This is the pathology that let the loop stall
    from iter_0012-0014 (24 consecutive multi_stream_bigru children). When it
    fires, propose_mutation tightens max_per_family to force diversity.
    """
    dist = ledger.recent_family_distribution(window)
    total = sum(dist.values())
    if total == 0:
        return False
    dominant = max(dist.values())
    return (dominant / total) >= threshold


# --- Typed operator constructors (FRAMEWORK.md Section 9.5) ---


def _clamp_to_guard(field_name: str, value: float) -> float:
    lo, hi = GENOME_RULE_GUARDS[field_name]
    return max(lo, min(hi, value))


def scale_param(parent_hash: str, current_genome: dict,
                field: str, factor: float) -> GenomeMutation:
    """SCALE_PARAM(field, factor in [0.5, 2.0]): multiply a numeric field
    and clamp to its rule-guard range. Raises ValueError on unknown field.
    """
    if field not in GENOME_RULE_GUARDS:
        raise ValueError(f"unknown field {field!r} (not in GENOME_RULE_GUARDS)")
    if field not in current_genome:
        raise ValueError(f"field {field!r} not present in current_genome")
    current_val = current_genome[field]
    if not isinstance(current_val, (int, float)):
        raise ValueError(f"field {field!r} is not numeric: {type(current_val).__name__}")
    new_val = _clamp_to_guard(field, current_val * factor)
    child = dict(current_genome)
    child[field] = new_val
    return GenomeMutation(
        parent_hash=parent_hash,
        child_hash=compute_genome_hash(child),
        description=f"SCALE_PARAM({field}, factor={factor:g}): {current_val} -> {new_val}",
        parameter_changes={field: new_val},
        operator_changes={},
    )


def add_island(parent_hash: str, current_genome: dict) -> GenomeMutation:
    """ADD_ISLAND: increment island_count, clamped at upper guard."""
    lo, hi = GENOME_RULE_GUARDS["island_count"]
    current = current_genome.get("island_count", 8)
    if current >= hi:
        raise ValueError(f"island_count {current} at cap {hi}, cannot ADD_ISLAND")
    new_val = current + 1
    child = dict(current_genome)
    child["island_count"] = new_val
    return GenomeMutation(
        parent_hash=parent_hash,
        child_hash=compute_genome_hash(child),
        description=f"ADD_ISLAND: island_count {current} -> {new_val}",
        parameter_changes={"island_count": new_val},
        operator_changes={},
    )


def drop_island(parent_hash: str, current_genome: dict,
                island_id: int) -> GenomeMutation:
    """DROP_ISLAND: decrement island_count, clamped at lower guard."""
    lo, hi = GENOME_RULE_GUARDS["island_count"]
    current = current_genome.get("island_count", 8)
    if current <= lo:
        raise ValueError(f"island_count {current} at floor {lo}, cannot DROP_ISLAND")
    new_val = current - 1
    child = dict(current_genome)
    child["island_count"] = new_val
    return GenomeMutation(
        parent_hash=parent_hash,
        child_hash=compute_genome_hash(child),
        description=f"DROP_ISLAND(island_id={island_id}): island_count {current} -> {new_val}",
        parameter_changes={"island_count": new_val},
        operator_changes={"dropped_island_id": island_id},
    )


def set_threshold(parent_hash: str, current_genome: dict,
                  name: str, value: float) -> GenomeMutation:
    """SET_THRESHOLD(name, value): set a numeric genome field to `value`,
    rejected if outside its rule-guard range.
    """
    if name not in GENOME_RULE_GUARDS:
        raise ValueError(f"threshold {name!r} not in GENOME_RULE_GUARDS")
    lo, hi = GENOME_RULE_GUARDS[name]
    if not (lo <= value <= hi):
        raise ValueError(f"{name}={value} outside [{lo}, {hi}]")
    child = dict(current_genome)
    child[name] = value
    return GenomeMutation(
        parent_hash=parent_hash,
        child_hash=compute_genome_hash(child),
        description=f"SET_THRESHOLD({name}, {value})",
        parameter_changes={name: value},
        operator_changes={},
    )


def set_axis_weight(parent_hash: str, current_genome: dict,
                    axis: str, weight: float) -> GenomeMutation:
    """SET_AXIS_WEIGHT(axis, weight in [0, 1]): re-weight a Pareto axis."""
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight {weight} outside [0, 1]")
    axis_weights = dict(current_genome.get("axis_weights", {}))
    axis_weights[axis] = weight
    child = dict(current_genome)
    child["axis_weights"] = axis_weights
    return GenomeMutation(
        parent_hash=parent_hash,
        child_hash=compute_genome_hash(child),
        description=f"SET_AXIS_WEIGHT({axis}, {weight})",
        parameter_changes={"axis_weights": axis_weights},
        operator_changes={"axis": axis, "weight": weight},
    )


def toggle_operator(parent_hash: str, current_genome: dict,
                    name: str, enabled: bool) -> GenomeMutation:
    """TOGGLE_OPERATOR(name, enabled): enable or disable a Level 1 component."""
    operators_enabled = dict(current_genome.get("operators_enabled", {}))
    operators_enabled[name] = bool(enabled)
    child = dict(current_genome)
    child["operators_enabled"] = operators_enabled
    return GenomeMutation(
        parent_hash=parent_hash,
        child_hash=compute_genome_hash(child),
        description=f"TOGGLE_OPERATOR({name}, enabled={enabled})",
        parameter_changes={"operators_enabled": operators_enabled},
        operator_changes={name: bool(enabled)},
    )


def swap_operator(parent_hash: str, current_genome: dict,
                  slot: str, name: str) -> GenomeMutation:
    """SWAP_OPERATOR(slot, name): replace an operator in a named slot."""
    operator_slots = dict(current_genome.get("operator_slots", {}))
    operator_slots[slot] = name
    child = dict(current_genome)
    child["operator_slots"] = operator_slots
    return GenomeMutation(
        parent_hash=parent_hash,
        child_hash=compute_genome_hash(child),
        description=f"SWAP_OPERATOR({slot}, {name})",
        parameter_changes={"operator_slots": operator_slots},
        operator_changes={slot: name},
    )


# --- Validate + apply ---


def validate_genome_mutation(proposed: dict, current: dict) -> str | None:
    """Return None if proposed parameter values stay in their rule guards,
    else a short string describing the violation. `current` is accepted for
    parity with the existing signature but no longer load-bearing.
    """
    for field_name, value in proposed.items():
        if field_name in GENOME_RULE_GUARDS:
            lo, hi = GENOME_RULE_GUARDS[field_name]
            if isinstance(value, (int, float)) and not (lo <= value <= hi):
                return f"{field_name}={value} outside [{lo}, {hi}]"
        if field_name == "axis_weights" and isinstance(value, dict):
            for axis, weight in value.items():
                if not (0.0 <= weight <= 1.0):
                    return f"axis_weights[{axis}]={weight} outside [0, 1]"
    return None


def apply_genome_mutation(mutation: GenomeMutation,
                          current_genome: dict) -> dict:
    """Apply a typed mutation to produce a new genome dict. Caller is
    responsible for HIP-H confirmation (ANTIPATTERNS 12) BEFORE invoking.
    Raises GenomeConstraintViolation if the parameter_changes leave any
    field outside its rule guard.
    """
    err = validate_genome_mutation(mutation.parameter_changes, current_genome)
    if err is not None:
        raise GenomeConstraintViolation(err)
    new_genome = dict(current_genome)
    for k, v in mutation.parameter_changes.items():
        if isinstance(v, dict) and isinstance(new_genome.get(k), dict):
            merged = dict(new_genome[k])
            merged.update(v)
            new_genome[k] = merged
        else:
            new_genome[k] = v
    return new_genome


# --- propose_mutation: ledger -> typed mutation ---


_CONSTRAINT_RULES = ("ast_tabu", "lineage_cap", "rule_guards")


def _entropy_ratio(ledger, window: int) -> float:
    h = ledger.fingerprint_entropy(window)
    h_max = math.log2(window) if window > 1 else 1.0
    return h / h_max if h_max > 0 else 1.0


def propose_mutation(ledger, current_genome: dict, current_iter: int,
                     config: DetectorConfig | None = None
                     ) -> GenomeMutation | None:
    """Read ledger state, pick the highest-priority pathology, return a
    typed GenomeMutation that addresses it. Returns None when nothing is
    broken (the loop is healthy).

    Priority chain (FRAMEWORK.md Section 7.3):
      1. over-rejection on any rule -> relax that rule
      2. stagnation -> SCALE_PARAM novelty_alpha up (fallback p_lit_drift_sigma)
      3. diversity collapse -> ADD_ISLAND or SET_AXIS_WEIGHT(novelty, +)
    """
    cfg = config or DetectorConfig()
    window = cfg.window
    parent_hash = compute_genome_hash(current_genome)

    # ----- Priority 0: family monoculture (iter_0015 rebuild) -----
    # Highest priority: a same-family monoculture means the search has stopped
    # exploring architectures entirely. Tighten max_per_family to force the
    # next batches to diversify.
    if detect_family_monoculture(ledger, window=24, threshold=0.7):
        current_mpf = current_genome.get("max_per_family", 3)
        mpf_lo, _ = GENOME_RULE_GUARDS["max_per_family"]
        if current_mpf > mpf_lo:
            return set_threshold(parent_hash, current_genome,
                                 "max_per_family", mpf_lo)

    # ----- Priority 1: over-rejection on any rule -----
    rates: list[tuple[float, str]] = []
    for rule in _CONSTRAINT_RULES:
        rates.append((ledger.constraint_rejection_rate(window, rule=rule), rule))
    rates.sort(reverse=True)
    top_rate, top_rule = rates[0]
    if top_rate > cfg.max_rejection_rate:
        if top_rule == "ast_tabu":
            return scale_param(parent_hash, current_genome, "ast_tabu_k", 0.5)
        if top_rule == "lineage_cap":
            return scale_param(parent_hash, current_genome, "lineage_cap", 1.5)
        if top_rule == "rule_guards":
            return scale_param(parent_hash, current_genome, "max_params", 1.5)

    # ----- Priority 2: stagnation -----
    median_delta = ledger.median_fitness_delta_per_island(window)
    if median_delta < cfg.epsilon:
        novelty_current = current_genome.get("novelty_alpha", 0.3)
        _, novelty_hi = GENOME_RULE_GUARDS["novelty_alpha"]
        if novelty_current < novelty_hi:
            return scale_param(parent_hash, current_genome,
                               "novelty_alpha", 1.5)
        return scale_param(parent_hash, current_genome,
                           "p_lit_drift_sigma", 1.5)

    # ----- Priority 3: diversity collapse -----
    entropy_ratio = _entropy_ratio(ledger, window)
    if entropy_ratio < cfg.entropy_drop_ratio:
        _, island_hi = GENOME_RULE_GUARDS["island_count"]
        if current_genome.get("island_count", 8) < island_hi:
            return add_island(parent_hash, current_genome)
        axis_weights = current_genome.get("axis_weights", {})
        new_w = min(1.0, axis_weights.get("novelty", 0.3) + 0.2)
        return set_axis_weight(parent_hash, current_genome, "novelty", new_w)

    return None


# --- Markdown prompt assembly ---


def assemble_introspection_prompt(ledger_recent: list,
                                  current_genome: dict,
                                  m_iter_window: int) -> str:
    """Return a Markdown blob the Claude Code session reads to propose a
    genome mutation. The diagram of sections matches FRAMEWORK.md Section 7.
    """
    lines: list[str] = []
    lines.append("# Level 2 Introspection Prompt")
    lines.append("")
    lines.append("## Recent fitness trajectory")
    if ledger_recent:
        lines.append(f"- Window: last {len(ledger_recent)} mutation traces (of {m_iter_window})")
        accepted = sum(1 for t in ledger_recent if t.get("accepted"))
        lines.append(f"- Accepted: {accepted} of {len(ledger_recent)}")
        fps = {t.get("fingerprint") for t in ledger_recent}
        lines.append(f"- Unique fingerprints: {len(fps)}")
    else:
        lines.append("- No data (cold start)")
    lines.append("")
    lines.append("## Novelty distribution")
    lines.append("- (computed by caller from confusion_matrix kNN; see Section 3.2)")
    lines.append("")
    lines.append("## Constraint rejection rates")
    lines.append("- (per rule; pulled from constraint_events table)")
    lines.append("")
    lines.append("## Per-island stagnation patterns")
    lines.append("- (per island; pulled from islands + experiments tables)")
    lines.append("")
    lines.append("## Critic-vs-program win rates")
    lines.append("- (from critic_population table once breakdown layer is running)")
    lines.append("")
    lines.append("## Current genome")
    lines.append("```json")
    lines.append(json.dumps(current_genome, indent=2, sort_keys=True, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Genome rule guards")
    lines.append("```json")
    lines.append(json.dumps({k: list(v) for k, v in GENOME_RULE_GUARDS.items()},
                            indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Mutation directive")
    lines.append("Propose ONE structural change to the genome that addresses the")
    lines.append("most prominent pathology in the metrics above. Stay inside")
    lines.append("GENOME_RULE_GUARDS (ANTIPATTERN 13) and do NOT disable any")
    lines.append("Level 1 hard constraint. Justify with one sentence per change.")
    return "\n".join(lines)
