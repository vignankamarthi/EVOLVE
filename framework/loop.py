"""Main loop driver.

Advances the ledger by exactly ONE iteration per call. Designed to be invoked
from a Claude Code session turn, not a daemon. The mutation operator IS this
Claude Code session.

Per-iteration lifecycle (FRAMEWORK.md Section 8):

  1. Read ledger
  2. If population empty -> seed via framework.seeds, render the first
     runnable seed, return IterationPaused(HIP-D) so Vignan pushes it
  3. Else: sample parent via population.Islands.sample_parent
  4. Apply meta-stochastic state via framework.meta.step_meta_state
  5. Build mutation prompt via framework.mutation.assemble_mutation_prompt
     -> Claude Code session reads, reasons, writes a child spec
  6. Constraint check (framework.constraints)
     -> reject + regenerate if violated
  7. Render spec via framework.render
  8. Write experiments/<run_id>/, return IterationPaused(HIP-D)
  9. After cluster round-trip + result.json: caller invokes report_result,
     loop updates ledger + meta_state, returns IterationCompleted
 10. Every M iterations: trigger Level 2 introspection (HIP-H)

This first impl is intentionally narrow:
  - On empty ledger, seeds the first runnable seed and returns IterationPaused.
  - On subsequent calls, also returns IterationPaused with a synthetic run_id
    for the next program. The actual Claude-Code-driven mutation step is
    invoked by the caller from outside this module (it's a conversational
    turn, not a function call).

The integration tests exercise individual pieces; the loop driver becomes
fully autonomous once the Claude Code session wraps `advance_one_iteration`
in its per-iteration turn.
"""
from pathlib import Path
import math

from framework import seeds, render
from framework.ledger import Ledger
from framework.introspect import (
    DetectorConfig,
    GenomeMutation,
    propose_mutation,
    should_fire,
)


class IterationOutcome:
    """Base class for what an iteration returns."""


class IterationPaused(IterationOutcome):
    """Returned when the loop hits a HIP. Caller (Claude Code) handles."""

    def __init__(self, hip: str, run_id: str, action_required: str):
        self.hip = hip
        self.run_id = run_id
        self.action_required = action_required

    def __repr__(self) -> str:
        return (f"IterationPaused(hip={self.hip!r}, run_id={self.run_id!r}, "
                f"action_required={self.action_required!r})")


class IterationCompleted(IterationOutcome):
    """Returned when the iteration has fully closed (result.json read, ledger updated)."""

    def __init__(self, run_id: str, fitness_vector: dict):
        self.run_id = run_id
        self.fitness_vector = fitness_vector

    def __repr__(self) -> str:
        return (f"IterationCompleted(run_id={self.run_id!r}, "
                f"fitness_vector={self.fitness_vector!r})")


def _first_runnable_seed(specs: list[dict]) -> dict | None:
    for s in specs:
        family = s.get("model", {}).get("family")
        if family in render.FAMILY_ENTRY_POINTS:
            return s
    return None


def advance_one_iteration(experiments_root: Path = Path("experiments"),
                          ledger_path: Path = Path("ledger/experiments.db")
                          ) -> IterationOutcome:
    """One iteration of the loop. Returns IterationPaused at HIP-D for the
    caller (Claude Code + Vignan) to advance manually."""
    experiments_root = Path(experiments_root)
    experiments_root.mkdir(parents=True, exist_ok=True)

    led = Ledger(ledger_path)
    try:
        led.init_schema()
        existing = led.get_recent_iterations(n=1)

        if not existing:
            seed_spec = _first_runnable_seed(seeds.default_seed_specs())
            if seed_spec is None:
                return IterationPaused(
                    hip="HIP-CONFIG",
                    run_id="none",
                    action_required=(
                        "no seeds have a runnable family in "
                        "framework.render.FAMILY_ENTRY_POINTS. Add at least "
                        "one family before running the loop."),
                )
            rid = led.allocate_run_id()
            run_dir = experiments_root / rid
            render.render_spec_to_code(seed_spec, run_dir)
            led.write_experiment(rid, seed_spec, parent_id=None, island_id=0)
            return IterationPaused(
                hip="HIP-D",
                run_id=rid,
                action_required=(
                    f"sbatch --array=0-0 --export=ALL,MANIFEST="
                    f"{run_dir.parent}/manifest.json scripts/run_array.slurm "
                    f"(generation and training happen on cluster; see PLAN.md)"),
            )

        # Subsequent calls: this minimal impl returns IterationPaused
        # waiting for the next manual cluster step. The full mutation loop
        # (parent select, prompt build, constraint check, render) is
        # exercised by integration tests and orchestrated by the Claude Code
        # session in turn-by-turn use.
        return IterationPaused(
            hip="HIP-D",
            run_id="next_pending",
            action_required=(
                "next-iteration generation handled by the Claude Code "
                "session turn; this minimal driver pauses at HIP-D"),
        )
    finally:
        led.close()


def report_result(run_id: str, fitness_vector: dict,
                  ledger_path: Path = Path("ledger/experiments.db")
                  ) -> IterationCompleted:
    """Caller invokes this after HIP-F brings result.json back. Updates the
    ledger row for `run_id` with the fitness vector and returns
    IterationCompleted.
    """
    led = Ledger(ledger_path)
    try:
        led.init_schema()
        led.write_result(run_id, fitness_vector)
    finally:
        led.close()
    return IterationCompleted(run_id=run_id, fitness_vector=fitness_vector)


# --- Section 11 observability + Section 7 Level 2 wiring ---


def record_mutation_attempt(iteration: int, run_id: str,
                            parent_run_ids: list[str],
                            prompt_context: str,
                            child_spec: dict,
                            fingerprint: str,
                            reasoning_summary: str,
                            accepted: bool,
                            run_dir: Path | None = None,
                            ledger_path: Path = Path("ledger/experiments.db"),
                            ) -> None:
    """Persist a mutation_trace row (SQLite + optional JSONL mirror at
    `<run_dir>/trace.jsonl`). Called by the Claude Code session each iteration
    after a child spec is emitted. Pass `run_dir` to land the JSONL alongside
    spec.json + run.py (the Section 11 canonical layout).
    """
    led = Ledger(ledger_path)
    try:
        led.init_schema()
        led.write_mutation_trace(
            iteration=iteration, run_id=run_id,
            parent_run_ids=parent_run_ids,
            prompt_context=prompt_context,
            child_spec=child_spec,
            fingerprint=fingerprint,
            reasoning_summary=reasoning_summary,
            accepted=accepted,
            run_dir=run_dir,
        )
    finally:
        led.close()


def record_constraint_check(iteration: int, fingerprint: str | None,
                            rule_name: str, accepted: bool,
                            reason_code: str | None = None,
                            reason_detail: str | None = None,
                            ledger_path: Path = Path("ledger/experiments.db"),
                            ) -> None:
    """Persist one constraint_event row. Called by the constraint pipeline
    for each rule check (rule_guards / ast_tabu / lineage_cap).
    """
    led = Ledger(ledger_path)
    try:
        led.init_schema()
        led.write_constraint_event(
            iteration=iteration, child_fingerprint=fingerprint,
            rule_name=rule_name, accepted=accepted,
            reason_code=reason_code, reason_detail=reason_detail,
        )
    finally:
        led.close()


def check_level2(ledger: Ledger, current_genome: dict, current_iter: int,
                 last_fire_iter: int = 0,
                 config: DetectorConfig | None = None,
                 ) -> tuple[bool, GenomeMutation | None, dict]:
    """Run the compound detector against ledger state. Returns
    (fired, proposal_or_None, signals_dict).

    `signals_dict` carries every quantity the detector consulted, so the
    Claude Code session can surface it in the HIP-H pause message to Vignan
    without re-querying.
    """
    cfg = config or DetectorConfig()
    window = cfg.window
    median_delta = ledger.median_fitness_delta_per_island(window)
    rejection_rate = ledger.constraint_rejection_rate(window)
    h = ledger.fingerprint_entropy(window)
    h_max = math.log2(window) if window > 1 else 1.0
    entropy_ratio = (h / h_max) if h_max > 0 else 1.0
    fired = should_fire(
        current_iter=current_iter, last_fire_iter=last_fire_iter,
        median_delta=median_delta, rejection_rate=rejection_rate,
        entropy_ratio=entropy_ratio, config=cfg,
    )
    signals = {
        "current_iter": current_iter,
        "last_fire_iter": last_fire_iter,
        "window": window,
        "median_fitness_delta": median_delta,
        "constraint_rejection_rate": rejection_rate,
        "fingerprint_entropy": h,
        "entropy_ratio": entropy_ratio,
    }
    if not fired:
        return False, None, signals
    proposal = propose_mutation(ledger, current_genome, current_iter, cfg)
    return True, proposal, signals
