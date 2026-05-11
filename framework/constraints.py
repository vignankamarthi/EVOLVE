"""Constraint layer.

Three mechanisms (FRAMEWORK.md Section 4) chained at the loop driver. Each
returns either None (clean) or a ConstraintViolation (rejected). The loop
asks the mutation operator to regenerate when any constraint fires.

  4.1 Rule guards
      Auto-reject programs that import pretrained loaders, fetch external data,
      or exceed resource caps. (Challenge rules, ANTIPATTERNS 1.)
  4.2 AST tabu
      Refuse near-duplicate structural fingerprints within last K accepted programs.
  4.3 Lineage inbreeding cap
      Reject child whose ancestry traces > N consecutive same-parent gens.

NOTE: the old curriculum_unlock filter (formerly 4.3) was removed on
2026-05-11 per Vignan's directive. Threshold-based unlocking proved brittle
for a dataset with a hard ceiling well below the 0.55 stage-1 threshold; we
were never going to unlock complex architectures. All families with entry
points are now eligible from iter 1. Pareto fitness handles param-count
vs. accuracy tradeoff in lieu of explicit gating.
"""
from dataclasses import dataclass


@dataclass
class ConstraintViolation:
    rule: str
    detail: str


BANNED_IMPORTS = frozenset({
    "transformers.from_pretrained",
    "torch.hub.load",
    "huggingface_hub",
    "urllib.request",
    "requests.get",
    "wget",
    "gdown",
    "datasets.load_dataset",
})


def _walk_strings(obj):
    """Yield every string value reachable from a nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


def rule_guards(spec: dict, max_params: int,
                max_train_seconds: int) -> ConstraintViolation | None:
    """Check banned imports and resource caps. Return violation or None.

    Banned-import check scans every string leaf in the spec; substring match
    against BANNED_IMPORTS catches both exact and prefixed forms (e.g.
    'huggingface_hub://something').

    Param and train-time caps are advisory at this layer (the real check
    happens at training time after the model is instantiated). We accept
    explicit hints in spec (`model.param_estimate`, `training.time_estimate_s`)
    and reject early if those exceed caps.
    """
    for s in _walk_strings(spec):
        for banned in BANNED_IMPORTS:
            if banned in s:
                return ConstraintViolation(
                    rule="rule_guards",
                    detail=f"banned string '{banned}' found in spec",
                )

    model = spec.get("model", {})
    if isinstance(model, dict):
        pe = model.get("param_estimate")
        if isinstance(pe, (int, float)) and pe > max_params:
            return ConstraintViolation(
                rule="rule_guards",
                detail=f"param_estimate {pe} > max_params {max_params}",
            )

    training = spec.get("training", {})
    if isinstance(training, dict):
        te = training.get("time_estimate_s")
        if isinstance(te, (int, float)) and te > max_train_seconds:
            return ConstraintViolation(
                rule="rule_guards",
                detail=f"time_estimate_s {te} > max_train_seconds {max_train_seconds}",
            )

    return None


def ast_tabu(spec_fingerprint: str,
             recent_fingerprints: list[str]) -> ConstraintViolation | None:
    """Refuse near-duplicates. recent_fingerprints is the last K accepted hashes."""
    if spec_fingerprint in recent_fingerprints:
        return ConstraintViolation(
            rule="ast_tabu",
            detail=f"fingerprint {spec_fingerprint[:12]}... appears in recent {len(recent_fingerprints)} accepted programs",
        )
    return None


def lineage_cap(parent_lineage: list[str],
                cap: int) -> ConstraintViolation | None:
    """Reject child if the lineage contains > cap consecutive same-parent ancestors.

    `parent_lineage` is the chain of parent run_ids from immediate parent
    backward through grandparent etc. Example: ['p1', 'p1', 'p1', 'p2', ...].
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")

    longest_run = 0
    current_run = 0
    prev = None
    for ancestor in parent_lineage:
        if ancestor == prev:
            current_run += 1
        else:
            current_run = 1
            prev = ancestor
        longest_run = max(longest_run, current_run)

    if longest_run > cap:
        return ConstraintViolation(
            rule="lineage_cap",
            detail=f"longest same-parent run {longest_run} > cap {cap}",
        )
    return None
