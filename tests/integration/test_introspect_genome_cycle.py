"""Integration: Level 2 framework-genome mutation cycle.

FRAMEWORK.md Section 7. Exercises:
  introspect.validate_genome_mutation
  introspect.apply_genome_mutation
  ledger.write_framework_mutation

ANTIPATTERNS rules 12 + 13 enforced: out-of-bounds genome change rejected,
HIP-H gating is the caller's responsibility (this test simulates approval).
"""
from framework import introspect, ledger


def test_full_genome_cycle(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    led.init_schema()
    current_genome = {
        "island_count": 8, "novelty_alpha": 0.3, "curriculum_threshold": 0.55,
    }
    proposed = {
        "island_count": 10, "novelty_alpha": 0.4, "curriculum_threshold": 0.6,
    }
    assert introspect.validate_genome_mutation(proposed, current_genome) is None

    mutation = introspect.GenomeMutation(
        parent_hash="hash_a", child_hash="hash_b",
        description="raise diversity",
        parameter_changes=proposed,
        operator_changes={},
    )
    new_genome = introspect.apply_genome_mutation(mutation, current_genome)
    led.write_framework_mutation(
        mutation.parent_hash, mutation.child_hash, mutation.description,
    )
    assert new_genome["island_count"] == 10
    assert new_genome["novelty_alpha"] == 0.4
    assert new_genome["curriculum_threshold"] == 0.6
    led.close()


def test_genome_cycle_rejects_out_of_bounds():
    err = introspect.validate_genome_mutation(
        proposed={"island_count": 100},
        current={"island_count": 8},
    )
    assert err is not None
    assert "island_count" in err
