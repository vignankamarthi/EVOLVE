"""Tests for framework.constraints. Spec: FRAMEWORK.md Section 4."""
import pytest
from framework import constraints


def test_module_imports():
    assert "transformers.from_pretrained" in constraints.BANNED_IMPORTS
    assert callable(constraints.rule_guards)
    assert callable(constraints.ast_tabu)
    assert callable(constraints.lineage_cap)


def test_curriculum_unlock_removed_from_module():
    """Curriculum was scorched-earth removed on 2026-05-11. The function no
    longer exists; all families with entry points are eligible from iter 1.
    """
    assert not hasattr(constraints, "curriculum_unlock")
    assert not hasattr(constraints, "DEFAULT_CURRICULUM_THRESHOLDS")


# ---------- rule_guards ----------

def test_rule_guards_passes_clean_spec(sample_program_spec):
    v = constraints.rule_guards(sample_program_spec, max_params=10_000_000,
                                max_train_seconds=1800)
    assert v is None


def test_rule_guards_rejects_pretrained_loader(sample_program_spec):
    bad = dict(sample_program_spec)
    bad["model"] = {**bad["model"], "uses_import": "transformers.from_pretrained"}
    v = constraints.rule_guards(bad, max_params=10_000_000, max_train_seconds=1800)
    assert v is not None
    assert v.rule == "rule_guards"


def test_rule_guards_rejects_external_data_fetch(sample_program_spec):
    bad = dict(sample_program_spec)
    bad["preprocessing"] = {**bad["preprocessing"], "augmentation_source": "huggingface_hub://something"}
    v = constraints.rule_guards(bad, max_params=10_000_000, max_train_seconds=1800)
    assert v is not None


def test_rule_guards_rejects_param_estimate_over_cap(sample_program_spec):
    bad = dict(sample_program_spec)
    bad["model"] = {**bad["model"], "param_estimate": 20_000_000}
    v = constraints.rule_guards(bad, max_params=10_000_000, max_train_seconds=1800)
    assert v is not None
    assert "param" in v.detail


def test_rule_guards_rejects_train_time_estimate_over_cap(sample_program_spec):
    bad = dict(sample_program_spec)
    bad["training"] = {**bad["training"], "time_estimate_s": 9999}
    v = constraints.rule_guards(bad, max_params=10_000_000, max_train_seconds=1800)
    assert v is not None
    assert "time" in v.detail


# ---------- ast_tabu ----------

def test_ast_tabu_rejects_duplicate_fingerprint():
    v = constraints.ast_tabu("hash_x", recent_fingerprints=["hash_a", "hash_x", "hash_b"])
    assert v is not None
    assert v.rule == "ast_tabu"


def test_ast_tabu_passes_when_fingerprint_novel():
    v = constraints.ast_tabu("hash_new", recent_fingerprints=["hash_a", "hash_b"])
    assert v is None


def test_ast_tabu_handles_empty_recent_list():
    v = constraints.ast_tabu("hash_x", recent_fingerprints=[])
    assert v is None


# ---------- lineage_cap ----------

def test_lineage_cap_rejects_inbred_chain():
    chain = ["p", "p", "p", "p", "p", "p"]
    v = constraints.lineage_cap(chain, cap=5)
    assert v is not None
    assert v.rule == "lineage_cap"


def test_lineage_cap_allows_diverse_chain():
    chain = ["p1", "p2", "p1", "p3", "p2"]
    v = constraints.lineage_cap(chain, cap=5)
    assert v is None


def test_lineage_cap_allows_run_below_cap():
    chain = ["p", "p", "p"]
    v = constraints.lineage_cap(chain, cap=5)
    assert v is None


def test_lineage_cap_handles_empty_lineage():
    v = constraints.lineage_cap([], cap=5)
    assert v is None


def test_lineage_cap_rejects_cap_below_one():
    with pytest.raises(ValueError):
        constraints.lineage_cap(["p", "p"], cap=0)
