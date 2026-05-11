"""Tests for framework.render. Spec: FRAMEWORK.md Section 2."""
import json
from pathlib import Path
import pytest
from framework import render


def test_module_imports():
    assert callable(render.render_spec_to_code)
    assert callable(render.fingerprint_spec)
    assert "bigru" in render.FAMILY_ENTRY_POINTS


def test_render_writes_run_py_and_spec_json(tmp_path: Path, sample_program_spec):
    out = render.render_spec_to_code(sample_program_spec, tmp_path)
    assert out.exists()
    assert out.name == "run.py"
    spec_path = tmp_path / "spec.json"
    assert spec_path.exists()
    assert json.loads(spec_path.read_text())["model"]["family"] == "bigru"


def test_render_run_py_dispatches_to_correct_entry_point(tmp_path: Path, sample_program_spec):
    out = render.render_spec_to_code(sample_program_spec, tmp_path)
    text = out.read_text()
    assert "from ai4pain.baselines import run_from_dir" in text
    assert "run_from_dir(args.run_dir, args.data_root)" in text


def test_render_run_py_finds_project_root_via_walk_up(tmp_path: Path, sample_program_spec):
    """The rendered run.py must locate framework/__init__.py by walking up
    from its own directory, NOT by hardcoded parents[N]. Verify by checking
    the generated source uses the walk-up pattern, not a fixed index."""
    out = render.render_spec_to_code(sample_program_spec, tmp_path / "deep" / "nested" / "dir")
    text = out.read_text()
    # Walk-up pattern present
    assert "while project_root != project_root.parent" in text
    assert 'framework / "__init__.py"' in text or 'framework' in text
    # Hardcoded parents[N] pattern absent
    assert ".parents[2]" not in text
    assert ".parents[3]" not in text


def test_render_rejects_unknown_family(tmp_path: Path, sample_program_spec):
    bad = dict(sample_program_spec)
    bad["model"] = {**bad["model"], "family": "alien"}
    with pytest.raises(ValueError):
        render.render_spec_to_code(bad, tmp_path)


def test_render_creates_missing_out_dir(tmp_path: Path, sample_program_spec):
    nested = tmp_path / "experiments" / "iter_0042"
    out = render.render_spec_to_code(sample_program_spec, nested)
    assert nested.is_dir()
    assert out.exists()


def test_fingerprint_is_deterministic(sample_program_spec):
    assert render.fingerprint_spec(sample_program_spec) == render.fingerprint_spec(sample_program_spec)


def test_fingerprint_ignores_numeric_hyperparams(sample_program_spec):
    spec1 = dict(sample_program_spec)
    spec1["model"] = {**spec1["model"], "hidden_size": 32, "num_layers": 1}
    spec2 = dict(sample_program_spec)
    spec2["model"] = {**spec2["model"], "hidden_size": 256, "num_layers": 1}
    assert render.fingerprint_spec(spec1) == render.fingerprint_spec(spec2)


def test_fingerprint_distinguishes_families():
    s1 = {"model": {"family": "bigru", "hidden_size": 64}}
    s2 = {"model": {"family": "transformer", "hidden_size": 64}}
    assert render.fingerprint_spec(s1) != render.fingerprint_spec(s2)


def test_fingerprint_distinguishes_operator_names(sample_program_spec):
    spec1 = dict(sample_program_spec)
    spec1["preprocessing"] = {**spec1["preprocessing"], "normalize": "standard"}
    spec2 = dict(sample_program_spec)
    spec2["preprocessing"] = {**spec2["preprocessing"], "normalize": "minmax"}
    assert render.fingerprint_spec(spec1) != render.fingerprint_spec(spec2)


def test_fingerprint_invariant_to_dict_key_order():
    s1 = {"model": {"family": "bigru"}, "training": {"lr": 1e-3}}
    s2 = {"training": {"lr": 1e-3}, "model": {"family": "bigru"}}
    assert render.fingerprint_spec(s1) == render.fingerprint_spec(s2)


def test_fingerprint_distinguishes_added_keys(sample_program_spec):
    spec1 = dict(sample_program_spec)
    spec2 = dict(sample_program_spec)
    spec2["model"] = {**spec2["model"], "attention_heads": 4}
    assert render.fingerprint_spec(spec1) != render.fingerprint_spec(spec2)
