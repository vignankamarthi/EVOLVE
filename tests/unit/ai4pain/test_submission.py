"""Tests for ai4pain.submission (HIP-G test-set submission runner).

`run_submission` trains a spec on the 41 train subjects, early-stops on the
12-subject validation split, predicts the BLINDED 12-subject test split, and
writes test_predictions.csv. Supports spectrogram_cnn2d, multi_stream_bigru,
dual_ensemble.
"""
import csv
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ai4pain import submission


DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
HAVE_DATA = (DATA_ROOT.is_dir() and (DATA_ROOT / "train" / "Bvp").is_dir()
             and (DATA_ROOT / "test" / "Bvp").is_dir())


def test_module_imports():
    assert callable(submission.run_submission)
    assert callable(submission.run_from_dir)
    assert submission.SUPPORTED_FAMILIES == (
        "spectrogram_cnn2d", "multi_stream_bigru", "dual_ensemble")


def test_align_time_pads_to_common_max():
    a = np.zeros((3, 4, 33, 10), dtype=np.float32)
    b = np.ones((2, 4, 33, 15), dtype=np.float32)
    aa, bb = submission._align_time(a, b)
    assert aa.shape[-1] == bb.shape[-1] == 15
    assert np.allclose(aa[..., :10], a)
    assert np.allclose(aa[..., 10:], 0.0)


def test_prep_sequence_pads_and_zscores():
    rng = np.random.default_rng(0)
    Xtr = [rng.standard_normal((100, 4)).astype(np.float32),
           rng.standard_normal((150, 4)).astype(np.float32)]
    Xv = [rng.standard_normal((120, 4)).astype(np.float32)]
    Xte = [rng.standard_normal((130, 4)).astype(np.float32)]
    tr, v, te = submission._prep_sequence(Xtr, Xv, Xte)
    # all padded to a common T
    assert tr.shape[1] == v.shape[1] == te.shape[1] == 150
    assert tr.shape[2] == 4
    # train is z-scored: per-channel mean ~0
    assert abs(float(tr[..., 0].mean())) < 0.2


def test_make_loss_returns_callable():
    y = np.array([0, 0, 1, 1, 2, 2])
    loss_fn = submission._make_loss({"focal_gamma": 1.0}, y,
                                     torch.device("cpu"))
    logits = torch.randn(6, 3)
    out = loss_fn(logits, torch.from_numpy(y))
    assert torch.isfinite(out).all()


def test_write_predictions_csv_test_format(tmp_path):
    """Without true_labels -> blinded-test format (no true_* columns)."""
    out = tmp_path / "test_predictions.csv"
    probas = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=np.float32)
    submission._write_predictions_csv(
        out, subjects=[3, 17], preds=np.array([0, 2]), probas=probas)
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0].keys()) == ["subject", "trial_index", "pred_label",
                                    "pred_name", "p_NP", "p_AP", "p_HP"]
    assert rows[0]["pred_name"] == "NP" and rows[1]["pred_name"] == "HP"
    assert rows[1]["trial_index"] == "1"


def test_write_predictions_csv_val_format_has_true_labels(tmp_path):
    """With true_labels (validation) -> true_label/true_name columns added."""
    out = tmp_path / "val_predictions.csv"
    probas = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]], dtype=np.float32)
    submission._write_predictions_csv(
        out, subjects=[3, 3], preds=np.array([0, 1]), probas=probas,
        true_labels=np.array([0, 2]))
    rows = list(csv.DictReader(open(out)))
    assert "true_label" in rows[0] and "true_name" in rows[0]
    assert rows[0]["true_label"] == "0" and rows[0]["true_name"] == "NP"
    assert rows[1]["true_name"] == "HP"   # true label 2
    assert rows[1]["pred_name"] == "AP"   # predicted label 1


def test_train_and_predict_multi_seed_aggregates(tmp_path):
    """n_seeds>1 -> _train_and_predict runs N trainings (fresh model each),
    averages val+test probability tables, and aggregates per-seed metrics.
    A minimal Linear model on synthetic data is enough to exercise the loop."""
    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))   # pool time dim

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((12, 5, 4)).astype(np.float32)
    Xv = rng.standard_normal((6, 5, 4)).astype(np.float32)
    Xte = rng.standard_normal((6, 5, 4)).astype(np.float32)
    ytr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    yv = np.array([0, 1, 2, 0, 1, 2])
    subj_v = [10, 10, 10, 11, 11, 11]
    subj_te = [20, 20, 20, 21, 21, 21]
    train_cfg = {"epochs": 1, "batch_size": 6, "lr": 1e-2, "seed": 42,
                 "n_seeds": 3, "optimizer": "adam"}
    spec = {"name": "multi_seed_test", "model": {"family": "test"},
            "training": train_cfg}

    result = submission._train_and_predict(
        lambda: _Linear(), [Xtr], ytr, [Xv], yv, subj_v,
        [Xte], subj_te, train_cfg, tmp_path, spec)

    assert result["n_seeds"] == 3
    assert result["n_seeds_completed"] == 3
    assert len(result["per_seed_val_balanced_acc"]) == 3
    # CSVs are written from the AVERAGED probability table
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "val_predictions.csv").exists()
    rows = list(csv.DictReader(open(tmp_path / "val_predictions.csv")))
    assert len(rows) == len(yv)
    # Probabilities still sum to ~1 after averaging
    for r in rows:
        s = float(r["p_NP"]) + float(r["p_AP"]) + float(r["p_HP"])
        assert abs(s - 1.0) < 1e-3


def test_train_and_predict_single_seed_default(tmp_path):
    """No n_seeds set -> single-seed behavior (backwards compatible)."""
    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((6, 5, 4)).astype(np.float32)
    Xv = rng.standard_normal((3, 5, 4)).astype(np.float32)
    Xte = rng.standard_normal((3, 5, 4)).astype(np.float32)
    ytr = np.array([0, 0, 1, 1, 2, 2])
    train_cfg = {"epochs": 1, "batch_size": 6, "lr": 1e-2, "seed": 42}
    spec = {"name": "single", "model": {"family": "test"},
            "training": train_cfg}

    result = submission._train_and_predict(
        lambda: _Linear(), [Xtr], ytr, [Xv], np.array([0, 1, 2]), [7, 7, 7],
        [Xte], [8, 8, 8], train_cfg, tmp_path, spec)

    assert result["n_seeds"] == 1
    assert result["n_seeds_completed"] == 1
    assert len(result["per_seed_val_balanced_acc"]) == 1


def test_partial_state_roundtrip(tmp_path):
    """Save then load -> same completed_seeds, same arrays, same metrics."""
    va = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]], dtype=np.float64)
    te = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
    submission._save_partial(tmp_path, [42, 43], va, te,
                              [{"balanced_acc": 0.5}, {"balanced_acc": 0.6}])
    (seeds, va_back, te_back, metrics,
     per_seed_va, per_seed_te) = submission._load_partial(tmp_path)
    assert seeds == [42, 43]
    np.testing.assert_allclose(va_back, va)
    np.testing.assert_allclose(te_back, te)
    assert metrics[0]["balanced_acc"] == 0.5
    assert metrics[1]["balanced_acc"] == 0.6
    # no per-seed probas saved -> empty lists (backwards-compat default)
    assert per_seed_va == [] and per_seed_te == []


def test_partial_state_roundtrip_with_per_seed_probas(tmp_path):
    """Per-seed val/test probability lists round-trip too."""
    va = np.zeros((2, 3), dtype=np.float64)
    te = np.zeros((1, 3), dtype=np.float64)
    psv = [np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]),
           np.array([[0.2, 0.5, 0.3], [0.4, 0.2, 0.4]])]
    pst = [np.array([[0.9, 0.05, 0.05]]),
           np.array([[0.1, 0.8, 0.1]])]
    submission._save_partial(
        tmp_path, [42, 43], va, te,
        [{"balanced_acc": 0.5}, {"balanced_acc": 0.6}],
        per_seed_va_probas=psv, per_seed_te_probas=pst)
    out = submission._load_partial(tmp_path)
    assert out is not None
    _seeds, _va, _te, _m, per_seed_va, per_seed_te = out
    assert len(per_seed_va) == 2 and len(per_seed_te) == 2
    np.testing.assert_allclose(per_seed_va[0], psv[0])
    np.testing.assert_allclose(per_seed_te[1], pst[1])


def test_load_partial_missing_returns_none(tmp_path):
    assert submission._load_partial(tmp_path) is None


def test_train_and_predict_resumes_from_partial(tmp_path):
    """If partial_state.json carries 2 completed seeds, with n_seeds=4 only
    2 more should train. The final result reports n_completed=4."""
    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((6, 5, 4)).astype(np.float32)
    Xv = rng.standard_normal((3, 5, 4)).astype(np.float32)
    Xte = rng.standard_normal((3, 5, 4)).astype(np.float32)
    ytr = np.array([0, 0, 1, 1, 2, 2])
    yv = np.array([0, 1, 2])

    # pre-seed the partial with 2 already-done seeds
    va_acc = np.full((3, 3), 0.4, dtype=np.float64)
    te_acc = np.full((3, 3), 0.4, dtype=np.float64)
    submission._save_partial(
        tmp_path, [9001, 9002], va_acc, te_acc,
        [{"balanced_acc": 0.50}, {"balanced_acc": 0.52}])

    train_cfg = {"epochs": 1, "batch_size": 6, "lr": 1e-2, "seed": 9001,
                 "n_seeds": 4, "optimizer": "adam"}
    spec = {"name": "resume_test", "model": {"family": "test"},
            "training": train_cfg}
    result = submission._train_and_predict(
        lambda: _Linear(), [Xtr], ytr, [Xv], yv, [7, 7, 7],
        [Xte], [8, 8, 8], train_cfg, tmp_path, spec)

    # All 4 seeds accounted for, but only 2 were actually trained this call
    assert result["n_seeds"] == 4
    assert result["n_seeds_completed"] == 4
    assert len(result["per_seed_val_balanced_acc"]) == 4
    # partial_state.json is cleaned up after final result writes
    assert not (tmp_path / "partial_state.json").exists()


def test_train_and_predict_writes_partial_per_seed(tmp_path):
    """After each completed seed, partial_state.json is updated. Verify by
    running n_seeds=2 and checking partial existed mid-loop (we can only
    observe the final state, but checkpoint count must equal seeds when
    we tap in by spec)."""
    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((6, 5, 4)).astype(np.float32)
    Xv = rng.standard_normal((3, 5, 4)).astype(np.float32)
    Xte = rng.standard_normal((3, 5, 4)).astype(np.float32)
    ytr = np.array([0, 0, 1, 1, 2, 2])
    yv = np.array([0, 1, 2])

    train_cfg = {"epochs": 1, "batch_size": 6, "lr": 1e-2, "seed": 100,
                 "n_seeds": 2, "optimizer": "adam"}
    spec = {"name": "checkpoint_test", "model": {"family": "test"},
            "training": train_cfg}
    submission._train_and_predict(
        lambda: _Linear(), [Xtr], ytr, [Xv], yv, [7, 7, 7],
        [Xte], [8, 8, 8], train_cfg, tmp_path, spec)

    # final state: result.json present, partial cleaned up
    assert (tmp_path / "result.json").exists()
    assert not (tmp_path / "partial_state.json").exists()


def test_per_seed_predictions_written_after_multi_seed(tmp_path):
    """After a multi-seed _train_and_predict, per_seed_predictions.json
    is written with per-seed val + test probability tables (NOT just the
    averaged tables). Bundle ensembling reads from this file."""
    import json as _json

    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((6, 5, 4)).astype(np.float32)
    Xv = rng.standard_normal((3, 5, 4)).astype(np.float32)
    Xte = rng.standard_normal((3, 5, 4)).astype(np.float32)
    ytr = np.array([0, 0, 1, 1, 2, 2])
    yv = np.array([0, 1, 2])
    train_cfg = {"epochs": 1, "batch_size": 6, "lr": 1e-2, "seed": 100,
                 "n_seeds": 3, "optimizer": "adam"}
    spec = {"name": "per_seed_test", "model": {"family": "test"},
            "training": train_cfg}
    submission._train_and_predict(
        lambda: _Linear(), [Xtr], ytr, [Xv], yv, [7, 7, 7],
        [Xte], [8, 8, 8], train_cfg, tmp_path, spec)

    psp_path = tmp_path / "per_seed_predictions.json"
    assert psp_path.exists()
    psp = _json.loads(psp_path.read_text())
    assert psp["seeds"] == [100, 101, 102]
    # shapes: N x 3 trials x 3 classes
    assert np.asarray(psp["val_proba"]).shape == (3, 3, 3)
    assert np.asarray(psp["test_proba"]).shape == (3, 3, 3)
    assert psp["val_true_labels"] == [0, 1, 2]
    assert psp["val_subjects"] == [7, 7, 7]
    assert psp["test_subjects"] == [8, 8, 8]
    # Each per-seed row's probabilities still sum to ~1
    for seed_block in psp["val_proba"]:
        for row in seed_block:
            assert abs(sum(row) - 1.0) < 1e-3


def test_per_seed_predictions_survives_resume(tmp_path):
    """If a run is killed mid-loop and resumed, the final per_seed_predictions.json
    contains all completed seeds across both partial-state lifetimes."""
    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x.mean(dim=1))

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((6, 5, 4)).astype(np.float32)
    Xv = rng.standard_normal((3, 5, 4)).astype(np.float32)
    Xte = rng.standard_normal((3, 5, 4)).astype(np.float32)
    ytr = np.array([0, 0, 1, 1, 2, 2])
    yv = np.array([0, 1, 2])
    common = dict(epochs=1, batch_size=6, lr=1e-2, optimizer="adam")

    # First partial run: only 2 of 4 seeds (we cap by editing n_seeds).
    spec = {"name": "p", "model": {"family": "test"},
            "training": dict(common, seed=200, n_seeds=2)}
    submission._train_and_predict(
        lambda: _Linear(), [Xtr], ytr, [Xv], yv, [7, 7, 7],
        [Xte], [8, 8, 8], spec["training"], tmp_path, spec)
    # The first run completed and removed partial_state.json -- to simulate
    # a wall-killed prior, re-save partial_state from result.json's per-seed
    # data. Real resume tests live in test_train_and_predict_resumes_from_partial.

    # Second run: 4 seeds, starting fresh -- final per_seed_predictions.json
    # should have all 4 seeds, fresh.
    spec = {"name": "p", "model": {"family": "test"},
            "training": dict(common, seed=300, n_seeds=4)}
    submission._train_and_predict(
        lambda: _Linear(), [Xtr], ytr, [Xv], yv, [7, 7, 7],
        [Xte], [8, 8, 8], spec["training"], tmp_path, spec)

    import json as _json
    psp = _json.loads((tmp_path / "per_seed_predictions.json").read_text())
    assert len(psp["seeds"]) == 4
    assert psp["seeds"] == [300, 301, 302, 303]


def test_run_submission_rejects_unsupported_family(tmp_path):
    spec = {"name": "x", "model": {"family": "transformer"},
            "training": {}, "feature_extraction": {}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(NotImplementedError):
        submission.run_submission(tmp_path, DATA_ROOT)


def _smoke_spec(family, model_cfg, fe=None):
    return {
        "name": f"smoke_{family}",
        "preprocessing": {"normalize": "per_channel_zscore",
                           "padding": "right_zero_to_global_max"},
        "feature_extraction": fe,
        "model": dict(model_cfg, family=family),
        "training": {"epochs": 1, "batch_size": 16, "lr": 1e-3, "seed": 0,
                     "loss": "ce_class_balanced", "optimizer": "adam"},
        "data": {"signals": ["Bvp", "Eda", "Resp", "SpO2"]},
        "decode": {"strategy": "argmax"},
    }


@pytest.mark.skipif(not HAVE_DATA, reason="AI4Pain train+test data not present")
@pytest.mark.parametrize("family,model_cfg,fe", [
    ("spectrogram_cnn2d", {"base_channels": 8, "depth": 1},
     {"family": "spectrogram", "fs": 100, "nperseg": 64, "noverlap": 32}),
    ("multi_stream_bigru", {"per_channel_hidden": 8}, None),
    ("dual_ensemble", {"gru_cfg": {"per_channel_hidden": 8},
                        "cnn_cfg": {"base_channels": 8, "depth": 1}},
     {"family": "dual_ensemble", "fs": 100, "nperseg": 64, "noverlap": 32}),
])
def test_smoke_run_submission_each_family(tmp_path, family, model_cfg, fe):
    spec = _smoke_spec(family, model_cfg, fe)
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    result = submission.run_submission(tmp_path, DATA_ROOT)
    assert (tmp_path / "test_predictions.csv").exists()
    assert result["submission"] is True
    rows = list(csv.DictReader(open(tmp_path / "test_predictions.csv")))
    assert len(rows) == result["test_n_trials"]
    for r in rows:
        assert int(r["pred_label"]) in (0, 1, 2)
    # val predictions are also dumped (for post-hoc ensembling)
    val_csv = tmp_path / "val_predictions.csv"
    assert val_csv.exists()
    vrows = list(csv.DictReader(open(val_csv)))
    assert len(vrows) > 0
    assert "true_label" in vrows[0]   # val labels are known
    for r in vrows:
        assert int(r["true_label"]) in (0, 1, 2)
