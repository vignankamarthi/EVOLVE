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
