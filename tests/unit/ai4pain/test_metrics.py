"""Tests for ai4pain.metrics."""
import math
import pytest
import numpy as np
from ai4pain import metrics


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    return arr / arr.sum(axis=1, keepdims=True)


def test_module_imports():
    assert callable(metrics.full_metric_suite)
    assert callable(metrics.expected_calibration_error)


def test_full_metric_suite_keys_present():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    proba = _normalize_rows(np.eye(3)[y_pred] * 0.7 + 0.1)
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    for k in ("balanced_acc", "macro_f1", "confusion_3x3", "auc_ovr", "ece",
              "per_class_pr"):
        assert k in out


def test_perfect_predictor():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y_true]
    out = metrics.full_metric_suite(y_true, y_true, proba)
    assert out["balanced_acc"] == pytest.approx(1.0)
    assert out["macro_f1"] == pytest.approx(1.0)


def test_balanced_acc_random_baseline_near_one_third():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=600)
    y_pred = rng.integers(0, 3, size=600)
    proba = rng.dirichlet([1, 1, 1], size=600)
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    assert 0.25 < out["balanced_acc"] < 0.42


def test_confusion_matrix_is_3x3_with_correct_total():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1])
    proba = np.eye(3)[y_pred]
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    cm = np.asarray(out["confusion_3x3"])
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == 4


def test_per_class_pr_keyed_by_label_names():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    proba = np.eye(3)
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    assert set(out["per_class_pr"].keys()) == {"NP", "AP", "HP"}


def test_full_metric_suite_rejects_wrong_proba_shape():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        metrics.full_metric_suite(y_true, y_pred, proba=np.eye(2))


def test_full_metric_suite_rejects_non_1d_labels():
    y_true = np.array([[0, 1, 2]])
    y_pred = np.array([[0, 1, 2]])
    proba = np.eye(3)
    with pytest.raises(ValueError):
        metrics.full_metric_suite(y_true, y_pred, proba)


def test_ece_zero_for_perfect_calibration():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y_true]
    ece = metrics.expected_calibration_error(y_true, proba)
    assert ece == pytest.approx(0.0)


def test_auc_ovr_handles_single_class_y_true_gracefully():
    """When y_true has only one class, AUC OvR is undefined. Return NaN."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 2])
    proba = _normalize_rows(np.eye(3)[y_pred] + 0.1)
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    assert math.isnan(out["auc_ovr"])
