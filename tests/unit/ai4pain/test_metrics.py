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


# ---------- binary (Pain vs No Pain) metrics, derived from 3-class ----------
# No Pain = Baseline (label 0); Pain = ARM (1) U HAND (2). The binary score is
# a projection of the 3-class prediction -- no separate model.


def test_binary_block_present():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y_pred]
    out = metrics.full_metric_suite(y_true, y_true, proba)
    assert "binary" in out
    for k in ("balanced_acc", "f1", "pain_precision", "pain_recall",
              "auc", "confusion_2x2"):
        assert k in out["binary"], f"missing binary.{k}"


def test_binary_perfect_when_3class_perfect():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y_true]
    out = metrics.full_metric_suite(y_true, y_true, proba)
    assert out["binary"]["balanced_acc"] == 1.0
    assert out["binary"]["pain_recall"] == 1.0
    assert out["binary"]["pain_precision"] == 1.0


def test_binary_collapses_arm_hand_confusion_to_correct():
    """A model that confuses ARM<->HAND but never confuses pain with no-pain
    should score 1.0 binary even though 3-class balanced_acc is below 1."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 2, 2, 1, 1])  # AP/HP swapped, NP perfect
    proba = np.eye(3)[y_pred]
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    assert out["binary"]["balanced_acc"] == 1.0       # pain detection perfect
    assert out["balanced_acc"] < 1.0                  # localization broken


def test_binary_confusion_is_2x2_with_correct_total():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 0, 2])
    proba = np.eye(3)[y_pred]
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    cm = out["binary"]["confusion_2x2"]
    assert len(cm) == 2 and len(cm[0]) == 2
    assert sum(sum(row) for row in cm) == len(y_true)


def test_binary_auc_uses_summed_pain_probability():
    """binary AUC must collapse PROBABILITIES (P(pain)=p_AP+p_HP), not argmax.
    A model with soft but correct pain ranking gets AUC 1.0."""
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 2])
    # No-pain rows: low pain mass; pain rows: high pain mass.
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6],
    ])
    out = metrics.full_metric_suite(y_true, y_pred, proba)
    assert out["binary"]["auc"] == 1.0
