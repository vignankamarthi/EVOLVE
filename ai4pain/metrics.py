"""AI4Pain 2026 metrics suite.

ANTIPATTERNS rule 7: every fitness vector reports the full suite. No
single-number summary is ever sufficient on a 3-class imbalanced problem.

Returns:
  balanced_acc, macro_f1, per_class_pr (dict of (precision, recall) per class
                                         keyed by NP / AP / HP),
  confusion_3x3 (3x3 list of lists), auc_ovr (one-vs-rest), ece (calibration),
  binary (Pain-vs-No-Pain block derived from the 3-class result).

The `binary` block is the 3-class prediction projected down: No Pain = Baseline
(label 0), Pain = ARM (1) U HAND (2). It is a free diagnostic -- the same model,
no separate training. binary AUC collapses PROBABILITIES (P(pain) = p_AP + p_HP)
rather than the argmax so soft confidence is preserved. The 3-class
balanced_acc remains the primary fitness / challenge metric; binary is the
"can it detect pain at all" lens.

`param_count`, `train_seconds`, `inference_seconds`, `generalization_gap`
are merged at the call site since they're not derivable from (y_true, y_pred, proba).
"""
import math
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)


CLASSES = [0, 1, 2]
LABEL_NAMES = ["NP", "AP", "HP"]


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                                n_bins: int = 10) -> float:
    """ECE on max-class confidence (Naeini et al. 2015).

    Bins predictions by max-class confidence, computes |bin_accuracy - bin_confidence|
    weighted by bin frequency.
    """
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        bin_n = int(in_bin.sum())
        if bin_n == 0:
            continue
        bin_acc = float(accuracies[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (bin_n / n) * abs(bin_acc - bin_conf)
    return float(ece)


def full_metric_suite(y_true: np.ndarray, y_pred: np.ndarray,
                      proba: np.ndarray) -> dict:
    """Compute the full metrics suite for a 3-class problem.

    Args:
        y_true: shape (n,) int labels in {0, 1, 2}.
        y_pred: shape (n,) int predictions in {0, 1, 2}.
        proba: shape (n, 3) predicted class probabilities, each row sums to 1.

    Returns:
        dict with keys: balanced_acc, macro_f1, per_class_pr (NP / AP / HP),
        confusion_3x3, auc_ovr, ece.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    if proba.shape != (y_true.shape[0], 3):
        raise ValueError(
            f"proba must have shape ({y_true.shape[0]}, 3), got {proba.shape}")

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                              labels=CLASSES, zero_division=0))
    per_p = precision_score(y_true, y_pred, average=None,
                            labels=CLASSES, zero_division=0)
    per_r = recall_score(y_true, y_pred, average=None,
                          labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()

    # AUC OvR. Requires at least 2 distinct classes in y_true to be defined.
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", labels=CLASSES))
    except ValueError:
        auc = math.nan

    ece = expected_calibration_error(y_true, proba)

    return {
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "per_class_pr": {
            LABEL_NAMES[i]: (float(per_p[i]), float(per_r[i]))
            for i in range(3)
        },
        "confusion_3x3": cm,
        "auc_ovr": auc,
        "ece": ece,
        "binary": _binary_block(y_true, y_pred, proba),
    }


def _binary_block(y_true: np.ndarray, y_pred: np.ndarray,
                  proba: np.ndarray) -> dict:
    """Pain-vs-No-Pain metrics projected from the 3-class result.

    No Pain = Baseline (label 0); Pain = ARM (1) U HAND (2). AUC collapses
    probabilities: P(pain) = p_AP + p_HP. This is a diagnostic lens, not the
    fitness metric.
    """
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    p_pain = proba[:, 1] + proba[:, 2]

    bin_bal = float(balanced_accuracy_score(y_true_bin, y_pred_bin))
    bin_f1 = float(f1_score(y_true_bin, y_pred_bin, pos_label=1,
                            zero_division=0))
    bin_p = float(precision_score(y_true_bin, y_pred_bin, pos_label=1,
                                  zero_division=0))
    bin_r = float(recall_score(y_true_bin, y_pred_bin, pos_label=1,
                               zero_division=0))
    bin_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist()
    try:
        bin_auc = float(roc_auc_score(y_true_bin, p_pain))
    except ValueError:
        bin_auc = math.nan

    return {
        "balanced_acc": bin_bal,
        "f1": bin_f1,
        "pain_precision": bin_p,
        "pain_recall": bin_r,
        "auc": bin_auc,
        "confusion_2x2": bin_cm,
    }
