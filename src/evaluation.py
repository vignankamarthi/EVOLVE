"""Classification metric suite for AI4Pain 2026.

Replaces the regression-style evaluation in
`Blood-Pressure-Inference-with-BVP/src/evaluation.py` (MAE / RMSE / R-squared
/ AAMI / BHS / Bland-Altman) with a 3-class classification suite required by
ANTIPATTERNS.md rule 6 (NO SILENT METRIC DROPS):

  - Balanced accuracy (primary)
  - Macro-F1 (secondary)
  - Per-class precision and recall for {NP, HP, AP}
  - Confusion matrix (3x3)
  - AUC (one-vs-rest)
  - Single-class collapse detector

Every evaluation function must produce all of these metrics. Single-number
summaries are forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .data_loader import LABEL_NAMES, N_CLASSES, INVERSE_LABEL_MAP
from .utils import atomic_json_write, get_logger


logger = get_logger("ai4pain_2026")


@dataclass
class ClassificationReport:
    balanced_accuracy: float
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: np.ndarray
    auc_ovr: float
    n_samples: int
    single_class_collapse: bool = False
    class_distribution_true: Dict[str, int] = field(default_factory=dict)
    class_distribution_pred: Dict[str, int] = field(default_factory=dict)

    def to_serializable(self) -> Dict:
        """Return a JSON-serializable dict (numpy arrays converted to lists)."""
        d = asdict(self)
        d["confusion_matrix"] = self.confusion_matrix.tolist()
        return d


def check_single_class_collapse(y_pred: np.ndarray) -> bool:
    """True if the predictor predicted exactly one class for every sample."""
    return len(np.unique(np.asarray(y_pred).ravel())) == 1


def class_count_dict(y: np.ndarray) -> Dict[str, int]:
    """Return ``{label_name: count}`` for all 3 classes (zero-padded)."""
    counts = {name: 0 for name in LABEL_NAMES}
    unique, freq = np.unique(y, return_counts=True)
    for u, c in zip(unique, freq):
        if 0 <= int(u) < N_CLASSES:
            counts[INVERSE_LABEL_MAP_PRETTY[int(u)]] = int(c)
    return counts


# Pretty mapping (NP -> "No Pain", etc.) so the JSON output is readable
INVERSE_LABEL_MAP_PRETTY = {
    0: "No Pain",
    1: "Hand Pain",
    2: "Arm Pain",
}


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> ClassificationReport:
    """Compute the full classification metric suite.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Integer labels in {0, 1, 2}.
    y_pred : array-like, shape (n,)
        Hard-label predictions in {0, 1, 2}.
    y_proba : array-like, shape (n, 3), optional
        Class probability scores for AUC-OVR. If None, AUC is reported as
        NaN. Most sklearn classifiers expose ``predict_proba``.

    Returns
    -------
    ClassificationReport

    Notes
    -----
    Macro-averaged precision/recall/F1 with ``zero_division=0`` because a
    single-class collapse must not silently raise; the collapse flag is set
    instead.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        average=None,
        zero_division=0,
    )
    per_class_precision = {LABEL_NAMES[i]: float(precision[i]) for i in range(N_CLASSES)}
    per_class_recall = {LABEL_NAMES[i]: float(recall[i]) for i in range(N_CLASSES)}

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            auc = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    labels=list(range(N_CLASSES)),
                    average="macro",
                )
            )
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    return ClassificationReport(
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion_matrix=cm,
        auc_ovr=auc,
        n_samples=int(len(y_true)),
        single_class_collapse=check_single_class_collapse(y_pred),
        class_distribution_true=class_count_dict(y_true),
        class_distribution_pred=class_count_dict(y_pred),
    )


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    results_dir: Path,
) -> Dict:
    """Evaluate one trained model and persist the report to ``results_dir``.

    Returns the serialized dict for downstream use (leaderboard generation).
    """
    report = evaluate_classifier(y_true, y_pred, y_proba)
    payload = {
        "model": model_name,
        "report": report.to_serializable(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_metrics.json"
    atomic_json_write(out_path, payload)

    collapse = " [COLLAPSED]" if report.single_class_collapse else ""
    logger.info(
        f"  {model_name}: bal_acc={report.balanced_accuracy:.4f}, "
        f"macro_f1={report.macro_f1:.4f}, AUC-OVR={report.auc_ovr:.4f}{collapse}"
    )
    return payload


def generate_leaderboard(results_dir: Path) -> Dict:
    """Build a leaderboard JSON ranking models by balanced accuracy."""
    import json

    entries = []
    for json_file in sorted(results_dir.glob("*_metrics.json")):
        if json_file.name == "leaderboard.json":
            continue
        with open(json_file) as f:
            payload = json.load(f)
        report = payload["report"]
        entries.append(
            {
                "model": payload["model"],
                "balanced_accuracy": report["balanced_accuracy"],
                "macro_f1": report["macro_f1"],
                "auc_ovr": report["auc_ovr"],
                "single_class_collapse": report["single_class_collapse"],
                "n_samples": report["n_samples"],
            }
        )

    entries.sort(key=lambda e: e["balanced_accuracy"], reverse=True)
    leaderboard = {
        "entries": entries,
        "best_model": entries[0]["model"] if entries else None,
    }
    atomic_json_write(results_dir / "leaderboard.json", leaderboard)
    logger.info(f"Leaderboard saved with {len(entries)} entries")
    return leaderboard
