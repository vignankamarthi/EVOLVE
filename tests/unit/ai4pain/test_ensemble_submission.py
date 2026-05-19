"""Tests for ai4pain.ensemble_submission (HIP-G submission #5).

Averages per-trial class probabilities across the independently-trained
single-model submissions (#1-#4), argmaxes -> a soft-voting ensemble.
"""
import csv
import json
from pathlib import Path
import pytest

from ai4pain import ensemble_submission as ens


def _write_pred_csv(path: Path, rows: list[tuple]):
    """rows: (subject, trial_index, p_NP, p_AP, p_HP)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_index", "pred_label", "pred_name",
                    "p_NP", "p_AP", "p_HP"])
        names = ["NP", "AP", "HP"]
        for subj, ti, pnp, pap, php in rows:
            probs = [pnp, pap, php]
            lab = max(range(3), key=lambda k: probs[k])
            w.writerow([subj, ti, lab, names[lab], pnp, pap, php])


def test_module_imports():
    assert callable(ens.average_predictions)
    assert callable(ens.run_ensemble)


def test_average_predictions_averages_and_argmaxes(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    # trial 0: A says HP-ish, B says NP-ish -> average tips to ... compute
    _write_pred_csv(a, [(7, 0, 0.1, 0.2, 0.7), (7, 1, 0.8, 0.1, 0.1)])
    _write_pred_csv(b, [(7, 0, 0.6, 0.3, 0.1), (7, 1, 0.2, 0.7, 0.1)])
    rows = ens.average_predictions([a, b])
    assert len(rows) == 2
    # trial 0: avg = [0.35, 0.25, 0.40] -> HP (2)
    assert rows[0]["pred_label"] == 2
    assert abs(rows[0]["p_NP"] - 0.35) < 1e-9
    # trial 1: avg = [0.50, 0.40, 0.10] -> NP (0)
    assert rows[1]["pred_label"] == 0


def test_average_predictions_rejects_mismatched_trials(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_pred_csv(a, [(7, 0, 0.5, 0.3, 0.2)])
    _write_pred_csv(b, [(7, 1, 0.5, 0.3, 0.2)])  # different trial index
    with pytest.raises(ValueError):
        ens.average_predictions([a, b])


def test_run_ensemble_writes_predictions(tmp_path):
    # two component submission dirs
    c1 = tmp_path / "submission_01"
    c2 = tmp_path / "submission_02"
    _write_pred_csv(c1 / "test_predictions.csv",
                    [(7, 0, 0.7, 0.2, 0.1), (7, 1, 0.1, 0.8, 0.1)])
    _write_pred_csv(c2 / "test_predictions.csv",
                    [(7, 0, 0.5, 0.4, 0.1), (7, 1, 0.2, 0.7, 0.1)])
    run_dir = tmp_path / "submission_05"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "name": "ens", "model": {"family": "prediction_ensemble",
                                   "components": [str(c1), str(c2)]}}))
    result = ens.run_ensemble(run_dir)
    assert (run_dir / "test_predictions.csv").exists()
    assert result["ensemble"] is True
    assert result["test_n_trials"] == 2
    rows = list(csv.DictReader(open(run_dir / "test_predictions.csv")))
    assert rows[0]["pred_name"] == "NP"   # avg [0.6,0.3,0.1]
    assert rows[1]["pred_name"] == "AP"   # avg [0.15,0.75,0.1]


def test_average_predictions_weighted(tmp_path):
    """Weighted soft-vote: a heavy weight on the first component pulls the
    average toward its prediction. weights are normalized internally."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    # trial 0: A confidently NP, B confidently HP
    _write_pred_csv(a, [(7, 0, 0.9, 0.05, 0.05)])
    _write_pred_csv(b, [(7, 0, 0.05, 0.05, 0.9)])
    # uniform -> avg [0.475, 0.05, 0.475] -- a near-tie
    uniform = ens.average_predictions([a, b])
    # weight A 3:1 -> avg = [0.75*0.9+0.25*0.05, ...] -> NP dominates
    weighted = ens.average_predictions([a, b], weights=[3, 1])
    assert weighted[0]["pred_label"] == 0          # NP -- A's call wins
    assert weighted[0]["p_NP"] > uniform[0]["p_NP"]


def test_average_predictions_rejects_bad_weight_count(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_pred_csv(a, [(7, 0, 0.5, 0.3, 0.2)])
    _write_pred_csv(b, [(7, 0, 0.4, 0.4, 0.2)])
    with pytest.raises(ValueError):
        ens.average_predictions([a, b], weights=[1, 1, 1])  # 3 for 2 comps


def test_run_ensemble_honors_spec_weights(tmp_path):
    c1 = tmp_path / "submission_01"
    c2 = tmp_path / "submission_02"
    _write_pred_csv(c1 / "test_predictions.csv", [(7, 0, 0.9, 0.05, 0.05)])
    _write_pred_csv(c2 / "test_predictions.csv", [(7, 0, 0.05, 0.05, 0.9)])
    run_dir = tmp_path / "submission_05"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "name": "ens", "model": {"family": "prediction_ensemble",
                                   "components": [str(c1), str(c2)],
                                   "weights": [3, 1]}}))
    result = ens.run_ensemble(run_dir)
    assert result["weights"] == [3, 1]
    rows = list(csv.DictReader(open(run_dir / "test_predictions.csv")))
    assert rows[0]["pred_name"] == "NP"  # weight pulls toward c1


def _write_val_csv(path: Path, rows: list[tuple]):
    """rows: (subject, trial_index, true_name, p_NP, p_AP, p_HP).
    Mirrors submission.py's val_predictions.csv format (with true labels)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["NP", "AP", "HP"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_index", "true_label", "true_name",
                    "pred_label", "pred_name", "p_NP", "p_AP", "p_HP"])
        for subj, ti, tname, pnp, pap, php in rows:
            t = names.index(tname)
            probs = [pnp, pap, php]
            pl = max(range(3), key=lambda k: probs[k])
            w.writerow([subj, ti, t, tname, pl, names[pl], pnp, pap, php])


def test_score_val_ensemble_3class_and_binary(tmp_path):
    """3-class and binary accuracy are scored independently. Trial 4 is
    designed wrong for 3-class (true AP, ensemble picks HP) but RIGHT for
    binary (HP and AP both collapse to Pain) -- so the two metrics differ."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_val_csv(a, [(7, 0, "NP", 0.8, 0.1, 0.1), (7, 1, "AP", 0.1, 0.8, 0.1),
                       (7, 2, "HP", 0.1, 0.1, 0.8), (7, 3, "NP", 0.2, 0.7, 0.1),
                       (7, 4, "AP", 0.1, 0.3, 0.6)])
    _write_val_csv(b, [(7, 0, "NP", 0.7, 0.2, 0.1), (7, 1, "AP", 0.2, 0.7, 0.1),
                       (7, 2, "HP", 0.2, 0.2, 0.6), (7, 3, "NP", 0.3, 0.6, 0.1),
                       (7, 4, "AP", 0.2, 0.3, 0.5)])
    m = ens.score_val_ensemble([a, b])
    assert m["n"] == 5
    # t0/t1/t2 correct, t3 wrong (true NP, pred AP), t4 wrong (true AP, pred HP)
    assert abs(m["acc_3class"] - 0.6) < 1e-9
    # binary: only t3 mismatches (NoPain vs Pain); t4 is Pain-vs-Pain -> correct
    assert abs(m["acc_binary"] - 0.8) < 1e-9


def test_score_val_ensemble_weighted(tmp_path):
    """Weighting a component up changes which class the ensemble argmaxes."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write_val_csv(a, [(7, 0, "NP", 0.9, 0.05, 0.05)])   # A: confidently NP
    _write_val_csv(b, [(7, 0, "NP", 0.05, 0.05, 0.9)])   # B: confidently HP
    # weight A 3:1 -> ensemble picks NP -> correct
    m = ens.score_val_ensemble([a, b], weights=[3, 1])
    assert m["acc_3class"] == 1.0


def test_run_val_ensemble_writes_metrics(tmp_path):
    c1, c2 = tmp_path / "submission_01", tmp_path / "submission_02"
    _write_val_csv(c1 / "val_predictions.csv",
                   [(7, 0, "NP", 0.8, 0.1, 0.1), (7, 1, "AP", 0.1, 0.8, 0.1)])
    _write_val_csv(c2 / "val_predictions.csv",
                   [(7, 0, "NP", 0.7, 0.2, 0.1), (7, 1, "AP", 0.2, 0.7, 0.1)])
    run_dir = tmp_path / "submission_05"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "name": "ens", "model": {"family": "prediction_ensemble",
                                  "components": [str(c1), str(c2)],
                                  "weights": [3, 1]}}))
    m = ens.run_val_ensemble(run_dir)
    assert m["acc_3class"] == 1.0
    assert (run_dir / "val_ensemble_metrics.json").exists()


def test_run_val_ensemble_missing_val_csv_raises(tmp_path):
    run_dir = tmp_path / "submission_05"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "name": "ens", "model": {"family": "prediction_ensemble",
                                  "components": [str(tmp_path / "nope")]}}))
    with pytest.raises(FileNotFoundError):
        ens.run_val_ensemble(run_dir)


def test_run_ensemble_missing_component_raises(tmp_path):
    run_dir = tmp_path / "submission_05"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({
        "name": "ens", "model": {"family": "prediction_ensemble",
                                   "components": [str(tmp_path / "nope")]}}))
    with pytest.raises(FileNotFoundError):
        ens.run_ensemble(run_dir)
