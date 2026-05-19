"""Prediction-ensemble submission runner (HIP-G submission #5).

Submissions #1-#4 are independently-trained single models, each of which has
written a `test_predictions.csv` with per-trial class probabilities. This
runner averages those probabilities across the components and argmaxes --
a post-hoc soft-voting ensemble.

Why this (vs the joint dual_ensemble): joint training co-adapts the sub-models
and correlates their errors (iter_0018 showed dual_baseline == its components).
Independently-trained models decorrelate better, and a probability average is
a pure variance-reduction step -- exactly the val->test failure mode.

The spec.json lists the component submission dirs; no training happens here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ai4pain.baselines import _atomic_write_json

LABEL_NAMES = ["NP", "AP", "HP"]


def _read_predictions(csv_path: Path) -> dict[int, dict]:
    """Read a component test_predictions.csv -> {trial_index: row dict}."""
    out: dict[int, dict] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ti = int(row["trial_index"])
            out[ti] = {
                "subject": int(row["subject"]),
                "p": [float(row["p_NP"]), float(row["p_AP"]),
                      float(row["p_HP"])],
            }
    return out


def average_predictions(component_csvs: list[Path]) -> list[dict]:
    """Soft-vote: average per-trial class probabilities across components,
    argmax. All components must cover the same trial indices.

    Returns a list of per-trial dicts: subject, trial_index, pred_label,
    pred_name, p_NP, p_AP, p_HP (the averaged probabilities).
    """
    if not component_csvs:
        raise ValueError("no component csvs given")
    comps = [_read_predictions(Path(c)) for c in component_csvs]
    trial_ids = sorted(comps[0].keys())
    for c in comps[1:]:
        if sorted(c.keys()) != trial_ids:
            raise ValueError("component predictions cover different trials")

    rows = []
    n = len(comps)
    for ti in trial_ids:
        avg = [0.0, 0.0, 0.0]
        for c in comps:
            for k in range(3):
                avg[k] += c[ti]["p"][k] / n
        pred = max(range(3), key=lambda k: avg[k])
        rows.append({
            "subject": comps[0][ti]["subject"],
            "trial_index": ti,
            "pred_label": pred,
            "pred_name": LABEL_NAMES[pred],
            "p_NP": avg[0], "p_AP": avg[1], "p_HP": avg[2],
        })
    return rows


def run_ensemble(run_dir: Path, data_root: Path | None = None) -> dict:
    """Read spec.json's component dirs, average their test predictions,
    write run_dir/test_predictions.csv + result.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads((run_dir / "spec.json").read_text())
    components = spec.get("model", {}).get("components", [])
    if not components:
        raise ValueError("spec.model.components is empty")

    csvs = [Path(c) / "test_predictions.csv" for c in components]
    missing = [str(p) for p in csvs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"component predictions not found: {missing}. "
            f"Run submissions 1-4 first.")

    rows = average_predictions(csvs)
    pred_path = run_dir / "test_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_index", "pred_label", "pred_name",
                    "p_NP", "p_AP", "p_HP"])
        for r in rows:
            w.writerow([r["subject"], r["trial_index"], r["pred_label"],
                        r["pred_name"], f"{r['p_NP']:.4f}",
                        f"{r['p_AP']:.4f}", f"{r['p_HP']:.4f}"])

    counts = {LABEL_NAMES[c]: sum(1 for r in rows if r["pred_label"] == c)
              for c in range(3)}
    result = {
        "name": spec.get("name", "ensemble_submission"),
        "submission": True,
        "ensemble": True,
        "components": components,
        "test_n_trials": len(rows),
        "test_pred_class_counts": counts,
        "test_predictions_csv": str(pred_path),
        "spec": spec,
    }
    _atomic_write_json(run_dir / "result.json", result)
    print(f"[ensemble] averaged {len(components)} components -> {pred_path} "
          f"({len(rows)} trials)", flush=True)
    print(f"[ensemble] test class counts: {counts}", flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path | None = None) -> dict:
    return run_ensemble(Path(run_dir), data_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
