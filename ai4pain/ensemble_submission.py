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

import numpy as np

LABEL_NAMES = ["NP", "AP", "HP"]


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically (tmp file + rename). Inlined so this module has
    NO torch dependency -- the ensemble is pure CSV arithmetic and must run
    instantly on a login node, not queue for a GPU."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


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


def average_predictions(component_csvs: list[Path],
                        weights: list[float] | None = None) -> list[dict]:
    """Weighted soft-vote: average per-trial class probabilities across
    components, argmax. All components must cover the same trial indices.

    `weights` -- per-component vote weights (any positive scale; normalized
    internally). None -> uniform. A heavier weight on a class-balanced
    component protects the classes the others starve (the AP-collapse fix).

    Returns a list of per-trial dicts: subject, trial_index, pred_label,
    pred_name, p_NP, p_AP, p_HP (the weighted-averaged probabilities).
    """
    if not component_csvs:
        raise ValueError("no component csvs given")
    comps = [_read_predictions(Path(c)) for c in component_csvs]
    trial_ids = sorted(comps[0].keys())
    for c in comps[1:]:
        if sorted(c.keys()) != trial_ids:
            raise ValueError("component predictions cover different trials")

    n = len(comps)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(
            f"got {len(weights)} weights for {n} components")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    w = [x / total for x in weights]  # normalize to sum 1

    rows = []
    for ti in trial_ids:
        avg = [0.0, 0.0, 0.0]
        for c, wc in zip(comps, w):
            for k in range(3):
                avg[k] += c[ti]["p"][k] * wc
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
    weights = spec.get("model", {}).get("weights")  # None -> uniform

    csvs = [Path(c) / "test_predictions.csv" for c in components]
    missing = [str(p) for p in csvs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"component predictions not found: {missing}. "
            f"Run submissions 1-4 first.")

    rows = average_predictions(csvs, weights=weights)
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
        "weights": weights,
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


def _read_true_labels(csv_path: Path) -> dict[int, int]:
    """Read {trial_index: true_label} from a val_predictions.csv."""
    out: dict[int, int] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[int(row["trial_index"])] = int(row["true_label"])
    return out


def score_val_ensemble(val_csvs: list[Path],
                       weights: list[float] | None = None) -> dict:
    """Score the weighted ensemble on the VALIDATION split.

    Each `val_predictions.csv` (from submission.py) carries per-trial class
    probabilities AND the true label. This weighted-averages the component
    probability tables, argmaxes, and scores against the true labels:

    - acc_3class: NP/AP/HP accuracy (the official challenge metric).
    - acc_binary: Pain-vs-No-Pain accuracy -- AP and HP both collapse to
      "Pain", so a 3-class miss inside {AP, HP} still counts as a binary hit.

    This is a label-backed val estimate; it costs no test submission.
    """
    rows = average_predictions([Path(c) for c in val_csvs], weights=weights)
    truth = _read_true_labels(Path(val_csvs[0]))
    n = len(rows)
    if n == 0:
        raise ValueError("no validation trials to score")
    correct_3 = correct_bin = 0
    for r in rows:
        true = truth[r["trial_index"]]
        pred = r["pred_label"]
        if pred == true:
            correct_3 += 1
        if (pred != 0) == (true != 0):   # 0 = No Pain, {1,2} = Pain
            correct_bin += 1
    return {
        "n": n,
        "acc_3class": correct_3 / n,
        "acc_binary": correct_bin / n,
        "weights": list(weights) if weights is not None else None,
    }


def run_val_ensemble(run_dir: Path) -> dict:
    """Read spec.json's component dirs, score the weighted ensemble on the
    validation split using each component's val_predictions.csv, write
    run_dir/val_ensemble_metrics.json. Pure CSV arithmetic -- no GPU."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    components = spec.get("model", {}).get("components", [])
    if not components:
        raise ValueError("spec.model.components is empty")
    weights = spec.get("model", {}).get("weights")  # None -> uniform

    val_csvs = [Path(c) / "val_predictions.csv" for c in components]
    missing = [str(p) for p in val_csvs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"component val predictions not found: {missing}. "
            f"Re-run submissions 1-4 (the runner now dumps val_predictions.csv).")

    metrics = score_val_ensemble(val_csvs, weights=weights)
    result = {
        "name": spec.get("name", "ensemble_submission"),
        "val_ensemble": True,
        "components": components,
        **metrics,
    }
    _atomic_write_json(run_dir / "val_ensemble_metrics.json", result)
    print(f"[ensemble-val] 3-class {metrics['acc_3class']:.4f}  "
          f"binary {metrics['acc_binary']:.4f}  (n={metrics['n']})",
          flush=True)
    return metrics


def _read_per_seed(path: Path) -> dict:
    """Read a component's per_seed_predictions.json -> dict with seeds,
    val_proba (N x T_val x 3), test_proba (N x T_test x 3), val_true_labels."""
    data = json.loads(Path(path).read_text())
    return {
        "seeds": list(data["seeds"]),
        "val_proba": data["val_proba"],
        "test_proba": data["test_proba"],
        "val_true_labels": data.get("val_true_labels"),
        "val_subjects": data.get("val_subjects"),
        "test_subjects": data.get("test_subjects"),
        "val_trial_indices": data.get("val_trial_indices"),
        "test_trial_indices": data.get("test_trial_indices"),
    }


def score_val_bundles(per_seed_json_paths: list[Path],
                      weights: list[float] | None = None) -> dict:
    """**Per-seed bundle ensembling** (the honest std methodology).

    For each seed i, form the weighted soft-vote across the K components'
    i-th-seed val probability tables -> bundle_i_val_proba -> argmax ->
    bundle_i accuracy against true val labels. N bundles -> mean +/- std
    for both 3-class and binary (Pain vs No Pain).

    All components MUST have the same seeds in the same order (the runner
    uses consecutive seeds starting at spec.training.seed; specs are
    re-runnable with the same seed list so alignment is structural).

    Returns dict with acc_*_mean, acc_*_std, per_bundle_acc_*, n_bundles.
    """
    comps = [_read_per_seed(p) for p in per_seed_json_paths]
    if not comps:
        raise ValueError("no components")
    K = len(comps)
    n_seeds = len(comps[0]["seeds"])
    for k, c in enumerate(comps[1:], start=1):
        if len(c["seeds"]) != n_seeds:
            raise ValueError(
                f"component {k} has {len(c['seeds'])} seeds; "
                f"component 0 has {n_seeds}. Bundle ensembling requires "
                f"matched seed counts across all components.")
    if weights is None:
        weights = [1.0] * K
    if len(weights) != K:
        raise ValueError(f"got {len(weights)} weights for {K} components")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    truth = np.asarray(comps[0]["val_true_labels"], dtype=int)
    n_val = len(truth)

    per_bundle_3 = []
    per_bundle_bin = []
    for i in range(n_seeds):
        # bundle_i_val_proba = sum_k w_k * p_k_val_seed_i
        bundle_val = np.zeros((n_val, 3), dtype=np.float64)
        for k in range(K):
            bundle_val += w[k] * np.asarray(comps[k]["val_proba"][i],
                                              dtype=np.float64)
        preds = bundle_val.argmax(axis=1)
        acc3 = float((preds == truth).mean())
        # binary: 0 = NP, {1,2} = Pain
        bin_preds = (preds != 0).astype(int)
        bin_truth = (truth != 0).astype(int)
        accbin = float((bin_preds == bin_truth).mean())
        per_bundle_3.append(acc3)
        per_bundle_bin.append(accbin)

    return {
        "n_bundles": n_seeds,
        "per_bundle_acc_3class": per_bundle_3,
        "per_bundle_acc_binary": per_bundle_bin,
        "acc_3class_mean": float(np.mean(per_bundle_3)),
        "acc_3class_std": float(np.std(per_bundle_3, ddof=0)),
        "acc_binary_mean": float(np.mean(per_bundle_bin)),
        "acc_binary_std": float(np.std(per_bundle_bin, ddof=0)),
        "weights": list(weights),
    }


def run_bundle_ensemble(run_dir: Path) -> dict:
    """End-to-end bundle ensemble: read spec.components' per_seed_predictions,
    score val bundles (mean +/- std), write averaged-bundle test
    predictions + val_ensemble_metrics.json. Pure CSV/JSON math, no torch."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    components = spec.get("model", {}).get("components", [])
    if not components:
        raise ValueError("spec.model.components is empty")
    weights = spec.get("model", {}).get("weights")

    psp_paths = [Path(c) / "per_seed_predictions.json" for c in components]
    missing = [str(p) for p in psp_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"per-seed prediction files not found: {missing}. "
            f"Re-run submissions 1-4 with the per-seed dump runner first.")

    val_metrics = score_val_bundles(psp_paths, weights=weights)

    # Test predictions: for each bundle i, compute bundle_i_test_proba; then
    # mean across bundles -> argmax. Linearity makes this equivalent to
    # averaging the component-averaged probabilities, but the methodological
    # origin is now per-seed-bundle (matches the val std story).
    comps = [_read_per_seed(p) for p in psp_paths]
    K = len(comps)
    n_seeds = len(comps[0]["seeds"])
    if weights is None:
        weights = [1.0] * K
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    n_test = len(comps[0]["test_proba"][0])
    bundle_test_mean = np.zeros((n_test, 3), dtype=np.float64)
    for i in range(n_seeds):
        bundle_i = np.zeros((n_test, 3), dtype=np.float64)
        for k in range(K):
            bundle_i += w[k] * np.asarray(comps[k]["test_proba"][i],
                                            dtype=np.float64)
        bundle_test_mean += bundle_i
    bundle_test_mean /= n_seeds
    test_preds = bundle_test_mean.argmax(axis=1)
    subjects_test = comps[0]["test_subjects"] or [0] * n_test

    pred_path = run_dir / "test_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["subject", "trial_index", "pred_label", "pred_name",
                        "p_NP", "p_AP", "p_HP"])
        for ti in range(n_test):
            p = int(test_preds[ti])
            w_csv.writerow([int(subjects_test[ti]), ti, p, LABEL_NAMES[p],
                            f"{bundle_test_mean[ti, 0]:.4f}",
                            f"{bundle_test_mean[ti, 1]:.4f}",
                            f"{bundle_test_mean[ti, 2]:.4f}"])

    counts = {LABEL_NAMES[c]: int((test_preds == c).sum()) for c in range(3)}
    result = {
        "name": spec.get("name", "ensemble_submission"),
        "bundle_ensemble": True,
        "components": components,
        "test_n_trials": int(n_test),
        "test_pred_class_counts": counts,
        **val_metrics,
    }
    _atomic_write_json(run_dir / "val_ensemble_metrics.json", result)
    print(f"[bundle-ensemble] n_bundles={val_metrics['n_bundles']}  "
          f"val 3-class {val_metrics['acc_3class_mean']:.4f} +/- "
          f"{val_metrics['acc_3class_std']:.4f}  "
          f"val binary {val_metrics['acc_binary_mean']:.4f} +/- "
          f"{val_metrics['acc_binary_std']:.4f}", flush=True)
    print(f"[bundle-ensemble] test predictions -> {pred_path}  counts {counts}",
          flush=True)
    return result


def run_from_dir(run_dir: Path, data_root: Path | None = None) -> dict:
    """Dispatcher: if per_seed_predictions.json exists for all components,
    use the per-seed bundle ensemble (the honest std methodology);
    otherwise fall back to legacy averaged-probability ensemble."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    components = spec.get("model", {}).get("components", [])
    have_per_seed = components and all(
        (Path(c) / "per_seed_predictions.json").exists() for c in components)
    if have_per_seed:
        return run_bundle_ensemble(run_dir)
    return run_ensemble(run_dir, data_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--val", action="store_true",
                        help="score the ensemble on the val split (legacy: "
                             "averages component val_predictions.csv files)")
    parser.add_argument("--bundle", action="store_true",
                        help="force per-seed bundle ensemble (val + test in "
                             "one pass; requires per_seed_predictions.json "
                             "per component)")
    parser.add_argument("--legacy", action="store_true",
                        help="force legacy averaged-probability ensemble even "
                             "if per-seed predictions are available")
    args = parser.parse_args()
    if args.bundle:
        run_bundle_ensemble(args.run_dir)
    elif args.val:
        run_val_ensemble(args.run_dir)
    elif args.legacy:
        run_ensemble(args.run_dir, args.data_root)
    else:
        run_from_dir(args.run_dir, args.data_root)
