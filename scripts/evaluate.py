"""Standalone evaluation of saved models on the validation split.

Usage:
    python scripts/evaluate.py --model-path results/classical_ml/rf.pkl --config bvp_eda
    python scripts/evaluate.py --model-path results/dl_cnn_baseline/model.pth --config bvp_eda

Writes the full classification metric suite (balanced accuracy, macro-F1,
per-class precision/recall, confusion matrix, AUC-OVR) to the model's results
directory. Does not touch the test set.

Stub during scaffolding.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--config", required=True, choices=["bvp_eda", "bvp_eda_resp", "all_four"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"[evaluate] Stub. model={args.model_path} config={args.config}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
