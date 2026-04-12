"""Test-set inference + submission CSV generation.

USAGE NOTE: running this script produces a file intended for submission. The
submission budget is capped at 5 per team (ANTIPATTERNS.md rule 2). Before
running this with real test data, update scripts/submission_tracker.md and
get explicit in-session approval (Runtime HIP RT-H).

Usage (after test data arrives, HIP-7):
    python scripts/predict_test.py \\
        --model-path results/dl_nn3_novel/model.pth \\
        --config all_four \\
        --output predictions_nn3.csv

Stub during scaffolding. Implementation lands in Phase 6.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--config", required=True, choices=["bvp_eda", "bvp_eda_resp", "all_four"])
    p.add_argument("--output", required=True, help="Submission CSV path")
    p.add_argument("--model-type", choices=["classical", "cnn1d", "nn2", "nn3"], required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(
        f"[predict_test] Stub. model_path={args.model_path} type={args.model_type} "
        f"config={args.config} output={args.output}. RT-H approval required before real run."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
