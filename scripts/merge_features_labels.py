"""Join Rust feature extractor output with label metadata.

The Rust extractor produces data/features/results_{split}_{signal}.csv with
40 feature columns per (subject_id, trial_id). This script joins those CSVs
with the label metadata (NP / HP / AP per trial) into a single dataframe per
split, which the Python data loader then consumes.

Stub during scaffolding. Implementation lands in Phase 2.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--labels-path", required=True)
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"[merge_features_labels] Stub. split={args.split}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
