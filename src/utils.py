"""Utility functions: logging, atomic writes, serialization, timing.

Direct port from `Blood-Pressure-Inference-with-BVP/src/utils.py` (per
ANTIPATTERNS.md rule 12: code is COPIED with attribution, not imported across
project boundaries). The logger name is changed from "bp_pipeline" to
"ai4pain_2026" but the rest is identical.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


DEFAULT_SEED = 42


def setup_logging(log_dir: str = "logs", name: str = "ai4pain_2026") -> logging.Logger:
    """Set up file + console logging.

    Returns a logger that writes DEBUG to a timestamped file in ``log_dir`` and
    INFO to stderr. Idempotent: re-importing this module won't double up
    handlers.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to {log_file}")
    return logger


def get_logger(name: str = "ai4pain_2026") -> logging.Logger:
    """Return the project logger, configuring it if it has no handlers yet."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(handler)
    return logger


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types into JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj


def atomic_json_write(path: Path, data: Dict) -> None:
    """Write JSON atomically: write to .tmp then rename.

    Used by every checkpoint and progress file. Prevents corruption if a SLURM
    job is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(convert_to_serializable(data), f, indent=2)
    os.replace(str(tmp_path), str(path))


def atomic_write_text(path: Path, content: str) -> None:
    """Atomic text write helper. Used by simple text checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file if it exists, else return None."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def clear_memory() -> None:
    """Force garbage collection. Useful between large model fits."""
    gc.collect()


def set_all_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for Python, numpy, and (if available) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    """Context manager that times the body and logs the elapsed seconds."""
    start = time.time()
    yield
    elapsed = time.time() - start
    msg = f"{label}: {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)
