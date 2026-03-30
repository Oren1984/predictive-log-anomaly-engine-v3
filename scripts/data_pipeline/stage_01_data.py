# scripts/stage_01_data.py

# Purpose: This script is responsible for the first stage of the anomaly detection pipeline,
# which involves loading, validating, and summarizing the raw unified dataset.
# It checks for the presence of the expected schema file, loads the raw data using a custom data loader,
# validates the required columns, and provides a summary of the dataset including memory usage and label distribution.

# Input: The script takes an optional command-line argument for the mode of operation (e.g., "demo" for a faster run using a subset of the data).

# Output: The script outputs logs that include information about the dataset, such as the number of rows and columns,
# memory usage, label distribution, and any issues encountered during validation.

# Used by: This script is used by the one-command pipeline script (scripts/run_0_4.py) to execute stage 01 of the anomaly detection pipeline.

"""
Stage 01 — Data: validate and summarise the raw unified dataset.

Usage:
    python scripts/stage_01_data.py
    python scripts/stage_01_data.py --mode demo  (first 5000 rows only)
    python scripts/stage_01_data.py --mode full
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_layer import KaggleDatasetLoader

LOG_DIR = ROOT / "ai_workspace" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_CSV = ROOT / "data" / "processed" / "events_unified.csv"
SCHEMA_MD   = ROOT / "data" / "processed" / "schema.md"


def _setup_logging(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logger = logging.getLogger("stage_01")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)
    # file
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    return logger


def _mem_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def main(mode: str) -> None:
    log_path = LOG_DIR / f"stage_01_data_{mode}.log"
    log = _setup_logging(log_path)
    t0 = time.time()

    log.info("=" * 60)
    log.info("Stage 01 | mode=%s | source=%s", mode, UNIFIED_CSV)
    log.info("Memory baseline: %.1f MB", _mem_mb())

    nrows = 5_000 if mode == "demo" else None

    # ---------------------------------------------------------------
    # 1. Check schema.md
    # ---------------------------------------------------------------
    if SCHEMA_MD.exists():
        log.info("schema.md  EXISTS  (%d B)", SCHEMA_MD.stat().st_size)
    else:
        log.warning("schema.md  MISSING — expected at %s", SCHEMA_MD)

    # ---------------------------------------------------------------
    # 2. load_raw()
    # ---------------------------------------------------------------
    loader = KaggleDatasetLoader(root=ROOT, nrows=nrows)
    df = loader.load_raw()
    log.info("load_raw(): %d rows x %d cols  (%.1f MB RSS)",
             len(df), df.shape[1], _mem_mb())
    log.info("Columns: %s", list(df.columns))
    log.info("Label distribution:\n%s",
             df.groupby(["dataset", "label"]).size().to_string())

    # Validate required columns
    required = {"dataset", "session_id", "message", "label"}
    missing = required - set(df.columns)
    if missing:
        log.error("Missing required columns: %s", missing)
        sys.exit(1)
    log.info("Required columns check: OK")

    # ---------------------------------------------------------------
    # 3. normalize_schema()
    # ---------------------------------------------------------------
    norm = loader.normalize_schema()
    log.info("normalize_schema(): %d rows x %d cols", len(norm), norm.shape[1])

    expected_norm = {"timestamp", "service", "level", "message", "session_id", "label"}
    missing_norm = expected_norm - set(norm.columns)
    if missing_norm:
        log.error("normalize_schema() missing columns: %s", missing_norm)
        sys.exit(1)

    # Spot-check mapping: 'dataset' column became 'service'
    unique_services = norm["service"].unique().tolist()
    log.info("normalize_schema() service values: %s", unique_services)
    assert "dataset" not in norm.columns, "Old 'dataset' column leaked into normalized schema"
    log.info("normalize_schema() structure check: OK")

    # ---------------------------------------------------------------
    # 4. Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    log.info("-" * 60)
    log.info("Stage 01 COMPLETE | elapsed=%.1fs | peak_mem=%.1f MB",
             elapsed, _mem_mb())
    log.info("Log written to: %s", log_path)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "demo"])
    args = parser.parse_args()
    main(args.mode)
