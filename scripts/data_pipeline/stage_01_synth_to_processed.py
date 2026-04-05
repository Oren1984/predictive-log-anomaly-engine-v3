# scripts/data_pipeline/stage_01_synth_to_processed.py

# Purpose: Copy events_synth.parquet to data/processed/, sorted by timestamp. 
# Ensures the synthetic parquet matches the same schema conventions used by the existing pipeline 
# (columns: timestamp, service, level, message, meta, label) and is sorted by timestamp. 
# Does NOT modify data/processed/events_tokenized.parquet or any other existing file.

# Input: Path to the synthetic parquet file (default: data/synth/events_synth.parquet)

# Output: Path to the processed parquet file (default: data/processed/events_synth.parquet)

# Used by: This script can be run independently to prepare synthetic data for testing the anomaly detection pipeline.
# It is not directly used by other scripts but provides a processed dataset
# that can be used in place of real data for stages 02 and beyond.

"""
Stage 01 -- Synthetic: copy events_synth.parquet to data/processed/, sorted by timestamp.

Ensures the synthetic parquet matches the same schema conventions used by
the existing pipeline (columns: timestamp, service, level, message, meta, label)
and is sorted by timestamp.

Does NOT modify data/processed/events_tokenized.parquet or any other existing file.

Usage:
    python scripts/data_pipeline/stage_01_synth_to_processed.py
    python scripts/data_pipeline/stage_01_synth_to_processed.py \\
        --in  data/synth/events_synth.parquet \\
        --out data/processed/events_synth.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "ai_workspace" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage_01_synth_to_processed.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"timestamp", "service", "level", "message", "meta", "label"}


def run(in_path: Path, out_path: Path) -> None:
    log.info("=== Stage 01 Synth to Processed ===")
    log.info("Input:  %s", in_path)
    log.info("Output: %s", out_path)

    t_start = time.perf_counter()

    if not in_path.exists():
        log.error("Input file not found: %s", in_path)
        log.error("Run stage_01_synth_generate.py first.")
        sys.exit(1)

    log.info("Loading %s ...", in_path)
    df = pd.read_parquet(in_path)
    log.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        log.error("Missing required columns: %s", sorted(missing))
        sys.exit(1)

    # Sort by timestamp (enforce strictly increasing order)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Sorted by timestamp. Range: [%.0f, %.0f]",
             df["timestamp"].min(), df["timestamp"].max())

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t_start
    log.info("Written %d rows to %s (%.2fs)", len(df), out_path, elapsed)
    log.info("=== Stage 01 Synth to Processed complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 01 Synth to Processed")
    parser.add_argument(
        "--in", dest="in_path", type=Path,
        default=ROOT / "data" / "synth" / "events_synth.parquet",
        help="Input parquet (canonical schema)",
    )
    parser.add_argument(
        "--out", dest="out_path", type=Path,
        default=ROOT / "data" / "processed" / "events_synth.parquet",
        help="Output parquet in data/processed/",
    )
    args = parser.parse_args()
    run(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
