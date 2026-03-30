# scripts/stage_03_sequences.py

# Purpose: Build train/val/test sequence splits and write parquet files for model training and evaluation.

# Input: Reads session sequences from data/intermediate/session_sequences_v2.csv, which contains session_id, tokens, and label columns.

# Output: Writes data/processed/sequences_train.parquet, data/processed/sequences_val.parquet, 
# and data/processed/sequences_test.parquet with columns: sequence_id, tokens (JSON string), label.

# Used by: This script is used by the main pipeline to prepare sequence splits for model training and evaluation. 
# The generated files are consumed in later stages for model input preparation and anomaly detection. 
# It can be run independently to regenerate sequence splits if the session sequences are updated or if you want to switch between demo and full modes.

"""
Stage 03 — Sequences: build train/val/test sequence splits and write parquet.

Reads  : data/intermediate/session_sequences_v2.csv
Writes : data/processed/sequences_train.parquet
         data/processed/sequences_val.parquet
         data/processed/sequences_test.parquet

Usage:
    python scripts/stage_03_sequences.py
    python scripts/stage_03_sequences.py --mode demo  (first 2000 sessions)
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parsing import EventTokenizer
from src.sequencing import SessionSequenceBuilder, DatasetSplitter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SESSION_SEQ_CSV = ROOT / "data" / "intermediate" / "session_sequences_v2.csv"
TEMPLATES_CSV   = ROOT / "data" / "intermediate" / "templates.csv"
OUT_DIR         = ROOT / "data" / "processed"


def sequences_to_df(seqs) -> pd.DataFrame:
    rows = []
    for s in seqs:
        rows.append({
            "sequence_id": s.sequence_id,
            "tokens":      json.dumps(s.tokens),
            "label":       s.label,
        })
    return pd.DataFrame(rows)


def main(mode: str) -> None:
    nrows = 2_000 if mode == "demo" else None
    log.info("Stage 03 | mode=%s | source=%s", mode, SESSION_SEQ_CSV)

    tok = EventTokenizer().load_from_csv(TEMPLATES_CSV)
    log.info("Tokenizer vocab_size=%d", tok.vocab_size)

    builder = SessionSequenceBuilder(tokenizer=tok, nrows=nrows)
    sequences = builder.load_csv(SESSION_SEQ_CSV)
    log.info("Built %d sequences", len(sequences))

    splitter = DatasetSplitter(val_ratio=0.10, test_ratio=0.10, seed=42)
    train, val, test = splitter.split_stratified(sequences)
    log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"sequences_{name}.parquet"
        sequences_to_df(split).to_parquet(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(split))

    log.info("Stage 03 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "demo"])
    args = parser.parse_args()
    main(args.mode)
