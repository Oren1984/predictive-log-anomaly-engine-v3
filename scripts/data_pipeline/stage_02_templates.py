# scripts/stage_02_templates.py

# Purpose: Load existing mining artifacts and write artifacts/templates.json, artifacts/vocab.json, and data/processed/events_tokenized.parquet.

# Input: Existing mining artifacts in data/intermediate/templates.csv and data/intermediate/events_with_templates.csv

# Output: artifacts/templates.json, artifacts/vocab.json, and data/processed/events_tokenized.parquet with columns: 
# timestamp, service, session_id, template_id, token_id, label

# Used by: This script is used by the main pipeline to prepare template and tokenization artifacts for model training and inference. 
# The generated files are consumed in later stages for sequence building, 
# model input preparation, and ultimately anomaly detection. 
# It can be run independently to regenerate templates and tokenized data 
# if the mining step is updated or if you want to switch between demo and full modes.

"""
Stage 02 — Templates: load existing mining artifacts and write
artifacts/templates.json, artifacts/vocab.json, and
data/processed/events_tokenized.parquet.

Parquet output columns: timestamp, service, session_id,
                         template_id, token_id, label

Usage:
    python scripts/stage_02_templates.py
    python scripts/stage_02_templates.py --mode demo
    python scripts/stage_02_templates.py --mode full
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parsing import TemplateMiner, EventTokenizer

LOG_DIR = ROOT / "ai_workspace" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_CSV     = ROOT / "data" / "intermediate" / "templates.csv"
EVENTS_TEMPLATES  = ROOT / "data" / "intermediate" / "events_with_templates.csv"
TEMPLATES_JSON    = ROOT / "artifacts" / "templates.json"
VOCAB_JSON        = ROOT / "artifacts" / "vocab.json"
TOKENIZED_PARQUET = ROOT / "data" / "processed" / "events_tokenized.parquet"


def _setup_logging(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logger = logging.getLogger("stage_02")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    return logger


def _mem_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def main(mode: str) -> None:
    log_path = LOG_DIR / f"stage_02_templates_{mode}.log"
    log = _setup_logging(log_path)
    t0 = time.time()

    log.info("=" * 60)
    log.info("Stage 02 | mode=%s", mode)
    log.info("Memory baseline: %.1f MB", _mem_mb())

    # ---------------------------------------------------------------
    # 1. Load template miner
    # ---------------------------------------------------------------
    miner = TemplateMiner().load_from_csv(TEMPLATES_CSV)
    log.info("Loaded %d templates", miner.vocab_size)

    # ---------------------------------------------------------------
    # 2. Load tokenizer
    # ---------------------------------------------------------------
    tok = EventTokenizer().load_from_csv(TEMPLATES_CSV)
    log.info("Vocab size (incl. PAD+UNK): %d", tok.vocab_size)

    # ---------------------------------------------------------------
    # 3. Write artifacts/templates.json
    # ---------------------------------------------------------------
    templates_payload = {
        str(tid): text
        for tid, text in miner._id_to_template.items()
    }
    TEMPLATES_JSON.write_text(json.dumps(templates_payload, indent=2))
    sz = TEMPLATES_JSON.stat().st_size
    log.info("Wrote templates.json  entries=%d  size=%d B",
             len(templates_payload), sz)

    # ---------------------------------------------------------------
    # 4. Write artifacts/vocab.json
    # ---------------------------------------------------------------
    vocab_dict = tok.to_vocab_dict()
    VOCAB_JSON.write_text(json.dumps(vocab_dict, indent=2))
    sz_v = VOCAB_JSON.stat().st_size
    log.info("Wrote vocab.json  entries=%d  size=%d B",
             len(vocab_dict), sz_v)
    log.info("PAD token: vocab['0']=%s", vocab_dict.get("0"))
    log.info("UNK token: vocab['1']=%s", vocab_dict.get("1"))

    if mode == "demo":
        log.info("Demo mode: skipping tokenized parquet export")
        elapsed = time.time() - t0
        log.info("Stage 02 COMPLETE (demo) | elapsed=%.1fs | mem=%.1f MB",
                 elapsed, _mem_mb())
        log.info("Log written to: %s", log_path)
        return

    # ---------------------------------------------------------------
    # 5. Write data/processed/events_tokenized.parquet (full mode)
    # ---------------------------------------------------------------
    log.info("Reading events_with_templates.csv ...")
    df = pd.read_csv(
        EVENTS_TEMPLATES,
        usecols=["timestamp", "dataset", "session_id",
                 "label", "template_id"],
        dtype={"label": "int8", "template_id": "int32"},
    )
    log.info("Read CSV: %d rows x %d cols  (mem=%.1f MB)",
             len(df), df.shape[1], _mem_mb())

    # Rename 'dataset' -> 'service' to match normalised schema
    df.rename(columns={"dataset": "service"}, inplace=True)

    # Encode template_ids to token_ids
    df["token_id"] = tok.encode(df["template_id"].tolist())
    df["token_id"] = df["token_id"].astype("int32")
    log.info("Encoded %d token_ids  (mem=%.1f MB)", len(df), _mem_mb())

    # Final column order
    df = df[["timestamp", "service", "session_id",
             "template_id", "token_id", "label"]]

    # Write parquet
    df.to_parquet(TOKENIZED_PARQUET, index=False)
    pq_size = TOKENIZED_PARQUET.stat().st_size
    log.info("Wrote events_tokenized.parquet")
    log.info("  rows=%d  cols=%s  size=%d B (%.1f MB)",
             len(df), list(df.columns), pq_size, pq_size / 1024 / 1024)

    # Validate written file
    check = pd.read_parquet(TOKENIZED_PARQUET, columns=["token_id"])
    assert len(check) == len(df), "Row count mismatch after write!"
    assert check["token_id"].notna().all(), "NaN token_ids detected!"
    log.info("Parquet validation: row count OK, no NaN token_ids")

    # ---------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    log.info("-" * 60)
    log.info("Stage 02 COMPLETE | elapsed=%.1fs | peak_mem=%.1f MB",
             elapsed, _mem_mb())
    log.info("Log written to: %s", log_path)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "demo"])
    args = parser.parse_args()
    main(args.mode)
