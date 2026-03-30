#!/usr/bin/env python
# training/train_embeddings.py
# Phase 2 — Data Pipeline: train Word2Vec embeddings on the processed log corpus.
#
# Architecture note (token-ID mode, extended corpus)
# ---------------------------------------------------
# events_tokenized.parquet stores integer token_ids from a DrainParser-style template
# miner — NOT raw log message text.  Word2Vec "words" are therefore str(token_id).
#
# The corpus is built from THREE sources to maximise vocabulary coverage:
#
#   Source 1 — sequences_train.parquet  'tokens' column
#       1600 real labelled session sequences (lengths 12–42).
#       Provides high-quality co-occurrence context for 14 frequently-seen token_ids.
#
#   Source 2 — events_tokenized.parquet  session groups (≥ 3 events)
#       ~130K multi-event sessions covering all 7,833 known template token_ids.
#       Provides real co-occurrence data across the full vocabulary.
#
#   Source 3 — templates.csv  template text tokenization
#       Templates are grouped by the first N non-placeholder words of their
#       template_text (e.g. "INFO dfs.DataNode$DataXceiver: Receiving block").
#       Each group becomes one Word2Vec sentence, giving semantically related
#       templates a shared embedding neighbourhood.
#       Guarantees every template token_id appears in the corpus.
#
# min_count=1 is used to guarantee full vocabulary coverage (all 7,833 templates
# appear in the corpus, including rare ones from Source 3 only).
#
# Input:
#   data/processed/sequences_train.parquet      (Source 1)
#   data/processed/events_tokenized.parquet     (Source 2)
#   data/intermediate/templates.csv             (Source 3)
#
# Output:
#   models/embeddings/word2vec.model            (gensim Word2Vec binary)
#
# Usage:
#   python -m training.train_embeddings
#   python -m training.train_embeddings --vec-dim 128 --epochs 10
#
# Environment variables (override CLI defaults):
#   VEC_DIM       embedding dimensionality   (default: 100)
#   W2V_EPOCHS    training epochs            (default: 10)
#   W2V_WINDOW    context window size        (default: 5)
#   W2V_MIN_COUNT minimum token frequency    (default: 1)
#   W2V_WORKERS   parallel workers           (default: 4)

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so the script can be run from any working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.modeling.embeddings.word2vec_trainer import Word2VecTrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training.train_embeddings")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_PARQUET = _PROJECT_ROOT / "data" / "processed" / "events_tokenized.parquet"
_SEQ_TRAIN = _PROJECT_ROOT / "data" / "processed" / "sequences_train.parquet"
_TEMPLATES_CSV = _PROJECT_ROOT / "data" / "intermediate" / "templates.csv"
_MODEL_OUT = _PROJECT_ROOT / "models" / "embeddings" / "word2vec.model"


# ---------------------------------------------------------------------------
# Corpus building
# ---------------------------------------------------------------------------

def _build_token_corpus() -> "list[list[str]]":
    """
    Build an extended Word2Vec corpus from three sources.

    Source 1 — sequences_train.parquet
        Real labelled session sequences (1600 rows, lengths 12–42).  Provides
        dense co-occurrence context for the 14 most frequent token_ids.

    Source 2 — events_tokenized.parquet (sessions with ≥ 3 events)
        ~130K multi-event sessions that together cover all 7,833 known token_ids.
        Provides real co-occurrence data across the full template vocabulary.

    Source 3 — templates.csv (template text tokenization)
        Templates are grouped by the first four non-placeholder words of their
        template_text (e.g. "INFO dfs.DataNode$DataXceiver: Receiving block").
        Each group becomes one Word2Vec sentence so that semantically related
        templates share an embedding neighbourhood.
        Guarantees every template token_id appears in the corpus at least once.

    Returns
    -------
    list[list[str]]
        Gensim-compatible corpus — each inner list is one sentence where every
        element is str(token_id).
    """
    import pandas as pd
    from collections import defaultdict

    corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Source 1: sequences_train.parquet
    # ------------------------------------------------------------------
    if _SEQ_TRAIN.exists():
        df_seq = pd.read_parquet(_SEQ_TRAIN)
        if "tokens" in df_seq.columns:
            n_before = len(corpus)
            for tids in df_seq["tokens"]:
                seq = [str(t) for t in tids]
                if seq:
                    corpus.append(seq)
            logger.info(
                "Source 1 (sequences_train.parquet): %d sequences",
                len(corpus) - n_before,
            )

    # ------------------------------------------------------------------
    # Source 2: events_tokenized.parquet — multi-event sessions
    # ------------------------------------------------------------------
    if _DATA_PARQUET.exists():
        n_before = len(corpus)
        df_tok = pd.read_parquet(_DATA_PARQUET, columns=["session_id", "token_id"])
        for _, group in df_tok.groupby("session_id", sort=False):
            if len(group) >= 3:
                corpus.append(group["token_id"].astype(str).tolist())
        logger.info(
            "Source 2 (events_tokenized sessions ≥3): %d sequences added",
            len(corpus) - n_before,
        )

    # ------------------------------------------------------------------
    # Source 3: templates.csv — template text tokenization
    # Group templates by the first four non-placeholder words of their
    # template_text. Each group becomes one sentence so semantically
    # related templates receive co-occurrence context.
    # ------------------------------------------------------------------
    if _TEMPLATES_CSV.exists():
        n_before = len(corpus)
        df_tmpl = pd.read_csv(_TEMPLATES_CSV, usecols=["template_id", "template_text"])
        groups: dict[str, list[str]] = defaultdict(list)
        for _, row in df_tmpl.iterrows():
            words = str(row["template_text"]).split()
            # Skip placeholder tokens (<BLK>, <NUM>, etc.) when building the key
            non_ph = [w for w in words if not (w.startswith("<") and w.endswith(">"))]
            key = " ".join(non_ph[:4]) if non_ph else "_other"
            groups[key].append(str(int(row["template_id"]) + 2))
        for tids in groups.values():
            corpus.append(tids)
        n_groups = len(corpus) - n_before
        n_tmpl = sum(len(v) for v in groups.values())
        logger.info(
            "Source 3 (template text tokenization): %d group sentences covering %d templates",
            n_groups, n_tmpl,
        )

    if not corpus:
        raise FileNotFoundError(
            "No corpus data found. Expected:\n"
            f"  {_SEQ_TRAIN}\n"
            f"  {_DATA_PARQUET}\n"
            f"  {_TEMPLATES_CSV}\n"
            "Run data preprocessing scripts first."
        )

    # Coverage report
    all_tids = {tok for sent in corpus for tok in sent}
    total_tokens = sum(len(s) for s in corpus)
    logger.info(
        "Corpus ready: %d sentences, %d total tokens, %d unique token_ids",
        len(corpus), total_tokens, len(all_tids),
    )
    return corpus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    vec_dim: int,
    epochs: int,
    window: int,
    min_count: int,
    workers: int,
) -> None:
    logger.info(
        "Starting Word2Vec training (token-ID mode, extended corpus): "
        "vec_dim=%d epochs=%d window=%d min_count=%d workers=%d",
        vec_dim, epochs, window, min_count, workers,
    )

    # Build extended corpus (three sources — see _build_token_corpus docstring).
    corpus = _build_token_corpus()
    if not corpus:
        raise ValueError("Token corpus is empty — cannot train embeddings.")

    trainer = Word2VecTrainer(
        vec_dim=vec_dim,
        epochs=epochs,
        window=window,
        min_count=min_count,
        workers=workers,
    )

    # corpus is already List[List[str]] — pass directly to train(), bypassing
    # build_corpus() which is designed for raw text tokenisation.
    logger.info("Training Word2Vec on %d token-ID sequences ...", len(corpus))
    trainer.train(corpus)

    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(_MODEL_OUT)
    logger.info("Word2Vec model saved to %s", _MODEL_OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Word2Vec embeddings for the v2 log anomaly pipeline."
    )
    parser.add_argument(
        "--vec-dim",
        type=int,
        default=int(os.environ.get("VEC_DIM", "100")),
        help="Embedding dimensionality (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("W2V_EPOCHS", "10")),
        help="Training epochs (default: 10)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=int(os.environ.get("W2V_WINDOW", "5")),
        help="Context window size (default: 5)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=int(os.environ.get("W2V_MIN_COUNT", "1")),
        help="Minimum token frequency (default: 1 — ensures full template coverage)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("W2V_WORKERS", "4")),
        help="Parallel workers for gensim (default: 4)",
    )
    args = parser.parse_args()
    main(
        vec_dim=args.vec_dim,
        epochs=args.epochs,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
    )
