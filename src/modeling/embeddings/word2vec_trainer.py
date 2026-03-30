# src/modeling/embeddings/word2vec_trainer.py
# v2 — Embedding sub-package adapter.
#
# Provides:
#   Word2VecTrainer — thin adapter over LogPreprocessor that adds a
#                     corpus-building helper and a cleaner training API.
#   build_corpus_from_messages — utility to tokenise a list of raw log strings.
#
# The actual embedding logic lives in src/preprocessing/log_preprocessor.py.
# This module re-wraps it under the v2 module path so training scripts and
# the v2 pipeline import from a stable, plan-aligned location.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from ...preprocessing.log_preprocessor import LogPreprocessor

logger = logging.getLogger(__name__)


def build_corpus_from_messages(messages: List[str]) -> List[List[str]]:
    """
    Tokenise a list of raw log message strings into a gensim-compatible corpus.

    Parameters
    ----------
    messages:
        Raw log strings (message field only — not full log lines with timestamps).

    Returns
    -------
    List[List[str]]
        Each inner list is the token sequence for one message.
    """
    prep = LogPreprocessor()
    corpus = []
    for msg in messages:
        cleaned = prep.clean(msg)
        tokens = prep.tokenize(cleaned)
        if tokens:
            corpus.append(tokens)
    logger.info(
        "build_corpus_from_messages: %d messages → %d non-empty token sequences",
        len(messages),
        len(corpus),
    )
    return corpus


class Word2VecTrainer:
    """
    Adapter around :class:`LogPreprocessor` with a corpus-building helper.

    Exposes the same ``train_embeddings``, ``save``, ``load``, and
    ``process_log`` interface as ``LogPreprocessor`` but adds
    :meth:`build_corpus` for convenience and is importable from the
    v2 module path ``src.modeling.embeddings.word2vec_trainer``.

    Parameters
    ----------
    vec_dim : int
        Embedding dimensionality (default 100).
    epochs : int
        Training epochs (default 10).
    window : int
        Context window size (default 5).
    min_count : int
        Minimum token frequency to include in vocabulary (default 2).
    workers : int
        Parallel workers for gensim (default 4).
    """

    def __init__(
        self,
        vec_dim: int = 100,
        epochs: int = 10,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
    ) -> None:
        self._preprocessor = LogPreprocessor(
            vec_dim=vec_dim,
            embedding_type="word2vec",
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            window=window,
        )

    # ------------------------------------------------------------------
    # Convenience delegation
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._preprocessor.is_trained

    @property
    def vec_dim(self) -> int:
        return self._preprocessor.vec_dim

    def build_corpus(self, messages: List[str]) -> List[List[str]]:
        """Tokenise *messages* into a gensim training corpus."""
        return build_corpus_from_messages(messages)

    def train(self, corpus: List[List[str]]) -> None:
        """Train the Word2Vec model on *corpus*."""
        self._preprocessor.train_embeddings(corpus)

    def save(self, path: Path) -> None:
        """Save the trained model to *path*."""
        self._preprocessor.save(path)

    def load(self, path: Path) -> None:
        """Load a previously saved model from *path*."""
        self._preprocessor.load(path)

    def process_log(self, raw_log: str):
        """
        End-to-end: clean → tokenise → embed one raw log string.

        Returns
        -------
        numpy.ndarray  [vec_dim], dtype float32
        """
        return self._preprocessor.process_log(raw_log)

    def get_preprocessor(self) -> LogPreprocessor:
        """Return the underlying :class:`LogPreprocessor` instance."""
        return self._preprocessor

    @property
    def word_vectors(self):
        """
        Return the trained gensim KeyedVectors object.

        Used by training scripts that embed integer token_ids directly via
        ``wv[str(token_id)]`` lookups, bypassing the text clean/tokenize path.

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded yet.
        """
        if self._preprocessor._model is None:
            raise RuntimeError(
                "No model is loaded. Call train() or load() before accessing word_vectors."
            )
        return self._preprocessor._model.wv
