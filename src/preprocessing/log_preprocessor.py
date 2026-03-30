# src/preprocessing/log_preprocessor.py
# Stage 1: NLP Embedding
#
# LogPreprocessor converts raw log message strings into fixed-size float
# vectors using Word2Vec (default) or FastText (experimental) embeddings.
#
# Design notes:
#   - Normalisation patterns are adapted from src/parsing/template_miner.py
#     and extended with improved IP, timestamp and service-name handling.
#   - gensim is imported lazily so the module is importable in environments
#     where gensim is not installed (e.g. CI runs that test only the API layer).
#   - FastText is guarded by embedding_type="fasttext" and emits a warning;
#     it must never become the production default.
#   - This class is completely isolated from the existing runtime pipeline.
#     It is not imported by any existing module and changes no behaviour.

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional gensim import — only required at training / embedding time
# ---------------------------------------------------------------------------
try:
    from gensim.models import FastText as _FastTextModel
    from gensim.models import Word2Vec as _Word2VecModel
    _GENSIM_AVAILABLE = True
except ImportError:                          # pragma: no cover
    _GENSIM_AVAILABLE = False
    _Word2VecModel = None                    # type: ignore[assignment,misc]
    _FastTextModel = None                    # type: ignore[assignment,misc]

EmbeddingType = Literal["word2vec", "fasttext"]

# ---------------------------------------------------------------------------
# Normalisation patterns
# ---------------------------------------------------------------------------
# Applied in order via re.sub.  Patterns are compiled once at import time.
# Placeholder tokens use [UPPER] convention so the tokeniser can recognise
# them as atomic units rather than splitting on the brackets.
#
# Sources / rationale:
#   [BLK]       — HDFS block IDs (blk_-1234567890)
#   [TIMESTAMP] — ISO dates, BGL dotted timestamps, date-only strings
#   [IP]        — IPv4 addresses with optional :port
#   [NODE]      — BGL rack/node identifiers (R3-M1-N1:J18-U11)
#   [PATH]      — Unix-style file paths
#   [HEX]       — Hex strings of 8+ chars (addresses, hashes, error codes)
#   [NUM]       — Remaining bare integers (counts, IDs, ports)
# ---------------------------------------------------------------------------
_NORM_PATTERNS: List[tuple] = [
    # BGL block IDs  e.g. blk_-1616688950519514760
    (re.compile(r"blk_-?\d+"),                                         "[BLK]"),
    # BGL dotted timestamp e.g. 2005-12-01-06.51.06.123456
    (re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+"),      "[TIMESTAMP]"),
    # ISO datetime  e.g. 2005-12-01T06:51:06 or 2005-12-01 06:51:06
    (re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}"),    "[TIMESTAMP]"),
    # Date only  e.g. 2005-12-01 or 2005/12/01 or 2005.12.01
    (re.compile(r"\d{4}[-./]\d{2}[-./]\d{2}"),                         "[TIMESTAMP]"),
    # IPv4 address with optional port  e.g. 10.0.0.1:8080
    (re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?"),      "[IP]"),
    # BGL node names  e.g. R3-M1-N1:J18-U11 (case-insensitive: applied after lowercase)
    (re.compile(r"R\d+(?:-[A-Z0-9]+)+(?::[A-Z]\d+-[A-Z]\d+)?", re.IGNORECASE), "[NODE]"),
    # Unix file paths  e.g. /user/hadoop/tmp/blk_12345
    (re.compile(r"/[a-zA-Z0-9_./-]+"),                                  "[PATH]"),
    # Hex strings 8+ chars  e.g. 0x1a2b3c4d or 1a2b3c4d
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"),                               "[HEX]"),
    # Bare integers  e.g. 3, 404, 1048576
    (re.compile(r"\b\d+\b"),                                            "[NUM]"),
    # Collapse whitespace
    (re.compile(r"\s+"),                                                " "),
]

# Tokeniser pattern: matches placeholder tokens OR word characters.
# This ensures [IP], [TIMESTAMP], etc. are treated as single tokens.
_TOKEN_RE = re.compile(r"\[[A-Z]+\]|\w+")

# Known service name prefixes (BGL / HDFS).  When present at the start of a
# cleaned line they are replaced with a generic [SERVICE] token so the model
# does not overfit to dataset-specific names.
_SERVICE_PREFIXES = re.compile(
    r"^(hdfs|bgl|dfs|hadoop|namenode|datanode|jobtracker|tasktracker)"
    r"[\s.:,]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# LogPreprocessor
# ---------------------------------------------------------------------------

class LogPreprocessor:
    """
    Stage 1: NLP Embedding.

    Converts raw log message strings into fixed-size float vectors using
    Word2Vec (default) or FastText (experimental) semantic embeddings.

    Typical usage
    -------------
    Offline training::

        preprocessor = LogPreprocessor(vec_dim=100)
        corpus = [preprocessor.tokenize(preprocessor.clean(msg))
                  for msg in raw_messages]
        preprocessor.train_embeddings(corpus)
        preprocessor.save(Path("models/word2vec.model"))

    Inference::

        preprocessor = LogPreprocessor()
        preprocessor.load(Path("models/word2vec.model"))
        vector = preprocessor.process_log("ERROR 10.0.0.1 disk full 2048")

    Parameters
    ----------
    vec_dim:
        Embedding dimensionality.  Defaults to 100.
    embedding_type:
        ``"word2vec"`` (default, production) or ``"fasttext"`` (experimental).
    min_count:
        Minimum token frequency for the embedding vocabulary.
    workers:
        Parallel workers for gensim training.
    epochs:
        Training epochs.
    window:
        Context window size for Word2Vec / FastText.
    """

    def __init__(
        self,
        vec_dim: int = 100,
        embedding_type: EmbeddingType = "word2vec",
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        window: int = 5,
    ) -> None:
        if embedding_type not in ("word2vec", "fasttext"):
            raise ValueError(
                f"embedding_type must be 'word2vec' or 'fasttext', got {embedding_type!r}"
            )
        if embedding_type == "fasttext":
            logger.warning(
                "LogPreprocessor: FastText is experimental. "
                "Word2Vec is the production default. "
                "Do not promote FastText without comparative evaluation."
            )

        self.vec_dim = vec_dim
        self.embedding_type: EmbeddingType = embedding_type
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.window = window
        self._model: Optional[object] = None   # gensim model, set after train/load

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True when an embedding model is loaded and ready."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def clean(self, raw_text: str) -> str:
        """
        Normalise a raw log message string.

        Steps applied in order:
        1. Lowercase
        2. Replace service-name prefixes with ``[SERVICE]``
        3. Apply all ``_NORM_PATTERNS`` (BLK, TIMESTAMP, IP, NODE, PATH, HEX, NUM)
        4. Strip leading / trailing whitespace

        Parameters
        ----------
        raw_text:
            The raw log message string (not a full log line — just the
            ``message`` field after structural parsing).

        Returns
        -------
        str
            Normalised, lowercase string with placeholders substituted.
        """
        text = raw_text.lower().strip()
        # Replace known service-name prefixes
        text = _SERVICE_PREFIXES.sub("[SERVICE] ", text)
        # Apply all normalisation substitutions
        for pattern, replacement in _NORM_PATTERNS:
            text = pattern.sub(replacement, text)
        return text.strip()

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """
        Split a normalised log string into tokens.

        Uses a regex that recognises placeholder tokens (``[IP]``,
        ``[TIMESTAMP]``, etc.) as atomic units, then matches any
        remaining word characters (``\\w+``).  Punctuation-only
        fragments are discarded.

        Parameters
        ----------
        text:
            A string that has already been passed through :meth:`clean`.

        Returns
        -------
        List[str]
            List of lowercase token strings.  May be empty for blank input.
        """
        return _TOKEN_RE.findall(text)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "No embedding model is loaded. "
                "Call train_embeddings() or load() before embedding."
            )

    def _mean_pool(self, tokens: List[str]) -> np.ndarray:
        """
        Mean-pool the word vectors for *tokens* that exist in the vocabulary.

        Tokens not found in the vocabulary are silently skipped.  If no
        token matches the vocabulary, a zero vector is returned.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``[vec_dim]``.
        """
        wv = self._model.wv  # type: ignore[union-attr]
        vectors = [wv[tok] for tok in tokens if tok in wv]
        if not vectors:
            return np.zeros(self.vec_dim, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def embed(self, tokens: List[str]) -> np.ndarray:
        """
        Convert a token list into a fixed-size float vector via mean pooling.

        Parameters
        ----------
        tokens:
            Token list produced by :meth:`tokenize`.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``[vec_dim]``.

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded yet.
        """
        self._require_model()
        return self._mean_pool(tokens)

    def process_log(self, log_line: str) -> np.ndarray:
        """
        End-to-end pipeline: normalise → tokenise → embed.

        This is the primary public interface for inference-time use.

        Parameters
        ----------
        log_line:
            Raw log message string.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``[vec_dim]``.

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded yet.
        """
        self._require_model()
        cleaned = self.clean(log_line)
        tokens = self.tokenize(cleaned)
        return self._mean_pool(tokens)

    def transform(self, raw_text: str) -> np.ndarray:
        """Alias for :meth:`process_log` — kept for interface consistency."""
        return self.process_log(raw_text)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_embeddings(self, corpus: List[List[str]]) -> None:
        """
        Train a Word2Vec (or FastText) model on a tokenised log corpus.

        Parameters
        ----------
        corpus:
            List of token lists.  Each inner list is the tokenised form of
            one log message, produced by ``tokenize(clean(msg))``.

        Raises
        ------
        ImportError
            If gensim is not installed.
        ValueError
            If ``corpus`` is empty.
        """
        if not _GENSIM_AVAILABLE:
            raise ImportError(
                "gensim is required for embedding training. "
                "Install it with: pip install 'gensim>=4.3.0'"
            )
        if not corpus:
            raise ValueError("corpus must not be empty")

        common_kwargs = dict(
            sentences=corpus,
            vector_size=self.vec_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
        )

        if self.embedding_type == "word2vec":
            logger.info(
                "Training Word2Vec: %d sentences, vec_dim=%d, epochs=%d",
                len(corpus), self.vec_dim, self.epochs,
            )
            self._model = _Word2VecModel(**common_kwargs)

        else:  # fasttext — experimental
            logger.warning(
                "Training FastText (experimental): %d sentences, vec_dim=%d",
                len(corpus), self.vec_dim,
            )
            self._model = _FastTextModel(**common_kwargs)

        logger.info(
            "Embedding training complete. Vocabulary size: %d",
            len(self._model.wv),  # type: ignore[union-attr]
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save the trained embedding model to disk.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``models/word2vec.model``).
            Parent directories are created automatically.

        Raises
        ------
        RuntimeError
            If no model has been trained or loaded yet.
        """
        self._require_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))          # type: ignore[union-attr]
        logger.info("Saved %s model to %s", self.embedding_type, path)

    def load(self, path: Path) -> None:
        """
        Load a previously saved embedding model from disk.

        Parameters
        ----------
        path:
            Path to a gensim model file saved by :meth:`save`.

        Raises
        ------
        ImportError
            If gensim is not installed.
        FileNotFoundError
            If the model file does not exist.
        """
        if not _GENSIM_AVAILABLE:
            raise ImportError(
                "gensim is required. Install with: pip install 'gensim>=4.3.0'"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if self.embedding_type == "word2vec":
            self._model = _Word2VecModel.load(str(path))
        else:  # fasttext
            self._model = _FastTextModel.load(str(path))

        # Sync vec_dim in case the loaded model was trained with a different size
        self.vec_dim = self._model.vector_size  # type: ignore[union-attr]
        logger.info(
            "Loaded %s model from %s (vec_dim=%d)",
            self.embedding_type, path, self.vec_dim,
        )
