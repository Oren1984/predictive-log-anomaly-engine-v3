# src/semantic/embeddings.py
#
# SemanticEmbedder — compute and cache sentence embeddings for log lines.
#
# When semantic_enabled=False the embedder is a no-op: embed() returns None
# and embed_batch() returns an empty list.  This allows callers to check
# ``if result is not None`` rather than branching on config state.

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np

from .config import SemanticConfig
from .loader import SemanticModelLoader

logger = logging.getLogger(__name__)


class SemanticEmbedder:
    """
    Compute semantic embeddings for log-line text using the loaded model.

    Embeddings are cached in an LRU cache keyed on the raw text string.
    Cache size is controlled by ``config.semantic_cache_size``.

    When ``config.semantic_enabled`` is ``False``:

    * :meth:`embed` returns ``None``.
    * :meth:`embed_batch` returns an empty list.
    * The cache is not populated.

    Parameters
    ----------
    config:
        A :class:`SemanticConfig` instance.  If omitted, a default
        (disabled) config is used.
    loader:
        A :class:`SemanticModelLoader` instance.  If omitted, one is
        created from ``config``.
    """

    def __init__(
        self,
        config: Optional[SemanticConfig] = None,
        loader: Optional[SemanticModelLoader] = None,
    ) -> None:
        self._config = config or SemanticConfig()
        self._loader = loader or SemanticModelLoader(self._config)
        self._cache_hits = 0
        self._cache_misses = 0

        # Build the cached _embed_one method at construction time so the
        # maxsize is determined by config rather than a module-level constant.
        cache_size = max(1, self._config.semantic_cache_size)
        self._embed_cached = lru_cache(maxsize=cache_size)(self._embed_raw)

    # ------------------------------------------------------------------
    # Internal: raw embedding (uncached)
    # ------------------------------------------------------------------

    def _embed_raw(self, text: str) -> Optional[np.ndarray]:
        """
        Compute a single embedding without cache.  Returns ``None`` when
        the model is not loaded.
        """
        if not self._loader.is_ready:
            return None
        vec = self._loader.model.encode(text, convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Return the embedding vector for ``text``.

        Returns ``None`` when ``semantic_enabled=False`` or the model
        has not been loaded yet.

        Parameters
        ----------
        text:
            Raw log line or template string to embed.

        Returns
        -------
        np.ndarray of shape ``[embedding_dim]``, dtype ``float32``,
        or ``None``.
        """
        if not self._config.semantic_enabled:
            return None

        result = self._embed_cached(text)
        return result

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Return embedding vectors for a list of strings.

        Returns an empty list when ``semantic_enabled=False``.

        Parameters
        ----------
        texts:
            List of raw log lines or template strings.

        Returns
        -------
        list of ``np.ndarray`` (or ``None`` per element if model not ready).
        """
        if not self._config.semantic_enabled:
            return []
        return [self.embed(t) for t in texts]

    # ------------------------------------------------------------------
    # Cache introspection
    # ------------------------------------------------------------------

    def cache_info(self) -> str:
        """Return a human-readable string with LRU cache statistics."""
        info = self._embed_cached.cache_info()
        return (
            f"SemanticEmbedder cache: hits={info.hits} misses={info.misses} "
            f"maxsize={info.maxsize} currsize={info.currsize}"
        )

    def cache_clear(self) -> None:
        """Clear the embedding LRU cache."""
        self._embed_cached.cache_clear()
        logger.debug("SemanticEmbedder: cache cleared")
