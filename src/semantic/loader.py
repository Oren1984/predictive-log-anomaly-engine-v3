# src/semantic/loader.py
#
# SemanticModelLoader — lazy loader for the sentence-transformers model.
#
# The loader is a thin wrapper that:
#   1. Does nothing (returns None) when semantic_enabled=False.
#   2. Imports sentence-transformers only at load time, not at module import
#      time, so the module is safe to import in CI environments where
#      sentence-transformers is not installed.
#   3. Is idempotent: calling load() twice returns the cached instance.

from __future__ import annotations

import logging
from typing import Any, Optional

from .config import SemanticConfig

logger = logging.getLogger(__name__)


class SemanticModelLoader:
    """
    Lazy loader for the sentence-transformers embedding model.

    When ``config.semantic_enabled`` is ``False``, :meth:`load` is a no-op
    and :attr:`model` remains ``None``.  The rest of the pipeline can check
    ``loader.is_ready`` before attempting to use the model.

    Parameters
    ----------
    config:
        A :class:`SemanticConfig` instance.  If omitted, a default
        (disabled) config is used.
    """

    def __init__(self, config: Optional[SemanticConfig] = None) -> None:
        self._config = config or SemanticConfig()
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """``True`` when the model has been loaded and is usable."""
        return self._model is not None

    @property
    def model(self) -> Optional[Any]:
        """The loaded ``SentenceTransformer`` instance, or ``None``."""
        return self._model

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the sentence-transformers model if ``semantic_enabled=True``.

        If ``semantic_enabled=False`` this is a no-op.  If the model is
        already loaded this is also a no-op (idempotent).

        Raises
        ------
        ImportError
            If ``semantic_enabled=True`` but ``sentence-transformers`` is
            not installed.
        RuntimeError
            If the model identifier is invalid or the download fails.
        """
        if not self._config.semantic_enabled:
            logger.debug(
                "SemanticModelLoader.load: skipped (SEMANTIC_ENABLED=false)"
            )
            return

        if self._model is not None:
            logger.debug(
                "SemanticModelLoader.load: already loaded (%s)",
                self._config.semantic_model,
            )
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required when SEMANTIC_ENABLED=true. "
                "Install it with: pip install 'sentence-transformers>=2.7.0'"
            ) from exc

        logger.info(
            "SemanticModelLoader: loading model '%s'",
            self._config.semantic_model,
        )
        self._model = SentenceTransformer(self._config.semantic_model)
        logger.info(
            "SemanticModelLoader: model '%s' ready",
            self._config.semantic_model,
        )

    def unload(self) -> None:
        """Release the model reference (allows GC to reclaim memory)."""
        if self._model is not None:
            logger.info(
                "SemanticModelLoader: unloading model '%s'",
                self._config.semantic_model,
            )
        self._model = None
