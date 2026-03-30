# src/semantic/config.py
#
# SemanticConfig — environment-driven configuration for the V3 semantic layer.
#
# All fields default to the safe/disabled state so the semantic layer is
# completely inert unless explicitly opted in via environment variables.
# This file has no imports from sentence-transformers or any heavy library,
# so it is always importable regardless of whether those dependencies are
# installed.

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean from an environment variable."""
    val = os.environ.get(name, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


@dataclass
class SemanticConfig:
    """
    Configuration for the V3 semantic / explanation layer.

    All fields read from environment variables with safe defaults.
    When ``semantic_enabled`` is ``False`` (the default), no model is
    loaded and no embedding computation takes place.

    Parameters
    ----------
    semantic_enabled:
        Master switch for the semantic layer.  Default ``False``.
        Set ``SEMANTIC_ENABLED=true`` to activate.
    semantic_model:
        Sentence-Transformers model identifier used for log-line embeddings.
        Only loaded when ``semantic_enabled=True``.
        Default ``"all-MiniLM-L6-v2"``.
    explanation_enabled:
        Whether the explanation sub-system is active.  Requires
        ``semantic_enabled=True`` to have any effect.  Default ``False``.
        Set ``EXPLANATION_ENABLED=true`` to activate.
    explanation_model:
        Explanation strategy identifier.  ``"rule-based"`` uses heuristic
        template matching; future values will support LLM back-ends.
        Default ``"rule-based"``.
    semantic_cache_size:
        Maximum number of (text → embedding) pairs kept in the in-process
        LRU cache.  ``0`` disables caching.  Default ``1000``.
    """

    semantic_enabled: bool = field(
        default_factory=lambda: _env_bool("SEMANTIC_ENABLED", False)
    )
    semantic_model: str = field(
        default_factory=lambda: os.environ.get(
            "SEMANTIC_MODEL", "all-MiniLM-L6-v2"
        )
    )
    explanation_enabled: bool = field(
        default_factory=lambda: _env_bool("EXPLANATION_ENABLED", False)
    )
    explanation_model: str = field(
        default_factory=lambda: os.environ.get(
            "EXPLANATION_MODEL", "rule-based"
        )
    )
    semantic_cache_size: int = field(
        default_factory=lambda: int(
            os.environ.get("SEMANTIC_CACHE_SIZE", "1000")
        )
    )
