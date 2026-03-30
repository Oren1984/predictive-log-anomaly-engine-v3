# src/semantic/__init__.py
#
# V3 Semantic layer — public API.
#
# This package is INERT by default (SEMANTIC_ENABLED=false).
# It is not connected to the runtime pipeline or API routes.
# Integration will happen in a later phase.

from .config import SemanticConfig
from .embeddings import SemanticEmbedder
from .loader import SemanticModelLoader

__all__ = [
    "SemanticConfig",
    "SemanticModelLoader",
    "SemanticEmbedder",
]
