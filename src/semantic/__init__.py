# src/semantic/__init__.py
#
# V3 Semantic layer — public API.
#
# Gated by SEMANTIC_ENABLED=false by default.
# Integrated into the pipeline in Phase 7; disabled flow is fully inert.

from .config import SemanticConfig
from .embeddings import SemanticEmbedder
from .explainer import RuleBasedExplainer
from .loader import SemanticModelLoader
from .similarity import SemanticSimilarity

__all__ = [
    "SemanticConfig",
    "SemanticModelLoader",
    "SemanticEmbedder",
    "SemanticSimilarity",
    "RuleBasedExplainer",
]
