# src/semantic/similarity.py
#
# V3 Semantic layer — cosine similarity between sentence embeddings.
#
# Inert by default: all methods require caller to check SemanticConfig.semantic_enabled.
# This module has no side effects on import and no dependency on sentence-transformers.

"""V3 Semantic layer — cosine similarity utilities."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class SemanticSimilarity:
    """
    Computes cosine similarity between sentence embeddings.

    All methods are pure functions with no internal state — instantiate once
    and reuse freely across calls.
    """

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Return cosine similarity between two 1-D float embeddings in [-1.0, 1.0].

        Returns 0.0 when either vector is all-zeros (avoids division-by-zero).
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def top_k(
        self,
        query: np.ndarray,
        candidates: List[Tuple[str, np.ndarray]],
        k: int = 3,
    ) -> List[dict]:
        """
        Return the top-k most similar entries from a candidate list.

        Parameters
        ----------
        query      : 1-D float embedding for the query
        candidates : list of (label_str, embedding) pairs to rank
        k          : maximum number of results to return

        Returns
        -------
        list of dicts ``{"text": str, "score": float}`` sorted by score descending.
        Empty list when candidates is empty.
        """
        if not candidates:
            return []
        scored = [
            {"text": label, "score": self.compute(query, emb)}
            for label, emb in candidates
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
