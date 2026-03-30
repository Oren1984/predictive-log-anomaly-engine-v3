# tests/unit/test_semantic.py
#
# Unit tests for the V3 semantic layer (src/semantic/).
#
# Design:
#   - All tests use mocked sentence-transformers so the test suite runs
#     without requiring the heavy model download.
#   - Tests verify inert behaviour when SEMANTIC_ENABLED=false (the default).
#   - Tests verify correct delegation when SEMANTIC_ENABLED=true with a mock.
#   - Phase 7: additional tests for SemanticSimilarity and RuleBasedExplainer.

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.semantic.config import SemanticConfig
from src.semantic.embeddings import SemanticEmbedder
from src.semantic.explainer import RuleBasedExplainer
from src.semantic.loader import SemanticModelLoader
from src.semantic.similarity import SemanticSimilarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def disabled_config() -> SemanticConfig:
    """Config with semantic layer disabled (default state)."""
    return SemanticConfig(
        semantic_enabled=False,
        semantic_model="all-MiniLM-L6-v2",
        explanation_enabled=False,
        explanation_model="rule-based",
        semantic_cache_size=1000,
    )


@pytest.fixture()
def enabled_config() -> SemanticConfig:
    """Config with semantic layer enabled."""
    return SemanticConfig(
        semantic_enabled=True,
        semantic_model="all-MiniLM-L6-v2",
        explanation_enabled=False,
        explanation_model="rule-based",
        semantic_cache_size=128,
    )


@pytest.fixture()
def mock_sentence_transformer():
    """
    Inject a fake sentence_transformers module into sys.modules so that the
    lazy `from sentence_transformers import SentenceTransformer` inside
    SemanticModelLoader.load() resolves without the real library installed.
    Returns (FakeClass, fake_model_instance).
    """
    fake_model = MagicMock()
    fake_model.encode.return_value = np.ones(384, dtype=np.float32)

    fake_cls = MagicMock(return_value=fake_model)
    fake_module = MagicMock()
    fake_module.SentenceTransformer = fake_cls

    with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
        yield fake_cls, fake_model


# ---------------------------------------------------------------------------
# SemanticConfig tests
# ---------------------------------------------------------------------------

class TestSemanticConfig:
    def test_defaults_are_disabled(self):
        """All safety-sensitive flags must default to False."""
        cfg = SemanticConfig()
        assert cfg.semantic_enabled is False
        assert cfg.explanation_enabled is False

    def test_default_model_names(self):
        cfg = SemanticConfig()
        assert cfg.semantic_model == "all-MiniLM-L6-v2"
        assert cfg.explanation_model == "rule-based"

    def test_default_cache_size(self):
        cfg = SemanticConfig()
        assert cfg.semantic_cache_size == 1000

    def test_env_override_semantic_enabled(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ENABLED", "true")
        cfg = SemanticConfig()
        assert cfg.semantic_enabled is True

    def test_env_override_explanation_enabled(self, monkeypatch):
        monkeypatch.setenv("EXPLANATION_ENABLED", "true")
        cfg = SemanticConfig()
        assert cfg.explanation_enabled is True

    def test_env_override_model(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_MODEL", "paraphrase-MiniLM-L3-v2")
        cfg = SemanticConfig()
        assert cfg.semantic_model == "paraphrase-MiniLM-L3-v2"

    def test_env_override_cache_size(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_CACHE_SIZE", "512")
        cfg = SemanticConfig()
        assert cfg.semantic_cache_size == 512

    def test_env_false_variants(self, monkeypatch):
        for val in ("false", "0", "no", "FALSE"):
            monkeypatch.setenv("SEMANTIC_ENABLED", val)
            cfg = SemanticConfig()
            assert cfg.semantic_enabled is False

    def test_env_true_variants(self, monkeypatch):
        for val in ("true", "1", "yes", "TRUE"):
            monkeypatch.setenv("SEMANTIC_ENABLED", val)
            cfg = SemanticConfig()
            assert cfg.semantic_enabled is True


# ---------------------------------------------------------------------------
# SemanticModelLoader tests
# ---------------------------------------------------------------------------

class TestSemanticModelLoader:
    def test_not_ready_by_default(self, disabled_config):
        loader = SemanticModelLoader(disabled_config)
        assert loader.is_ready is False
        assert loader.model is None

    def test_load_noop_when_disabled(self, disabled_config):
        """load() must not import sentence-transformers when disabled."""
        loader = SemanticModelLoader(disabled_config)
        loader.load()  # must not raise even if library absent
        assert loader.is_ready is False

    def test_load_raises_import_error_when_missing(self, enabled_config):
        """When enabled but sentence-transformers not installed, raise ImportError."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            loader = SemanticModelLoader(enabled_config)
            with pytest.raises(ImportError, match="sentence-transformers"):
                loader.load()

    def test_load_sets_model_when_enabled(
        self, enabled_config, mock_sentence_transformer
    ):
        patched_cls, fake_model = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()

        assert loader.is_ready is True
        assert loader.model is fake_model
        patched_cls.assert_called_once_with("all-MiniLM-L6-v2")

    def test_load_is_idempotent(self, enabled_config, mock_sentence_transformer):
        patched_cls, _ = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()
        loader.load()  # second call must not re-instantiate
        assert patched_cls.call_count == 1

    def test_unload_clears_model(self, enabled_config, mock_sentence_transformer):
        _, _ = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()
        assert loader.is_ready is True
        loader.unload()
        assert loader.is_ready is False
        assert loader.model is None


# ---------------------------------------------------------------------------
# SemanticEmbedder tests — disabled mode
# ---------------------------------------------------------------------------

class TestSemanticEmbedderDisabled:
    def test_embed_returns_none_when_disabled(self, disabled_config):
        embedder = SemanticEmbedder(config=disabled_config)
        result = embedder.embed("Connection timed out on host db-01")
        assert result is None

    def test_embed_batch_returns_empty_when_disabled(self, disabled_config):
        embedder = SemanticEmbedder(config=disabled_config)
        result = embedder.embed_batch(["log line 1", "log line 2"])
        assert result == []

    def test_embed_does_not_call_model_when_disabled(self, disabled_config):
        loader = MagicMock()
        loader.is_ready = True  # even if somehow ready, disabled config wins
        embedder = SemanticEmbedder(config=disabled_config, loader=loader)
        embedder.embed("some text")
        loader.model.encode.assert_not_called()


# ---------------------------------------------------------------------------
# SemanticEmbedder tests — enabled mode with mocked model
# ---------------------------------------------------------------------------

class TestSemanticEmbedderEnabled:
    def test_embed_returns_ndarray(
        self, enabled_config, mock_sentence_transformer
    ):
        _, fake_model = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()
        embedder = SemanticEmbedder(config=enabled_config, loader=loader)

        result = embedder.embed("disk usage exceeded threshold on node-07")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (384,)

    def test_embed_batch_returns_list(
        self, enabled_config, mock_sentence_transformer
    ):
        _, fake_model = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()
        embedder = SemanticEmbedder(config=enabled_config, loader=loader)

        texts = ["OOM killed pid 1234", "SSH login failed from 10.0.0.1"]
        results = embedder.embed_batch(texts)
        assert len(results) == 2
        for vec in results:
            assert vec is not None
            assert isinstance(vec, np.ndarray)

    def test_embed_caches_results(
        self, enabled_config, mock_sentence_transformer
    ):
        _, fake_model = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()
        embedder = SemanticEmbedder(config=enabled_config, loader=loader)

        text = "repeated log line"
        embedder.embed(text)
        embedder.embed(text)  # second call should hit cache
        # encode should have been called only once
        assert fake_model.encode.call_count == 1

    def test_cache_clear_resets_cache(
        self, enabled_config, mock_sentence_transformer
    ):
        _, fake_model = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()
        embedder = SemanticEmbedder(config=enabled_config, loader=loader)

        text = "some log line"
        embedder.embed(text)
        embedder.cache_clear()
        embedder.embed(text)  # must recompute after clear
        assert fake_model.encode.call_count == 2

    def test_cache_info_returns_string(
        self, enabled_config, mock_sentence_transformer
    ):
        _, _ = mock_sentence_transformer
        loader = SemanticModelLoader(enabled_config)
        loader.load()
        embedder = SemanticEmbedder(config=enabled_config, loader=loader)
        info = embedder.cache_info()
        assert isinstance(info, str)
        assert "hits" in info
        assert "misses" in info

    def test_embed_returns_none_when_model_not_loaded(self, enabled_config):
        """embed() returns None if loader.is_ready is False even when enabled."""
        loader = SemanticModelLoader(enabled_config)
        # intentionally do NOT call load()
        embedder = SemanticEmbedder(config=enabled_config, loader=loader)
        result = embedder.embed("some log")
        assert result is None


# ---------------------------------------------------------------------------
# Package import test
# ---------------------------------------------------------------------------

class TestSemanticPackageImport:
    def test_package_importable(self):
        """The semantic package must be importable without sentence-transformers."""
        from src.semantic import SemanticConfig, SemanticEmbedder, SemanticModelLoader
        assert SemanticConfig is not None
        assert SemanticModelLoader is not None
        assert SemanticEmbedder is not None

    def test_disabled_by_default_at_module_level(self):
        """Importing the module must not trigger any model loading."""
        import src.semantic  # noqa: F401  — must not raise
        cfg = SemanticConfig()
        assert cfg.semantic_enabled is False


# ---------------------------------------------------------------------------
# SemanticSimilarity tests (Phase 7)
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    def setup_method(self):
        self.sim = SemanticSimilarity()

    def _unit(self, *values: float) -> np.ndarray:
        """Return a float32 vector from given values."""
        return np.array(values, dtype=np.float32)

    def test_identical_vectors_return_one(self):
        v = self._unit(1.0, 0.0, 0.0)
        assert abs(self.sim.compute(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        a = self._unit(1.0, 0.0)
        b = self._unit(0.0, 1.0)
        assert abs(self.sim.compute(a, b)) < 1e-6

    def test_opposite_vectors_return_minus_one(self):
        a = self._unit(1.0, 0.0)
        b = self._unit(-1.0, 0.0)
        assert abs(self.sim.compute(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        zero = self._unit(0.0, 0.0, 0.0)
        other = self._unit(1.0, 2.0, 3.0)
        assert self.sim.compute(zero, other) == 0.0
        assert self.sim.compute(other, zero) == 0.0

    def test_top_k_empty_candidates_returns_empty(self):
        result = self.sim.top_k(self._unit(1.0, 0.0), candidates=[], k=3)
        assert result == []

    def test_top_k_sorted_by_score_descending(self):
        query = self._unit(1.0, 0.0)
        candidates = [
            ("low", self._unit(0.0, 1.0)),    # orthogonal → score ≈ 0
            ("high", self._unit(1.0, 0.0)),   # identical → score ≈ 1
            ("mid", self._unit(1.0, 1.0)),    # 45° → score ≈ 0.707
        ]
        result = self.sim.top_k(query, candidates, k=3)
        assert len(result) == 3
        assert result[0]["text"] == "high"
        assert result[1]["text"] == "mid"
        assert result[2]["text"] == "low"

    def test_top_k_respects_k_limit(self):
        query = self._unit(1.0, 0.0)
        candidates = [(f"item{i}", self._unit(float(i), 0.0)) for i in range(10)]
        result = self.sim.top_k(query, candidates, k=3)
        assert len(result) == 3

    def test_top_k_result_has_text_and_score_keys(self):
        query = self._unit(1.0, 0.0)
        candidates = [("alpha", self._unit(1.0, 0.0))]
        result = self.sim.top_k(query, candidates, k=1)
        assert "text" in result[0]
        assert "score" in result[0]
        assert isinstance(result[0]["score"], float)


# ---------------------------------------------------------------------------
# RuleBasedExplainer tests (Phase 7)
# ---------------------------------------------------------------------------

class TestRuleBasedExplainer:
    def setup_method(self):
        self.exp = RuleBasedExplainer()

    def _explain(self, templates=None, token_count=0) -> dict:
        ew = {"templates_preview": templates or [], "token_count": token_count}
        return self.exp.explain(ew)

    def test_returns_explanation_string(self):
        result = self._explain(["normal startup sequence"], token_count=10)
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_returns_evidence_tokens_list(self):
        result = self._explain(["disk error detected"], token_count=5)
        assert isinstance(result["evidence_tokens"], list)

    def test_error_keyword_detected(self):
        result = self._explain(["Connection timeout on db-01"], token_count=10)
        assert "error-indicative" in result["explanation"]
        assert len(result["evidence_tokens"]) >= 1

    def test_multiple_error_templates_counted(self):
        templates = [
            "HDFS: failed to allocate block",
            "SSH: exception in handler",
            "Service timeout after 30s",
        ]
        result = self._explain(templates, token_count=3)
        assert "3" in result["explanation"] or "error-indicative" in result["explanation"]

    def test_no_error_keywords_gives_generic_message(self):
        # 3+ unique templates, no error keywords, moderate token_count → no rule fires
        templates = [
            "info: heartbeat ok",
            "info: stats flushed",
            "info: connection established",
        ]
        result = self._explain(templates, token_count=20)
        assert "threshold" in result["explanation"] or "rule" in result["explanation"]
        assert result["evidence_tokens"] == []

    def test_low_diversity_flagged(self):
        # Only 1 unique template but token_count > 0 → diversity rule fires
        result = self._explain(["same template"] * 5, token_count=5)
        assert "diversity" in result["explanation"]

    def test_high_density_flagged(self):
        result = self._explain(["info: heartbeat ok"], token_count=100)
        assert "density" in result["explanation"] or "100" in result["explanation"]

    def test_empty_window_flagged(self):
        result = self._explain(templates=[], token_count=0)
        assert "empty" in result["explanation"].lower() or "threshold" in result["explanation"].lower()

    def test_evidence_tokens_capped_at_five(self):
        templates = [f"error event {i}" for i in range(10)]
        result = self._explain(templates, token_count=10)
        assert len(result["evidence_tokens"]) <= 5

    def test_empty_evidence_window_dict(self):
        """explain() must not raise on an empty dict."""
        result = self.exp.explain({})
        assert isinstance(result["explanation"], str)
        assert isinstance(result["evidence_tokens"], list)
