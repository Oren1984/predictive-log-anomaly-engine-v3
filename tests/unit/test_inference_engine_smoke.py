# test/unit/test_inference_engine_smoke.py

# Purpose: Smoke tests for the InferenceEngine end-to-end flow.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are unit tests for the InferenceEngine class)

"""Smoke tests for InferenceEngine end-to-end flow."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_slow = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.inference_engine import InferenceEngine
from src.runtime.types import RiskResult

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACTS = [
    ROOT / "artifacts" / "vocab.json",
    ROOT / "artifacts" / "threshold.json",
    ROOT / "models" / "baseline.pkl",
]

_artifacts_present = all(p.exists() for p in REQUIRED_ARTIFACTS)
_transformer_present = (ROOT / "models" / "transformer.pt").exists()

needs_artifacts = pytest.mark.skipif(
    not _artifacts_present,
    reason="Baseline model artifacts not found; skipping smoke tests",
)
needs_transformer = pytest.mark.skipif(
    not (_artifacts_present and _transformer_present),
    reason="Transformer model artifacts not found; skipping transformer smoke tests",
)


def _synthetic_events(
    n: int,
    service: str = "bgl",
    session: str = "win_0",
    token_id: int = 5415,   # a real token seen in training
) -> list[dict]:
    return [
        {
            "timestamp":   float(i),
            "service":     service,
            "session_id":  session,
            "token_id":    token_id,
            "template_id": token_id - 2,
            "label":       0,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    def test_valid_modes(self):
        for m in ("baseline", "transformer", "ensemble"):
            eng = InferenceEngine(mode=m, window_size=10, stride=5)
            assert eng.mode == m

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            InferenceEngine(mode="invalid")

    def test_default_params(self):
        eng = InferenceEngine()
        assert eng.window_size == 50
        assert eng.stride == 10
        assert eng.mode == "baseline"


# ---------------------------------------------------------------------------
# Returns None until window full
# ---------------------------------------------------------------------------

@_slow
class TestNoResultBeforeWindowFull:
    @needs_artifacts
    def test_none_until_window_full(self):
        W = 20
        eng = InferenceEngine(mode="baseline", window_size=W, stride=5)
        eng.load_artifacts()
        events = _synthetic_events(W - 1)
        for ev in events:
            result = eng.ingest(ev)
            assert result is None, f"Expected None before window full, got {result}"

    @needs_artifacts
    def test_returns_result_at_window_size(self):
        W = 20
        eng = InferenceEngine(mode="baseline", window_size=W, stride=5)
        eng.load_artifacts()
        events = _synthetic_events(W)
        result = None
        for ev in events:
            result = eng.ingest(ev)
        assert result is not None

    @needs_artifacts
    def test_result_is_risk_result_instance(self):
        W = 15
        eng = InferenceEngine(mode="baseline", window_size=W, stride=5)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert isinstance(result, RiskResult)


# ---------------------------------------------------------------------------
# RiskResult field validation
# ---------------------------------------------------------------------------

@_slow
class TestRiskResultFields:
    @needs_artifacts
    def test_all_required_fields_present(self):
        W = 10
        eng = InferenceEngine(mode="baseline", window_size=W, stride=10)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert result is not None

        # Required fields
        assert isinstance(result.stream_key, str)
        assert result.model in ("baseline", "transformer", "ensemble")
        assert isinstance(result.risk_score, float)
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.threshold, float)
        assert isinstance(result.evidence_window, dict)
        assert isinstance(result.meta, dict)

    @needs_artifacts
    def test_evidence_window_keys(self):
        W = 10
        eng = InferenceEngine(mode="baseline", window_size=W, stride=10)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert result is not None
        ew = result.evidence_window
        for key in ("tokens", "template_ids", "templates_preview",
                    "window_start_ts", "window_end_ts"):
            assert key in ew, f"Missing key in evidence_window: {key}"

    @needs_artifacts
    def test_evidence_tokens_not_empty(self):
        W = 10
        eng = InferenceEngine(mode="baseline", window_size=W, stride=10)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert result is not None
        assert len(result.evidence_window["tokens"]) > 0

    @needs_artifacts
    def test_meta_contains_window_size(self):
        W = 10
        eng = InferenceEngine(mode="baseline", window_size=W, stride=10)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert result is not None
        assert result.meta.get("window_size") == W

    @needs_artifacts
    def test_stream_key_format(self):
        W = 10
        eng = InferenceEngine(mode="baseline", window_size=W, stride=10)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W, service="bgl", session="win_0"):
            result = eng.ingest(ev)
        assert result is not None
        assert result.stream_key == "bgl:win_0"


# ---------------------------------------------------------------------------
# Stride behaviour in engine
# ---------------------------------------------------------------------------

@_slow
class TestEngineStride:
    @needs_artifacts
    def test_multiple_emissions_with_stride(self):
        W, S = 10, 5
        eng = InferenceEngine(mode="baseline", window_size=W, stride=S)
        eng.load_artifacts()

        emitted = []
        for ev in _synthetic_events(W + S * 2):
            r = eng.ingest(ev)
            if r is not None:
                emitted.append(r)

        # Should emit at events 10, 15, 20 → 3 results
        assert len(emitted) == 3

    @needs_artifacts
    def test_risk_scores_are_finite(self):
        W, S = 10, 5
        eng = InferenceEngine(mode="baseline", window_size=W, stride=S)
        eng.load_artifacts()

        import math
        for ev in _synthetic_events(W + S):
            r = eng.ingest(ev)
            if r is not None:
                assert math.isfinite(r.risk_score)


# ---------------------------------------------------------------------------
# Baseline scoring helper
# ---------------------------------------------------------------------------

@_slow
class TestScoreBaseline:
    @needs_artifacts
    def test_score_baseline_returns_float(self):
        eng = InferenceEngine(mode="baseline")
        eng.load_artifacts()

        from src.sequencing.models import Sequence
        seq = Sequence(
            sequence_id="test",
            tokens=[5415] * 20,
            label=0,
        )
        score = eng.score_baseline(seq)
        assert isinstance(score, float)
        assert score >= 0.0   # negated IsolationForest score is non-negative

    @needs_artifacts
    def test_decide_returns_bool(self):
        eng = InferenceEngine(mode="baseline")
        eng.load_artifacts()
        assert eng.decide(0.5, 0.3) is True
        assert eng.decide(0.2, 0.3) is False


# ---------------------------------------------------------------------------
# Transformer mode (smoke)
# ---------------------------------------------------------------------------

@_slow
class TestTransformerSmoke:
    @needs_transformer
    def test_transformer_mode_returns_result(self):
        W = 15
        eng = InferenceEngine(mode="transformer", window_size=W, stride=W)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert result is not None
        assert result.model == "transformer"

    @needs_transformer
    def test_ensemble_mode_returns_result(self):
        W = 15
        eng = InferenceEngine(mode="ensemble", window_size=W, stride=W)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert result is not None
        assert result.model == "ensemble"
        # Ensemble threshold is 1.0 (normalised)
        assert result.threshold == 1.0


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

@_slow
class TestRiskResultSerialisation:
    @needs_artifacts
    def test_to_dict_is_serialisable(self):
        import json
        W = 10
        eng = InferenceEngine(mode="baseline", window_size=W, stride=10)
        eng.load_artifacts()
        result = None
        for ev in _synthetic_events(W):
            result = eng.ingest(ev)
        assert result is not None
        d = result.to_dict()
        # Should be JSON-serialisable
        text = json.dumps(d, default=str)
        reloaded = json.loads(text)
        assert reloaded["model"] == "baseline"
