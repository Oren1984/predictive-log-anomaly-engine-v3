# tests/system/test_streaming_simulation.py
"""
Phase 7.5 — Streaming Simulation Validation.

Validates rolling-buffer and stride behavior when many sequential events
are ingested into InferenceEngine.  Uses demo_mode=True so the fallback
scorer returns a fixed score without needing any trained model files.

Scenarios:
  - Single-service stream of 1 000 events -> count emitted windows
  - Multi-service interleaved stream -> per-service buffer isolation
  - LRU eviction when max_stream_keys is exceeded
  - Buffer capacity stays bounded at window_size (no unbounded growth)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.inference_engine import InferenceEngine
from src.runtime.sequence_buffer import SequenceBuffer
from src.runtime.types import RiskResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(
    i: int,
    service: str = "bgl",
    session: str = "sim",
    token_id: int = 10,
) -> dict:
    return {
        "timestamp": float(i),
        "service": service,
        "session_id": session,
        "token_id": (token_id + i) % 200 + 2,  # vary token within valid range
        "label": 0,
    }


def _stream(n: int, **kwargs) -> list[dict]:
    return [_event(i, **kwargs) for i in range(n)]


def _make_engine(window_size: int = 10, stride: int = 5) -> InferenceEngine:
    eng = InferenceEngine(mode="baseline", window_size=window_size, stride=stride)
    eng.demo_mode = True
    eng.fallback_score = 2.0
    return eng


# ---------------------------------------------------------------------------
# SequenceBuffer-level streaming tests (no model loading)
# ---------------------------------------------------------------------------

class TestSequenceBufferStreaming:
    """Unit-level buffer behavior during long streams."""

    def test_buffer_bounded_by_window_size(self):
        W = 20
        buf = SequenceBuffer(window_size=W, stride=5)
        for i in range(200):
            buf.ingest(_event(i))
        assert buf.buffer_length("bgl:sim") <= W

    def test_emission_count_single_stream(self):
        """For N events, W window, S stride: expected = (N - W) // S + 1."""
        W, S, N = 10, 5, 1000
        expected_emissions = (N - W) // S + 1
        buf = SequenceBuffer(window_size=W, stride=S)
        emitted = 0
        for ev in _stream(N):
            key = buf.ingest(ev)
            if buf.should_emit(key):
                buf.get_window(key)
                emitted += 1
        assert emitted == expected_emissions

    def test_single_key_in_buffer_for_single_service(self):
        buf = SequenceBuffer(window_size=5, stride=1)
        for ev in _stream(100):
            buf.ingest(ev)
        assert len(buf.active_keys()) == 1

    def test_multiple_keys_for_multiple_services(self):
        buf = SequenceBuffer(window_size=5, stride=1)
        for ev in _stream(50, service="svc_a"):
            buf.ingest(ev)
        for ev in _stream(50, service="svc_b"):
            buf.ingest(ev)
        keys = buf.active_keys()
        assert "svc_a:sim" in keys
        assert "svc_b:sim" in keys

    def test_lru_eviction_keeps_max_keys(self):
        max_k = 3
        buf = SequenceBuffer(window_size=5, stride=5, max_stream_keys=max_k)
        for svc_idx in range(10):
            for ev in _stream(5, service=f"svc_{svc_idx}"):
                buf.ingest(ev)
        assert len(buf.active_keys()) <= max_k

    def test_clear_resets_all_state(self):
        buf = SequenceBuffer(window_size=5, stride=5)
        for ev in _stream(20):
            buf.ingest(ev)
        buf.clear()
        assert len(buf) == 0
        assert buf.active_keys() == []


# ---------------------------------------------------------------------------
# InferenceEngine-level streaming tests
# ---------------------------------------------------------------------------

class TestEngineStreaming:
    """Streaming tests through the full InferenceEngine with demo fallback."""

    def test_single_stream_emits_correct_count(self):
        W, S, N = 10, 5, 1000
        expected = (N - W) // S + 1
        eng = _make_engine(window_size=W, stride=S)
        emitted = []
        for ev in _stream(N):
            r = eng.ingest(ev)
            if r is not None:
                emitted.append(r)
        assert len(emitted) == expected

    def test_all_results_are_risk_result_instances(self):
        eng = _make_engine(window_size=10, stride=5)
        for ev in _stream(200):
            r = eng.ingest(ev)
            if r is not None:
                assert isinstance(r, RiskResult)

    def test_stream_key_consistent_per_service(self):
        eng = _make_engine(window_size=10, stride=5)
        for ev in _stream(100, service="bgl", session="s1"):
            r = eng.ingest(ev)
            if r is not None:
                assert r.stream_key == "bgl:s1"

    def test_two_interleaved_services_independent(self):
        eng = _make_engine(window_size=10, stride=5)
        counts = {"a": 0, "b": 0}
        events = []
        for i in range(500):
            svc = "a" if i % 2 == 0 else "b"
            events.append(_event(i, service=svc, session=svc))

        for ev in events:
            r = eng.ingest(ev)
            if r is not None:
                svc = r.stream_key.split(":")[0]
                counts[svc] += 1

        # Both services should have emitted at least some windows
        assert counts["a"] > 0
        assert counts["b"] > 0

    def test_no_crash_on_1000_events(self):
        eng = _make_engine(window_size=10, stride=5)
        try:
            for ev in _stream(1000):
                eng.ingest(ev)
        except Exception as exc:
            pytest.fail(f"Engine crashed during 1000-event stream: {exc}")

    def test_result_score_is_finite(self):
        import math
        eng = _make_engine(window_size=10, stride=10)
        for ev in _stream(100):
            r = eng.ingest(ev)
            if r is not None:
                assert math.isfinite(r.risk_score), f"Non-finite score: {r.risk_score}"

    def test_result_has_evidence_window(self):
        eng = _make_engine(window_size=10, stride=10)
        for ev in _stream(10):
            r = eng.ingest(ev)
        assert r is not None
        assert isinstance(r.evidence_window, dict)
        assert "tokens" in r.evidence_window

    def test_emit_count_in_meta_increments(self):
        """emit_index in meta is the post-increment counter (1-based sequence)."""
        W = 10
        eng = _make_engine(window_size=W, stride=W)
        results = []
        for ev in _stream(W * 5):
            r = eng.ingest(ev)
            if r is not None:
                results.append(r)
        emit_indices = [r.meta.get("emit_index", -1) for r in results]
        # emit_index is recorded after get_window() increments the counter,
        # so the first window has emit_index=1, second=2, etc.
        assert emit_indices == sorted(emit_indices)
        assert emit_indices[0] >= 1

    def test_buffer_length_stays_bounded(self):
        W = 20
        eng = _make_engine(window_size=W, stride=5)
        for ev in _stream(500):
            eng.ingest(ev)
        length = eng.buffer.buffer_length("bgl:sim")
        assert length <= W

    def test_lru_eviction_prevents_unbounded_key_growth(self):
        max_k = 10
        eng = InferenceEngine(
            mode="baseline",
            window_size=5,
            stride=5,
            max_stream_keys=max_k,
        )
        eng.demo_mode = True
        eng.fallback_score = 0.0  # avoid anomalies
        for svc_idx in range(50):
            for ev in _stream(5, service=f"svc_{svc_idx}"):
                eng.ingest(ev)
        assert len(eng.buffer.active_keys()) <= max_k
