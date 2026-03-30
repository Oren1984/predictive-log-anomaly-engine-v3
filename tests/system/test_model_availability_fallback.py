# tests/system/test_model_availability_fallback.py
"""
Phase 7.5 — Model Availability and Fallback Validation.

Validates that InferenceEngine degrades gracefully when trained model files
are absent.  Uses a temporary directory with no model artifacts so no real
files are affected.

Key assertions:
  - Engine initialises without crashing (no model files present)
  - load_artifacts() completes without crashing (warns, does not raise)
  - ingest() returns a RiskResult when the window fills (fallback scorer used)
  - demo_mode=True  -> fallback_score is returned (triggers anomaly)
  - demo_mode=False -> 0.0 is returned (no spurious alert)
  - RiskResult fields are complete and well-typed in both modes
  - Alert manager receives fallback result without error
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.alerts import AlertManager, AlertPolicy
from src.runtime.inference_engine import InferenceEngine
from src.runtime.types import RiskResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _events(n: int, service: str = "bgl", session: str = "fb") -> list[dict]:
    return [
        {
            "timestamp": float(i),
            "service": service,
            "session_id": session,
            "token_id": (i % 50) + 2,
            "label": 0,
        }
        for i in range(n)
    ]


def _engine_with_empty_root(
    tmp_path: Path,
    mode: str = "baseline",
    window_size: int = 5,
    stride: int = 5,
    demo_mode: bool = True,
    fallback_score: float = 2.0,
) -> InferenceEngine:
    """Return an InferenceEngine pointed at an empty temp directory."""
    eng = InferenceEngine(
        mode=mode,
        window_size=window_size,
        stride=stride,
        root=tmp_path,
    )
    eng.demo_mode = demo_mode
    eng.fallback_score = fallback_score
    return eng


# ---------------------------------------------------------------------------
# Initialisation without model files
# ---------------------------------------------------------------------------

class TestEngineInitialisationNullRoot:

    def test_engine_construction_does_not_crash(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path)
        assert eng is not None

    def test_mode_is_set_correctly(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path, mode="baseline")
        assert eng.mode == "baseline"

    def test_artifacts_not_loaded_on_construction(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path)
        assert eng._artifacts_loaded is False

    def test_load_artifacts_does_not_raise(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path)
        try:
            eng.load_artifacts()
        except Exception as exc:
            pytest.fail(f"load_artifacts() raised with empty root: {exc}")

    def test_artifacts_marked_loaded_after_call(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path)
        eng.load_artifacts()
        assert eng._artifacts_loaded is True

    def test_baseline_model_is_none_without_file(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path)
        eng.load_artifacts()
        assert eng._baseline_model is None

    def test_extractor_is_none_without_model_file(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path)
        eng.load_artifacts()
        assert eng._extractor is None


# ---------------------------------------------------------------------------
# Fallback scoring — demo_mode=True
# ---------------------------------------------------------------------------

class TestFallbackScoringDemoMode:

    def test_ingest_returns_result_at_window_boundary(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path, demo_mode=True, fallback_score=2.0)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None

    def test_result_is_risk_result_instance(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path, demo_mode=True, fallback_score=2.0)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert isinstance(result, RiskResult)

    def test_fallback_score_returned(self, tmp_path):
        """fallback_score=2.0 should yield risk_score=2.0 when model absent."""
        eng = _engine_with_empty_root(tmp_path, demo_mode=True, fallback_score=2.0)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        assert result.risk_score == pytest.approx(2.0)

    def test_anomaly_flag_set_with_high_fallback(self, tmp_path):
        """fallback_score=2.0 >> threshold=0.33 -> is_anomaly must be True."""
        eng = _engine_with_empty_root(tmp_path, demo_mode=True, fallback_score=2.0)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        assert result.is_anomaly is True

    def test_result_fields_complete(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path, demo_mode=True, fallback_score=2.0)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        assert isinstance(result.stream_key, str)
        assert isinstance(result.risk_score, float)
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.threshold, float)
        assert isinstance(result.evidence_window, dict)
        assert isinstance(result.meta, dict)

    def test_result_model_field_is_baseline(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path, mode="baseline",
                                      demo_mode=True, fallback_score=2.0)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        assert result.model == "baseline"


# ---------------------------------------------------------------------------
# Fallback scoring — demo_mode=False (production safe default)
# ---------------------------------------------------------------------------

class TestFallbackScoringProductionMode:

    def test_ingest_still_returns_result(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path, demo_mode=False)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None

    def test_score_is_zero_in_prod_mode(self, tmp_path):
        """Production fallback returns 0.0 to avoid spurious alerts."""
        eng = _engine_with_empty_root(tmp_path, demo_mode=False)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        assert result.risk_score == pytest.approx(0.0)

    def test_no_anomaly_in_prod_mode(self, tmp_path):
        """0.0 score < any threshold -> is_anomaly=False."""
        eng = _engine_with_empty_root(tmp_path, demo_mode=False)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        assert result.is_anomaly is False

    def test_no_alert_fired_in_prod_fallback(self, tmp_path):
        eng = _engine_with_empty_root(tmp_path, demo_mode=False)
        policy = AlertPolicy(cooldown_seconds=0.0)
        manager = AlertManager(policy=policy)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        alerts = manager.emit(result)
        assert alerts == []


# ---------------------------------------------------------------------------
# Transformer and ensemble modes also degrade gracefully
# ---------------------------------------------------------------------------

class TestFallbackAllModes:

    @pytest.mark.parametrize("mode", ["baseline", "transformer", "ensemble"])
    def test_mode_survives_empty_root(self, tmp_path, mode):
        eng = _engine_with_empty_root(tmp_path, mode=mode,
                                      demo_mode=True, fallback_score=2.0)
        result = None
        for ev in _events(5):
            result = eng.ingest(ev)
        assert result is not None
        assert isinstance(result, RiskResult)

    @pytest.mark.parametrize("mode", ["baseline", "transformer", "ensemble"])
    def test_mode_does_not_crash_with_100_events(self, tmp_path, mode):
        eng = _engine_with_empty_root(tmp_path, mode=mode,
                                      window_size=10, stride=5,
                                      demo_mode=True, fallback_score=0.0)
        try:
            for ev in _events(100):
                eng.ingest(ev)
        except Exception as exc:
            pytest.fail(f"mode={mode} crashed on 100-event stream: {exc}")


# ---------------------------------------------------------------------------
# Chaos / robustness — malformed inputs
# ---------------------------------------------------------------------------

class TestMalformedInputRobustness:
    """Validate graceful handling of malformed or edge-case events."""

    def _demo_engine(self, tmp_path, window_size=3, stride=3) -> InferenceEngine:
        eng = _engine_with_empty_root(tmp_path, window_size=window_size,
                                      stride=stride, demo_mode=True,
                                      fallback_score=0.0)
        return eng

    def test_empty_dict_event(self, tmp_path):
        eng = self._demo_engine(tmp_path)
        try:
            for _ in range(3):
                eng.ingest({})
        except Exception as exc:
            pytest.fail(f"Crashed on empty dict event: {exc}")

    def test_none_token_id(self, tmp_path):
        """
        A None token_id propagates as a TypeError in the buffer's int() cast.
        This is a known limitation: the engine does not sanitize token_id.
        The test documents the behavior (crash on None) rather than asserting
        graceful handling, to avoid hiding real bugs.
        """
        eng = self._demo_engine(tmp_path)
        ev = {"timestamp": 1.0, "service": "svc", "session_id": "s", "token_id": None}
        # Feed enough to fill the window; crash occurs at get_window() on emit
        with pytest.raises((TypeError, ValueError)):
            for _ in range(3):
                eng.ingest(ev)

    def test_very_large_token_id(self, tmp_path):
        eng = self._demo_engine(tmp_path)
        ev = {"timestamp": 1.0, "service": "svc", "session_id": "s",
              "token_id": 999_999}
        try:
            for _ in range(3):
                eng.ingest(ev)
        except Exception as exc:
            pytest.fail(f"Crashed on large token_id: {exc}")

    def test_missing_service_field(self, tmp_path):
        eng = self._demo_engine(tmp_path)
        ev = {"timestamp": 1.0, "token_id": 5}
        try:
            for _ in range(3):
                eng.ingest(ev)
        except Exception as exc:
            pytest.fail(f"Crashed on missing service field: {exc}")

    def test_empty_string_service(self, tmp_path):
        eng = self._demo_engine(tmp_path)
        ev = {"service": "", "session_id": "", "token_id": 5}
        try:
            for _ in range(3):
                eng.ingest(ev)
        except Exception as exc:
            pytest.fail(f"Crashed on empty-string service: {exc}")

    def test_huge_message_field_ignored_gracefully(self, tmp_path):
        eng = self._demo_engine(tmp_path)
        big_msg = "X" * 100_000
        ev = {"service": "svc", "session_id": "s", "token_id": 5,
              "message": big_msg}
        try:
            for _ in range(3):
                eng.ingest(ev)
        except Exception as exc:
            pytest.fail(f"Crashed on huge message field: {exc}")

    def test_mixed_valid_invalid_events(self, tmp_path):
        """Engine stays alive after a mix of valid and invalid events."""
        eng = self._demo_engine(tmp_path, window_size=5, stride=5)
        events = [
            {},
            {"service": "s", "session_id": "x", "token_id": 10},
            {"service": None, "session_id": None, "token_id": "bad"},
            {"service": "s", "session_id": "x", "token_id": 20},
            {"service": "s", "session_id": "x", "token_id": 30},
        ]
        try:
            for ev in events:
                eng.ingest(ev)
        except Exception as exc:
            pytest.fail(f"Crashed on mixed input stream: {exc}")
