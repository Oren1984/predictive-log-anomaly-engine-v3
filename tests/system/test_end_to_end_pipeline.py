# tests/system/test_end_to_end_pipeline.py
"""
Phase 7.5 — End-to-End Pipeline Validation.

Validates the complete pipeline flow:
  raw log event -> InferenceEngine (buffer + scoring) -> RiskResult
               -> AlertManager -> Alert

Two test strategies are used:
  1. MockPipeline  : fast, no model loading, exercising the full HTTP-level flow.
  2. InferenceEngine with demo_mode=True : real buffer + real result construction,
     using the fallback scorer when trained models are absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.alerts import AlertManager, AlertPolicy
from src.runtime.inference_engine import InferenceEngine
from src.runtime.types import RiskResult
from tests.helpers_stage_07 import MockPipeline, _stub_risk_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    service: str = "bgl",
    session: str = "s0",
    token_id: int = 42,
    timestamp: float = 1_000.0,
    label: int = 0,
) -> dict:
    return {
        "timestamp": timestamp,
        "service": service,
        "session_id": session,
        "token_id": token_id,
        "template_id": token_id - 2,
        "label": label,
    }


def _make_events(n: int, **kwargs) -> list[dict]:
    base = kwargs.pop("timestamp", 0.0)
    return [_make_event(timestamp=base + i, **kwargs) for i in range(n)]


# ---------------------------------------------------------------------------
# Section 1: MockPipeline structural tests (fast, no models)
# ---------------------------------------------------------------------------

class TestMockPipelineStructure:
    """Verify that MockPipeline returns correctly-structured dicts."""

    def test_process_event_keys_present(self):
        pipeline = MockPipeline()
        result = pipeline.process_event(_make_event())
        assert "window_emitted" in result
        assert "risk_result" in result
        assert "alert" in result

    def test_no_window_before_stride(self):
        pipeline = MockPipeline()
        result = pipeline.process_event(_make_event())
        assert result["window_emitted"] is False
        assert result["risk_result"] is None

    def test_window_emitted_when_stub_set(self):
        pipeline = MockPipeline()
        pipeline.engine.next_result = _stub_risk_result(is_anomaly=False)
        result = pipeline.process_event(_make_event())
        assert result["window_emitted"] is True
        assert result["risk_result"] is not None

    def test_anomaly_triggers_alert(self):
        pipeline = MockPipeline()
        pipeline.engine.next_result = _stub_risk_result(
            score=2.0, is_anomaly=True, threshold=1.0
        )
        result = pipeline.process_event(_make_event())
        assert result["window_emitted"] is True
        assert result["alert"] is not None

    def test_normal_result_no_alert(self):
        pipeline = MockPipeline()
        pipeline.engine.next_result = _stub_risk_result(
            score=0.1, is_anomaly=False, threshold=1.0
        )
        result = pipeline.process_event(_make_event())
        assert result["alert"] is None

    def test_alert_fields_present(self):
        pipeline = MockPipeline()
        pipeline.engine.next_result = _stub_risk_result(
            score=2.5, is_anomaly=True, threshold=1.0
        )
        result = pipeline.process_event(_make_event())
        alert = result["alert"]
        assert alert is not None
        for field in ("alert_id", "severity", "service", "score", "timestamp"):
            assert field in alert, f"Missing field in alert: {field}"

    def test_alert_severity_is_string(self):
        pipeline = MockPipeline()
        pipeline.engine.next_result = _stub_risk_result(
            score=2.5, is_anomaly=True, threshold=1.0
        )
        result = pipeline.process_event(_make_event())
        assert isinstance(result["alert"]["severity"], str)

    def test_recent_alerts_returns_list(self):
        pipeline = MockPipeline()
        alerts = pipeline.recent_alerts()
        assert isinstance(alerts, list)

    def test_recent_alerts_grows_after_anomaly(self):
        pipeline = MockPipeline()
        pipeline.engine.next_result = _stub_risk_result(
            score=2.0, is_anomaly=True, threshold=1.0
        )
        pipeline.process_event(_make_event())
        assert len(pipeline.recent_alerts()) == 1

    def test_risk_result_dict_json_serialisable(self):
        pipeline = MockPipeline()
        pipeline.engine.next_result = _stub_risk_result(
            score=0.9, is_anomaly=False, threshold=1.0
        )
        result = pipeline.process_event(_make_event())
        rr = result["risk_result"]
        assert rr is not None
        text = json.dumps(rr, default=str)
        parsed = json.loads(text)
        assert "risk_score" in parsed
        assert "is_anomaly" in parsed

    def test_load_models_noop(self):
        pipeline = MockPipeline()
        pipeline.load_models()  # Should not raise


# ---------------------------------------------------------------------------
# Section 2: InferenceEngine with demo_mode (fallback, no real models needed)
# ---------------------------------------------------------------------------

class TestInferenceEngineEndToEnd:
    """Test InferenceEngine with demo_mode=True so fallback scorer is used."""

    def _make_engine(self, window_size: int = 10, stride: int = 10) -> InferenceEngine:
        eng = InferenceEngine(
            mode="baseline",
            window_size=window_size,
            stride=stride,
        )
        eng.demo_mode = True
        eng.fallback_score = 2.0  # above any reasonable threshold
        return eng

    def test_engine_returns_none_before_window_full(self):
        eng = self._make_engine(window_size=10, stride=10)
        for ev in _make_events(9):
            result = eng.ingest(ev)
            assert result is None

    def test_engine_returns_result_at_window_boundary(self):
        eng = self._make_engine(window_size=10, stride=10)
        result = None
        for ev in _make_events(10):
            result = eng.ingest(ev)
        assert result is not None

    def test_result_is_risk_result_instance(self):
        eng = self._make_engine(window_size=10, stride=10)
        result = None
        for ev in _make_events(10):
            result = eng.ingest(ev)
        assert isinstance(result, RiskResult)

    def test_result_fields_are_correct_types(self):
        eng = self._make_engine(window_size=10, stride=10)
        result = None
        for ev in _make_events(10):
            result = eng.ingest(ev)
        assert result is not None
        assert isinstance(result.stream_key, str)
        assert isinstance(result.risk_score, float)
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.threshold, float)
        assert isinstance(result.evidence_window, dict)
        assert isinstance(result.meta, dict)
        assert result.model in ("baseline", "transformer", "ensemble")

    def test_stream_key_format(self):
        eng = self._make_engine(window_size=5, stride=5)
        result = None
        for ev in _make_events(5, service="hdfs", session="sess1"):
            result = eng.ingest(ev)
        assert result is not None
        assert result.stream_key == "hdfs:sess1"

    def test_fallback_score_is_anomaly(self, tmp_path):
        """
        With demo_mode=True, fallback_score=2.0, and NO model artifacts,
        the fallback scorer returns 2.0 >> threshold -> is_anomaly=True.
        A real baseline.pkl (if present) would override this; we ensure
        the empty tmp_path root forces the fallback path.
        """
        eng = InferenceEngine(
            mode="baseline",
            window_size=5,
            stride=5,
            root=tmp_path,  # empty dir -> no models found -> fallback used
        )
        eng.demo_mode = True
        eng.fallback_score = 2.0
        result = None
        for ev in _make_events(5):
            result = eng.ingest(ev)
        assert result is not None
        # fallback_score=2.0 >> default threshold 0.33 -> is_anomaly must be True
        assert result.is_anomaly is True

    def test_evidence_window_keys(self):
        eng = self._make_engine(window_size=5, stride=5)
        result = None
        for ev in _make_events(5):
            result = eng.ingest(ev)
        assert result is not None
        ew = result.evidence_window
        for key in ("tokens", "template_ids", "templates_preview",
                    "window_start_ts", "window_end_ts"):
            assert key in ew, f"Missing evidence_window key: {key}"

    def test_meta_contains_window_size(self):
        W = 8
        eng = self._make_engine(window_size=W, stride=W)
        result = None
        for ev in _make_events(W):
            result = eng.ingest(ev)
        assert result is not None
        assert result.meta.get("window_size") == W

    def test_result_serialisable(self):
        eng = self._make_engine(window_size=5, stride=5)
        result = None
        for ev in _make_events(5):
            result = eng.ingest(ev)
        assert result is not None
        d = result.to_dict()
        text = json.dumps(d, default=str)
        parsed = json.loads(text)
        assert parsed["model"] == "baseline"

    def test_multiple_stream_keys_independent(self):
        """Two different services should not interfere with each other."""
        eng = self._make_engine(window_size=5, stride=5)
        # Feed 4 events to "svc_a"
        for ev in _make_events(4, service="svc_a", session="a"):
            eng.ingest(ev)
        # Feed 5 events to "svc_b" — svc_b should emit, svc_a should not
        result_b = None
        for ev in _make_events(5, service="svc_b", session="b"):
            result_b = eng.ingest(ev)
        assert result_b is not None
        assert result_b.stream_key == "svc_b:b"

    def test_stride_controls_emission_frequency(self):
        W, S = 10, 3
        eng = self._make_engine(window_size=W, stride=S)
        emitted = []
        # Events 0..19: W + 3*S = 10 + 9 = 19 events -> should emit at 10, 13, 16, 19
        for ev in _make_events(W + S * 3):
            r = eng.ingest(ev)
            if r is not None:
                emitted.append(r)
        assert len(emitted) == 4


# ---------------------------------------------------------------------------
# Section 3: Alert integration with RiskResult
# ---------------------------------------------------------------------------

class TestAlertIntegrationFromRiskResult:
    """Validate that RiskResult objects flow correctly into the AlertManager."""

    def test_anomalous_result_fires_alert(self):
        policy = AlertPolicy(cooldown_seconds=0.0)
        manager = AlertManager(policy=policy)
        rr = _stub_risk_result(score=2.0, is_anomaly=True, threshold=1.0)
        alerts = manager.emit(rr)
        assert len(alerts) == 1

    def test_normal_result_no_alert(self):
        policy = AlertPolicy(cooldown_seconds=0.0)
        manager = AlertManager(policy=policy)
        rr = _stub_risk_result(score=0.2, is_anomaly=False, threshold=1.0)
        alerts = manager.emit(rr)
        assert len(alerts) == 0

    def test_alert_dict_has_required_fields(self):
        policy = AlertPolicy(cooldown_seconds=0.0)
        manager = AlertManager(policy=policy)
        rr = _stub_risk_result(score=2.0, is_anomaly=True, threshold=1.0)
        alerts = manager.emit(rr)
        assert alerts
        d = alerts[0].to_dict()
        for field in ("alert_id", "severity", "service", "score",
                      "timestamp", "model_name", "threshold"):
            assert field in d

    def test_severity_is_valid_label(self):
        policy = AlertPolicy(cooldown_seconds=0.0)
        manager = AlertManager(policy=policy)
        rr = _stub_risk_result(score=2.0, is_anomaly=True, threshold=1.0)
        alerts = manager.emit(rr)
        assert alerts[0].severity in ("critical", "high", "medium", "low", "info")

    def test_alert_id_is_unique_across_emissions(self):
        policy = AlertPolicy(cooldown_seconds=0.0)
        manager = AlertManager(policy=policy)
        ids = set()
        for i in range(5):
            rr = _stub_risk_result(
                stream_key=f"svc:{i}", score=2.0, is_anomaly=True, threshold=1.0
            )
            alerts = manager.emit(rr)
            if alerts:
                ids.add(alerts[0].alert_id)
        assert len(ids) == 5  # all UUIDs must be distinct
