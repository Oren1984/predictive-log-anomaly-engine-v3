# tests/integration/test_smoke_api.py

# Purpose: Smoke tests for the API endpoints to ensure basic functionality.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are integration tests for the API)

"""
Stage 08 integration smoke tests.

These tests verify the full ingest -> RiskResult -> Alert pipeline path
using the FastAPI TestClient (no running server required).  The MockPipeline
from helpers_stage_07 is used so tests run without trained model files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.api.app import create_app
from src.api.settings import Settings
from tests.helpers_stage_07 import MockPipeline, _stub_risk_result


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def api():
    """Full app with auth disabled and a fresh MockPipeline."""
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = True
    pipeline = MockPipeline(settings=cfg)
    app = create_app(settings=cfg, pipeline=pipeline)
    return TestClient(app, raise_server_exceptions=True), pipeline


# ---------------------------------------------------------------------------
# Health + metrics endpoints are reachable
# ---------------------------------------------------------------------------

def test_health_returns_200(api):
    client, _ = api
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body_has_status(api):
    client, _ = api
    body = client.get("/health").json()
    assert "status" in body


def test_metrics_returns_200(api):
    client, _ = api
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_body_contains_ingest_counter(api):
    client, _ = api
    body = client.get("/metrics").text
    assert "ingest_events_total" in body


# ---------------------------------------------------------------------------
# Ingest -> alert (core Stage 08 integration requirement)
# ---------------------------------------------------------------------------

def test_ingest_no_window_returns_200(api):
    """POST /ingest succeeds even when no window is emitted yet."""
    client, _ = api
    resp = client.post("/ingest", json={"service": "hdfs", "token_id": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_emitted"] is False
    assert body["alert"] is None


def test_ingest_anomaly_produces_alert(api):
    """
    Core integration path: ingest event -> engine scores window -> alert fired.

    The MockPipeline's engine is primed with an anomalous RiskResult so the
    single POST /ingest call exercises the full
      InferenceEngine.ingest()
        -> AlertManager.emit()
          -> Alert created
            -> stored in ring buffer
    path end-to-end.
    """
    client, pipeline = api
    pipeline.engine.next_result = _stub_risk_result(
        stream_key="hdfs:",
        score=2.5,
        is_anomaly=True,
        threshold=1.0,
    )
    resp = client.post("/ingest", json={"service": "hdfs", "token_id": 10})
    assert resp.status_code == 200
    body = resp.json()

    assert body["window_emitted"] is True, "Expected a scored window"
    assert body["risk_result"] is not None
    assert body["risk_result"]["is_anomaly"] is True
    assert body["alert"] is not None, "Expected an alert to be fired"


def test_fired_alert_has_required_fields(api):
    """Alert payload must have all required fields."""
    client, pipeline = api
    pipeline.engine.next_result = _stub_risk_result(
        score=2.5, is_anomaly=True, threshold=1.0
    )
    body = client.post(
        "/ingest", json={"service": "auth", "token_id": 7}
    ).json()
    alert = body["alert"]
    for field in ("alert_id", "severity", "service", "score",
                  "timestamp", "evidence_window", "model_name", "threshold"):
        assert field in alert, f"Missing field in alert: {field}"


def test_fired_alert_appears_in_get_alerts(api):
    """Alert stored in ring buffer must appear in GET /alerts."""
    client, pipeline = api
    pipeline.engine.next_result = _stub_risk_result(
        score=3.0, is_anomaly=True, threshold=1.0
    )
    client.post("/ingest", json={"service": "billing", "token_id": 5})
    alerts_resp = client.get("/alerts")
    assert alerts_resp.status_code == 200
    data = alerts_resp.json()
    assert data["count"] >= 1
    assert len(data["alerts"]) >= 1


def test_alert_severity_critical_for_high_score(api):
    """score 3.0 / threshold 1.0 = ratio 3.0 >= 1.5 -> critical."""
    client, pipeline = api
    pipeline.engine.next_result = _stub_risk_result(
        score=3.0, is_anomaly=True, threshold=1.0
    )
    body = client.post(
        "/ingest", json={"service": "svc", "token_id": 5}
    ).json()
    assert body["alert"]["severity"] == "critical"


def test_alert_counter_increments_in_metrics(api):
    """Firing an alert must increment alerts_total in Prometheus output."""
    client, pipeline = api
    pipeline.engine.next_result = _stub_risk_result(
        score=2.0, is_anomaly=True, threshold=1.0
    )
    client.post("/ingest", json={"service": "svc", "token_id": 5})
    metrics_body = client.get("/metrics").text
    assert "alerts_total" in metrics_body


def test_no_anomaly_no_alert(api):
    """A non-anomalous window must not produce an alert."""
    client, pipeline = api
    pipeline.engine.next_result = _stub_risk_result(
        score=0.3, is_anomaly=False, threshold=1.0
    )
    body = client.post(
        "/ingest", json={"service": "svc", "token_id": 5}
    ).json()
    assert body["window_emitted"] is True
    assert body["alert"] is None


# ---------------------------------------------------------------------------
# Stage 7.1 — Demo UI smoke tests
# ---------------------------------------------------------------------------

def test_ui_index_returns_200(api):
    """GET / must return the HTML demo page."""
    client, _ = api
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_ui_index_contains_expected_sections(api):
    """The HTML page must include the Phase 8 observability dashboard section identifiers."""
    client, _ = api
    body = client.get("/").text
    for section in ("section-dashboard", "section-alerts", "section-investigation",
                    "section-health", "section-metrics"):
        assert section in body, f"Missing section: {section}"


def test_query_returns_200(api):
    """POST /query must return HTTP 200."""
    client, _ = api
    resp = client.post("/query", json={"question": "How do alerts work?"})
    assert resp.status_code == 200


def test_query_response_has_answer_and_sources(api):
    """POST /query response must contain 'answer' (str) and 'sources' (list)."""
    client, _ = api
    data = client.post("/query", json={"question": "What model is used?"}).json()
    assert "answer" in data
    assert isinstance(data["answer"], str) and len(data["answer"]) > 0
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_query_sources_have_required_fields(api):
    """Each source document must have id, score, and snippet."""
    client, _ = api
    data = client.post("/query", json={"question": "Tell me about the dataset"}).json()
    for src in data["sources"]:
        assert "id" in src
        assert "score" in src
        assert "snippet" in src
