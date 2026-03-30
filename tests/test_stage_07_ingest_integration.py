# test/test_stage_07_ingest_integration.py

# Purpose: Tests for Stage 7 POST /ingest and GET /alerts integration.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are tests for the ingest and alerts endpoints,
# indirectly used by the API and any downstream components that rely on these endpoints)

"""Tests for Stage 7 POST /ingest and GET /alerts integration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.app import create_app
from src.api.settings import Settings
from tests.helpers_stage_07 import MockPipeline, _stub_risk_result, make_mock_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Unauthenticated client (auth disabled, metrics off)."""
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = False
    app = create_app(settings=cfg, pipeline=make_mock_pipeline(settings=cfg))
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def pipeline() -> MockPipeline:
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = False
    return make_mock_pipeline(settings=cfg)


@pytest.fixture()
def client_with_pipeline(pipeline):
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = False
    app = create_app(settings=cfg, pipeline=pipeline)
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# POST /ingest — no window emitted
# ---------------------------------------------------------------------------

def test_ingest_no_window_returns_200(client):
    resp = client.post("/ingest", json={"service": "hdfs", "token_id": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_emitted"] is False
    assert body["risk_result"] is None
    assert body["alert"] is None


def test_ingest_missing_service_returns_422(client):
    """token_id missing should fail validation."""
    resp = client.post("/ingest", json={"service": "hdfs"})
    assert resp.status_code == 422


def test_ingest_missing_token_id_returns_422(client):
    resp = client.post("/ingest", json={"token_id": 5})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /ingest — window emitted, no anomaly
# ---------------------------------------------------------------------------

def test_ingest_with_window_no_anomaly(client_with_pipeline, pipeline):
    pipeline.engine.next_result = _stub_risk_result(
        score=0.4, is_anomaly=False, threshold=1.0
    )
    resp = client_with_pipeline.post(
        "/ingest", json={"service": "svc", "token_id": 5}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_emitted"] is True
    assert body["risk_result"] is not None
    assert body["risk_result"]["is_anomaly"] is False
    assert body["alert"] is None


def test_ingest_risk_result_fields(client_with_pipeline, pipeline):
    pipeline.engine.next_result = _stub_risk_result(
        stream_key="hdfs:", score=0.6, is_anomaly=False
    )
    resp = client_with_pipeline.post(
        "/ingest", json={"service": "hdfs", "token_id": 10}
    )
    rr = resp.json()["risk_result"]
    assert rr["stream_key"] == "hdfs:"
    assert rr["risk_score"] == 0.6
    assert rr["model"] == "ensemble"
    assert "evidence_window" in rr


# ---------------------------------------------------------------------------
# POST /ingest — anomaly triggers alert
# ---------------------------------------------------------------------------

def test_ingest_anomaly_fires_alert(client_with_pipeline, pipeline):
    pipeline.engine.next_result = _stub_risk_result(
        score=2.0, is_anomaly=True, threshold=1.0
    )
    resp = client_with_pipeline.post(
        "/ingest", json={"service": "auth", "token_id": 20}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_emitted"] is True
    assert body["alert"] is not None
    alert = body["alert"]
    assert "alert_id" in alert
    assert alert["severity"] in ("critical", "high", "medium", "low")


def test_ingest_alert_severity_critical(client_with_pipeline, pipeline):
    """score=3.0 with threshold=1.0 -> ratio=3.0 -> critical (>=1.5x)."""
    pipeline.engine.next_result = _stub_risk_result(
        score=3.0, is_anomaly=True, threshold=1.0
    )
    resp = client_with_pipeline.post(
        "/ingest", json={"service": "billing", "token_id": 7}
    )
    alert = resp.json()["alert"]
    assert alert["severity"] == "critical"


def test_ingest_alert_stored_in_buffer(client_with_pipeline, pipeline):
    """Fired alert must appear in GET /alerts."""
    pipeline.engine.next_result = _stub_risk_result(
        score=2.0, is_anomaly=True, threshold=1.0
    )
    client_with_pipeline.post(
        "/ingest", json={"service": "svc", "token_id": 5}
    )
    resp = client_with_pipeline.get("/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1


# ---------------------------------------------------------------------------
# GET /alerts
# ---------------------------------------------------------------------------

def test_alerts_empty_on_startup(client):
    resp = client.get("/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["alerts"] == []


def test_alerts_response_schema(client_with_pipeline, pipeline):
    pipeline.engine.next_result = _stub_risk_result(
        score=2.0, is_anomaly=True, threshold=1.0
    )
    client_with_pipeline.post(
        "/ingest", json={"service": "svc", "token_id": 5}
    )
    resp = client_with_pipeline.get("/alerts")
    alerts = resp.json()["alerts"]
    if alerts:
        a = alerts[0]
        for key in ("alert_id", "severity", "service", "score",
                    "timestamp", "evidence_window", "model_name", "threshold"):
            assert key in a, f"Missing key: {key}"


def test_multiple_alerts_accumulate(client_with_pipeline, pipeline):
    for _ in range(3):
        pipeline.engine.next_result = _stub_risk_result(
            score=2.0, is_anomaly=True, threshold=1.0
        )
        client_with_pipeline.post(
            "/ingest", json={"service": "svc", "token_id": 5}
        )
    resp = client_with_pipeline.get("/alerts")
    assert resp.json()["count"] >= 3


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_status_field(client):
    body = client.get("/health").json()
    assert "status" in body


def test_health_mock_pipeline_is_healthy(client):
    """MockPipeline sets _artifacts_loaded=True, so health should be healthy."""
    body = client.get("/health").json()
    assert body["status"] == "healthy"
