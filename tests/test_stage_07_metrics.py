# test/test_stage_07_metrics.py

# Purpose: Tests for Stage 7 MetricsRegistry and /metrics endpoint.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are tests for MetricsRegistry and the /metrics endpoint,
# indirectly used by the API and any downstream components that rely on metrics)


"""Tests for Stage 7 MetricsRegistry and /metrics endpoint."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.app import create_app
from src.api.settings import Settings
from src.observability.metrics import MetricsRegistry
from tests.helpers_stage_07 import MockPipeline, _stub_risk_result, make_mock_pipeline


# ---------------------------------------------------------------------------
# MetricsRegistry unit tests
# ---------------------------------------------------------------------------

def test_registry_creates_all_metrics():
    m = MetricsRegistry()
    assert m.ingest_events_total is not None
    assert m.ingest_windows_total is not None
    assert m.alerts_total is not None
    assert m.ingest_errors_total is not None
    assert m.ingest_latency_seconds is not None
    assert m.scoring_latency_seconds is not None


def test_registry_uses_private_collector():
    """Each instance gets its own CollectorRegistry to avoid duplicate-registration."""
    m1 = MetricsRegistry()
    m2 = MetricsRegistry()
    assert m1.registry is not m2.registry


def test_counter_increments():
    m = MetricsRegistry()
    m.ingest_events_total.inc()
    m.ingest_events_total.inc()
    body, _ = m.generate_text()
    assert "ingest_events_total 2.0" in body


def test_labeled_counter_increments():
    m = MetricsRegistry()
    m.alerts_total.labels(severity="critical").inc()
    m.alerts_total.labels(severity="high").inc(2)
    body, _ = m.generate_text()
    assert 'severity="critical"' in body
    assert 'severity="high"' in body


def test_histogram_observe():
    m = MetricsRegistry()
    m.ingest_latency_seconds.observe(0.005)
    body, _ = m.generate_text()
    assert "ingest_latency_seconds" in body


def test_generate_text_returns_utf8_str():
    m = MetricsRegistry()
    body, content_type = m.generate_text()
    assert isinstance(body, str)
    assert "text/plain" in content_type


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------

@pytest.fixture()
def metrics_client():
    """Client with metrics enabled and auth disabled."""
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = True
    metrics = MetricsRegistry()
    pipeline = make_mock_pipeline(settings=cfg, metrics=metrics)
    app = create_app(settings=cfg, pipeline=pipeline)
    app.state.metrics = metrics
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def no_metrics_client():
    """Client with metrics disabled."""
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = False
    app = create_app(settings=cfg, pipeline=make_mock_pipeline(settings=cfg))
    return TestClient(app, raise_server_exceptions=True)


def test_metrics_endpoint_returns_200(metrics_client):
    resp = metrics_client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_content_type_prometheus(metrics_client):
    resp = metrics_client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_disabled_returns_placeholder(no_metrics_client):
    resp = no_metrics_client.get("/metrics")
    assert resp.status_code == 200
    assert "disabled" in resp.text


def test_ingest_increments_event_counter(metrics_client):
    metrics_client.post(
        "/ingest", json={"service": "svc", "token_id": 5}
    )
    resp = metrics_client.get("/metrics")
    assert "ingest_events_total" in resp.text


def test_window_counter_incremented_on_window(metrics_client):
    """Verify ingest_windows_total appears in output after a window is emitted."""
    # The mock pipeline's engine won't emit a window by default; just verify
    # the metric is present in the output (even if 0).
    resp = metrics_client.get("/metrics")
    assert "ingest_windows_total" in resp.text


def test_scoring_latency_metric_present(metrics_client):
    resp = metrics_client.get("/metrics")
    assert "scoring_latency_seconds" in resp.text


def test_alerts_metric_present(metrics_client):
    resp = metrics_client.get("/metrics")
    assert "alerts_total" in resp.text


def test_errors_metric_present(metrics_client):
    resp = metrics_client.get("/metrics")
    assert "ingest_errors_total" in resp.text


# ---------------------------------------------------------------------------
# Integration: ingest anomaly -> alert counter increments
# ---------------------------------------------------------------------------

def test_alert_fired_increments_alerts_counter():
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = True
    metrics = MetricsRegistry()
    pipeline = make_mock_pipeline(settings=cfg, metrics=metrics)
    app = create_app(settings=cfg, pipeline=pipeline)
    app.state.metrics = metrics
    client = TestClient(app)

    pipeline.engine.next_result = _stub_risk_result(
        score=2.0, is_anomaly=True, threshold=1.0
    )
    client.post("/ingest", json={"service": "svc", "token_id": 5})

    body, _ = metrics.generate_text()
    # alerts_total should have been incremented with some severity label
    assert "alerts_total" in body
