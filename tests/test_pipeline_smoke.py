# tests/test_pipeline_smoke.py

# Purpose: Always-fast pipeline smoke test (Part A, Stage 7.2).
#          No model files, no data downloads, no external services required.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are smoke tests for the ingest pipeline, indirectly used by the pipeline itself and its outputs)

"""Stage 7.2 — Always-fast pipeline smoke test."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.app import create_app
from src.api.settings import Settings
from src.synthetic import (
    MemoryLeakPattern,
    NetworkFlapPattern,
    ScenarioBuilder,
    SyntheticLogGenerator,
)
from tests.helpers_stage_07 import MockPipeline, _stub_risk_result

_BASE_TS = 1_704_067_200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_events(n: int = 50) -> list[dict]:
    """Generate n synthetic events with no external dependencies."""
    n_actual = max(n, 2)  # ScenarioBuilder requires n_events >= 1 per scenario
    gen = SyntheticLogGenerator([MemoryLeakPattern(), NetworkFlapPattern()], seed=42)
    builder = ScenarioBuilder()
    scenario = builder.build_scenario(
        scenario_id="smoke",
        service="bgl",
        host="host-01",
        start_ts=_BASE_TS,
        n_events=n_actual,
        pattern_name="memory_leak",
    )
    raw = gen.generate(n_actual, scenario)[:n]
    return [
        {
            "service": ev.service,
            "token_id": abs(hash(ev.message)) % 7833 + 2,
            "session_id": f"{ev.service}-smoke",
            "timestamp": float(ev.timestamp or 0),
            "label": int(ev.label or 0),
        }
        for ev in raw
    ]


@pytest.fixture(scope="module")
def smoke_api():
    """
    In-process FastAPI app with MockPipeline.
    MockPipeline's engine is pre-loaded with an anomalous result so that
    any ingest call that triggers a window also fires an alert.
    """
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = False
    cfg.window_size = 5
    cfg.stride = 1
    cfg.alert_cooldown_seconds = 0.0
    pipeline = MockPipeline(settings=cfg)
    app = create_app(settings=cfg, pipeline=pipeline)
    client = TestClient(app, raise_server_exceptions=True)
    return client, pipeline


# ---------------------------------------------------------------------------
# Part 1 — Synthetic event generation (no API needed)
# ---------------------------------------------------------------------------

class TestSyntheticGeneration:
    """Verify synthetic log events are generated with the correct schema."""

    def test_correct_count(self):
        events = _generate_events(50)
        assert len(events) == 50

    def test_required_keys(self):
        for ev in _generate_events(10):
            assert "service" in ev
            assert "token_id" in ev
            assert "session_id" in ev

    def test_token_ids_positive_ints(self):
        for ev in _generate_events(20):
            assert isinstance(ev["token_id"], int) and ev["token_id"] > 0

    def test_services_non_empty(self):
        services = {ev["service"] for ev in _generate_events(20)}
        assert len(services) >= 1


# ---------------------------------------------------------------------------
# Part 2 — Ingest pipeline: prepare -> sequences -> minimal inference
# ---------------------------------------------------------------------------

class TestIngestPipeline:
    """Verify POST /ingest works end-to-end with synthetic events."""

    def test_single_ingest_returns_200(self, smoke_api):
        client, _ = smoke_api
        ev = _generate_events(1)[0]
        assert client.post("/ingest", json=ev).status_code == 200

    def test_ingest_response_schema(self, smoke_api):
        client, _ = smoke_api
        body = client.post("/ingest", json=_generate_events(1)[0]).json()
        assert "window_emitted" in body
        assert "risk_result" in body
        assert "alert" in body

    def test_anomaly_event_fires_alert(self, smoke_api):
        client, pipeline = smoke_api
        pipeline.engine.next_result = _stub_risk_result(
            score=3.0, is_anomaly=True, threshold=1.0
        )
        body = client.post("/ingest", json=_generate_events(1)[0]).json()
        assert body["window_emitted"] is True
        assert body["alert"] is not None
        assert body["alert"]["severity"] in ("critical", "high", "medium", "low")

    def test_50_events_all_return_200(self, smoke_api):
        """Batch ingest of 50 synthetic events; all must succeed."""
        client, _ = smoke_api
        events = _generate_events(50)
        statuses = [client.post("/ingest", json=ev).status_code for ev in events]
        assert all(s == 200 for s in statuses), (
            f"Non-200 responses: {set(statuses) - {200}}"
        )

    def test_ingest_then_alert_in_buffer(self, smoke_api):
        """After firing an anomaly, GET /alerts must return >= 1 alert."""
        client, pipeline = smoke_api
        pipeline.engine.next_result = _stub_risk_result(
            score=2.5, is_anomaly=True, threshold=1.0
        )
        client.post("/ingest", json=_generate_events(1)[0])
        data = client.get("/alerts").json()
        assert data["count"] >= 1


# ---------------------------------------------------------------------------
# Part 3 — Alerts endpoint
# ---------------------------------------------------------------------------

class TestAlertsEndpoint:
    def test_returns_200(self, smoke_api):
        client, _ = smoke_api
        assert client.get("/alerts").status_code == 200

    def test_response_has_count_and_alerts(self, smoke_api):
        client, _ = smoke_api
        body = client.get("/alerts").json()
        assert "count" in body
        assert "alerts" in body
        assert isinstance(body["alerts"], list)


# ---------------------------------------------------------------------------
# Part 4 — RAG /query endpoint
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_returns_200(self, smoke_api):
        client, _ = smoke_api
        resp = client.post("/query", json={"question": "How do alerts work?"})
        assert resp.status_code == 200

    def test_response_has_answer(self, smoke_api):
        client, _ = smoke_api
        body = client.post("/query", json={"question": "What model is used?"}).json()
        assert "answer" in body
        assert isinstance(body["answer"], str) and len(body["answer"]) > 0

    def test_response_has_sources(self, smoke_api):
        client, _ = smoke_api
        body = client.post("/query", json={"question": "Tell me about the dataset"}).json()
        assert "sources" in body
        assert isinstance(body["sources"], list) and len(body["sources"]) > 0

    def test_sources_have_required_fields(self, smoke_api):
        client, _ = smoke_api
        body = client.post("/query", json={"question": "docker threshold"}).json()
        for src in body["sources"]:
            assert "id" in src
            assert "score" in src
            assert "snippet" in src


# ---------------------------------------------------------------------------
# Part 5 — Demo UI page
# ---------------------------------------------------------------------------

class TestDemoUI:
    def test_ui_returns_200(self, smoke_api):
        client, _ = smoke_api
        assert client.get("/").status_code == 200

    def test_ui_is_html(self, smoke_api):
        client, _ = smoke_api
        resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")

    def test_ui_has_all_panels(self, smoke_api):
        client, _ = smoke_api
        html = client.get("/").text
        for section_id in ("section-dashboard", "section-alerts",
                           "section-investigation", "section-health", "section-metrics"):
            assert section_id in html, f"Missing section: {section_id}"
