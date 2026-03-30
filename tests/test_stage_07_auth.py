# test/test_stage_07_auth.py

# Purpose: Tests for Stage 7 AuthMiddleware (X-API-Key).

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are tests for AuthMiddleware, indirectly used by the API endpoints 
# and any downstream components that rely on the API)


"""Tests for Stage 7 AuthMiddleware (X-API-Key)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.settings import Settings
from src.api.app import create_app
from tests.helpers_stage_07 import make_mock_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def authed_client():
    """App with auth enabled and API key = 'secret'."""
    cfg = Settings()
    cfg.api_key = "secret"
    cfg.disable_auth = False
    cfg.metrics_enabled = False
    app = create_app(settings=cfg, pipeline=make_mock_pipeline())
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def noauth_client():
    """App with auth disabled."""
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = False
    app = create_app(settings=cfg, pipeline=make_mock_pipeline())
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Protected endpoint requires valid key
# ---------------------------------------------------------------------------

def test_missing_key_returns_401(authed_client):
    resp = authed_client.post(
        "/ingest",
        json={"service": "svc", "token_id": 5},
    )
    assert resp.status_code == 401


def test_wrong_key_returns_401(authed_client):
    resp = authed_client.post(
        "/ingest",
        json={"service": "svc", "token_id": 5},
        headers={"x-api-key": "wrong"},
    )
    assert resp.status_code == 401


def test_correct_key_passes(authed_client):
    resp = authed_client.post(
        "/ingest",
        json={"service": "svc", "token_id": 5},
        headers={"x-api-key": "secret"},
    )
    assert resp.status_code == 200


def test_alerts_requires_key(authed_client):
    resp = authed_client.get("/alerts")
    assert resp.status_code == 401


def test_alerts_passes_with_key(authed_client):
    resp = authed_client.get("/alerts", headers={"x-api-key": "secret"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Public endpoints skip auth
# ---------------------------------------------------------------------------

def test_health_no_key_required(authed_client):
    resp = authed_client.get("/health")
    assert resp.status_code == 200


def test_metrics_no_key_required(authed_client):
    resp = authed_client.get("/metrics")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Disabled auth lets everything through
# ---------------------------------------------------------------------------

def test_disabled_auth_no_key_needed(noauth_client):
    resp = noauth_client.post(
        "/ingest",
        json={"service": "svc", "token_id": 5},
    )
    assert resp.status_code == 200


def test_disabled_auth_alerts_open(noauth_client):
    resp = noauth_client.get("/alerts")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Header case-insensitive
# ---------------------------------------------------------------------------

def test_header_case_insensitive(authed_client):
    """HTTP headers are case-insensitive; both forms must work."""
    for header_name in ("x-api-key", "X-Api-Key", "X-API-KEY"):
        resp = authed_client.post(
            "/ingest",
            json={"service": "svc", "token_id": 5},
            headers={header_name: "secret"},
        )
        assert resp.status_code == 200, f"Failed with header {header_name!r}"


# ---------------------------------------------------------------------------
# No key configured -> pass-through with warning
# ---------------------------------------------------------------------------

def test_no_key_configured_allows_any(monkeypatch):
    """When API_KEY is empty the middleware should pass every request."""
    monkeypatch.delenv("API_KEY", raising=False)
    cfg = Settings()
    cfg.api_key = ""
    cfg.disable_auth = False
    cfg.metrics_enabled = False
    app = create_app(settings=cfg, pipeline=make_mock_pipeline())
    client = TestClient(app)
    resp = client.post("/ingest", json={"service": "svc", "token_id": 5})
    assert resp.status_code == 200
