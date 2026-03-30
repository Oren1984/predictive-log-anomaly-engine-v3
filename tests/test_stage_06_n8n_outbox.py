# test/test_stage_06_n8n_outbox.py

# Purpose: Tests for N8nWebhookClient: DRY_RUN writes files, payload integrity.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are tests for N8nWebhookClient,
# indirectly used by the alerting pipeline and any downstream components that consume alerts)


"""Tests for N8nWebhookClient: DRY_RUN writes files, payload integrity."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alerts import Alert, AlertPolicy, N8nWebhookClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(
    alert_id: str = "test-alert-001",
    severity: str = "high",
    service: str = "auth",
    score: float = 132.0,
) -> Alert:
    return Alert(
        alert_id=alert_id,
        severity=severity,
        service=service,
        score=score,
        timestamp=1_704_067_200.0,
        evidence_window={
            "templates_preview": ["tid=5: memory_check heap=512MB"],
            "token_count": 50,
            "start_ts": 1_704_067_150.0,
            "end_ts": 1_704_067_200.0,
        },
        model_name="ensemble",
        threshold=1.0,
        meta={"stream_key": "auth:", "is_anomaly": True},
    )


# ---------------------------------------------------------------------------
# DRY_RUN writes files
# ---------------------------------------------------------------------------

def test_dry_run_creates_file():
    """DRY_RUN mode must write <alert_id>.json to outbox_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)
        alert = _make_alert(alert_id="abc-001")

        result = client.send(alert)

        assert result["status"] == "dry_run"
        expected_file = outbox / "abc-001.json"
        assert expected_file.exists(), f"Expected {expected_file} to exist"


def test_dry_run_file_non_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)
        alert = _make_alert(alert_id="abc-002")
        client.send(alert)

        content = (outbox / "abc-002.json").read_text(encoding="utf-8")
        assert len(content) > 0


def test_dry_run_payload_valid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)
        alert = _make_alert(alert_id="abc-003")
        client.send(alert)

        payload = json.loads((outbox / "abc-003.json").read_text(encoding="utf-8"))
        assert isinstance(payload, dict)


def test_dry_run_payload_has_required_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)
        alert = _make_alert(alert_id="abc-004")
        client.send(alert)

        payload = json.loads((outbox / "abc-004.json").read_text(encoding="utf-8"))
        required = {"alert_id", "severity", "service", "score", "timestamp",
                    "evidence_window", "model_name", "threshold", "meta"}
        assert required.issubset(payload.keys())


def test_dry_run_payload_values_match_alert():
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)
        alert = _make_alert(
            alert_id="abc-005",
            severity="critical",
            service="billing",
            score=200.0,
        )
        client.send(alert)

        payload = json.loads((outbox / "abc-005.json").read_text(encoding="utf-8"))
        assert payload["alert_id"] == "abc-005"
        assert payload["severity"] == "critical"
        assert payload["service"] == "billing"
        assert payload["score"] == 200.0


def test_dry_run_result_contains_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)
        alert = _make_alert(alert_id="abc-006")
        result = client.send(alert)

        assert "path" in result
        assert "abc-006" in result["path"]


def test_dry_run_multiple_alerts_separate_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)

        for i in range(5):
            alert = _make_alert(alert_id=f"alert-{i:03d}")
            client.send(alert)

        files = list(outbox.glob("*.json"))
        assert len(files) == 5


# ---------------------------------------------------------------------------
# No URL -> always dry-runs even if dry_run=False
# ---------------------------------------------------------------------------

def test_no_url_forces_dry_run():
    """Without a webhook URL, even dry_run=False should write to outbox."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "outbox"
        client = N8nWebhookClient(webhook_url="", dry_run=False, outbox_dir=outbox)
        alert = _make_alert(alert_id="no-url-001")
        result = client.send(alert)
        # Should fall back to dry_run outbox (status may be "dry_run")
        assert (outbox / "no-url-001.json").exists()


# ---------------------------------------------------------------------------
# Env var configuration
# ---------------------------------------------------------------------------

def test_env_dry_run_default_true(monkeypatch):
    """Default N8N_DRY_RUN should be true when env var is absent."""
    monkeypatch.delenv("N8N_DRY_RUN", raising=False)
    monkeypatch.delenv("N8N_WEBHOOK_URL", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        client = N8nWebhookClient(outbox_dir=Path(tmpdir) / "outbox")
        assert client.dry_run is True


def test_env_dry_run_false_explicit(monkeypatch):
    """N8N_DRY_RUN=false sets dry_run=False (live mode attempted)."""
    monkeypatch.setenv("N8N_DRY_RUN", "false")
    client = N8nWebhookClient(webhook_url="http://localhost:5678/webhook/test")
    assert client.dry_run is False


def test_env_webhook_url_read(monkeypatch):
    monkeypatch.setenv("N8N_WEBHOOK_URL", "http://n8n.internal/hook/abc")
    client = N8nWebhookClient()
    assert client.webhook_url == "http://n8n.internal/hook/abc"


def test_constructor_args_override_env(monkeypatch):
    monkeypatch.setenv("N8N_DRY_RUN", "false")
    monkeypatch.setenv("N8N_WEBHOOK_URL", "http://should-be-ignored")
    client = N8nWebhookClient(webhook_url="http://override", dry_run=True)
    assert client.dry_run is True
    assert client.webhook_url == "http://override"


# ---------------------------------------------------------------------------
# Outbox dir auto-created
# ---------------------------------------------------------------------------

def test_outbox_dir_created_automatically():
    with tempfile.TemporaryDirectory() as tmpdir:
        outbox = Path(tmpdir) / "deep" / "nested" / "outbox"
        client = N8nWebhookClient(dry_run=True, outbox_dir=outbox)
        alert = _make_alert(alert_id="nested-001")
        client.send(alert)
        assert outbox.exists()
        assert (outbox / "nested-001.json").exists()
