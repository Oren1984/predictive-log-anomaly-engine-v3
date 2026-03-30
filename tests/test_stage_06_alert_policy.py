# test/test_stage_06_alert_policy.py

# Purpose: Tests for AlertPolicy: severity classification, should_alert, risk_to_alert.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are tests for AlertPolicy, 
# indirectly used by the alerting pipeline 
# and any downstream components that consume alerts)

"""Tests for AlertPolicy: severity classification, should_alert, risk_to_alert."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from typing import Optional

from src.alerts import Alert, AlertPolicy


# ---------------------------------------------------------------------------
# Minimal RiskResult stub (no src.runtime import needed)
# ---------------------------------------------------------------------------

@dataclass
class _FakeRiskResult:
    stream_key: str = "auth:"
    timestamp: float = 1_704_067_200.0
    model: str = "ensemble"
    risk_score: float = 132.0
    is_anomaly: bool = True
    threshold: float = 1.0
    evidence_window: dict = field(default_factory=lambda: {
        "templates_preview": ["tid=5: memory_check heap=512MB", "tid=6: gc_runs=30"],
        "tokens": list(range(50)),
        "window_start_ts": 1_704_067_150.0,
        "window_end_ts": 1_704_067_200.0,
    })
    top_predictions: Optional[list] = None
    meta: dict = field(default_factory=lambda: {"window_size": 50, "emit_index": 5})


# ---------------------------------------------------------------------------
# should_alert
# ---------------------------------------------------------------------------

def test_should_alert_anomaly_true():
    policy = AlertPolicy()
    rr = _FakeRiskResult(is_anomaly=True)
    assert policy.should_alert(rr) is True


def test_should_alert_anomaly_false():
    policy = AlertPolicy()
    rr = _FakeRiskResult(is_anomaly=False)
    assert policy.should_alert(rr) is False


def test_should_alert_additional_threshold_filter():
    """Policy.threshold > 0 adds an extra score gate beyond is_anomaly."""
    policy = AlertPolicy(threshold=200.0)
    rr = _FakeRiskResult(is_anomaly=True, risk_score=132.0)
    assert policy.should_alert(rr) is False  # 132 < 200 -> filtered


def test_should_alert_additional_threshold_passes():
    policy = AlertPolicy(threshold=100.0)
    rr = _FakeRiskResult(is_anomaly=True, risk_score=132.0)
    assert policy.should_alert(rr) is True  # 132 >= 100


def test_should_alert_threshold_zero_disabled():
    """threshold=0 means no extra filtering — trust is_anomaly."""
    policy = AlertPolicy(threshold=0.0)
    rr = _FakeRiskResult(is_anomaly=True, risk_score=0.001)
    assert policy.should_alert(rr) is True


# ---------------------------------------------------------------------------
# classify_severity
# ---------------------------------------------------------------------------

def test_classify_severity_critical():
    policy = AlertPolicy()
    # 1.5x threshold -> critical
    sev = policy.classify_severity(score=1.5, threshold=1.0)
    assert sev == "critical"


def test_classify_severity_high():
    policy = AlertPolicy()
    sev = policy.classify_severity(score=1.2, threshold=1.0)
    assert sev == "high"


def test_classify_severity_medium():
    policy = AlertPolicy()
    sev = policy.classify_severity(score=1.0, threshold=1.0)
    assert sev == "medium"


def test_classify_severity_low_below_medium():
    policy = AlertPolicy()
    sev = policy.classify_severity(score=0.5, threshold=1.0)
    assert sev == "low"


def test_classify_severity_ensemble_scale():
    """Ensemble scores are ~130 with threshold=1.0 -> critical."""
    policy = AlertPolicy()
    sev = policy.classify_severity(score=132.0, threshold=1.0)
    assert sev == "critical"


def test_classify_severity_baseline_scale():
    """Baseline score ~0.5 with threshold=0.33 -> ratio ~1.5 -> critical."""
    policy = AlertPolicy()
    sev = policy.classify_severity(score=0.50, threshold=0.33)
    assert sev == "critical"


def test_classify_severity_custom_buckets():
    policy = AlertPolicy(severity_buckets={"urgent": 2.0, "normal": 1.0})
    assert policy.classify_severity(score=2.0, threshold=1.0) == "urgent"
    assert policy.classify_severity(score=1.5, threshold=1.0) == "normal"
    assert policy.classify_severity(score=0.5, threshold=1.0) == "low"


# ---------------------------------------------------------------------------
# risk_to_alert
# ---------------------------------------------------------------------------

def test_risk_to_alert_returns_alert():
    policy = AlertPolicy()
    rr = _FakeRiskResult()
    alert = policy.risk_to_alert(rr)
    assert isinstance(alert, Alert)


def test_risk_to_alert_service_extracted():
    policy = AlertPolicy()
    rr = _FakeRiskResult(stream_key="billing:session-42")
    alert = policy.risk_to_alert(rr)
    assert alert.service == "billing"


def test_risk_to_alert_service_no_colon():
    policy = AlertPolicy()
    rr = _FakeRiskResult(stream_key="authonly")
    alert = policy.risk_to_alert(rr)
    assert alert.service == "authonly"


def test_risk_to_alert_fields_match():
    policy = AlertPolicy()
    rr = _FakeRiskResult(
        risk_score=132.5,
        threshold=1.0,
        model="ensemble",
        timestamp=1_704_067_200.0,
    )
    alert = policy.risk_to_alert(rr)
    assert alert.score == 132.5
    assert alert.threshold == 1.0
    assert alert.model_name == "ensemble"
    assert alert.timestamp == 1_704_067_200.0


def test_risk_to_alert_evidence_trimmed():
    """evidence_window includes templates_preview capped at 5 items."""
    policy = AlertPolicy()
    long_preview = [f"tid={i}: template" for i in range(10)]
    rr = _FakeRiskResult(evidence_window={
        "templates_preview": long_preview,
        "tokens": list(range(50)),
        "window_start_ts": 0.0,
        "window_end_ts": 100.0,
    })
    alert = policy.risk_to_alert(rr)
    assert len(alert.evidence_window["templates_preview"]) <= 5


def test_risk_to_alert_unique_alert_ids():
    """Two calls with identical input produce different alert_ids."""
    policy = AlertPolicy()
    rr = _FakeRiskResult()
    a1 = policy.risk_to_alert(rr)
    a2 = policy.risk_to_alert(rr)
    assert a1.alert_id != a2.alert_id


def test_risk_to_alert_timestamp_string():
    """String timestamp is converted to float."""
    policy = AlertPolicy()
    rr = _FakeRiskResult(timestamp="1704067200.0")
    alert = policy.risk_to_alert(rr)
    assert isinstance(alert.timestamp, float)


def test_alert_to_dict_keys():
    policy = AlertPolicy()
    rr = _FakeRiskResult()
    alert = policy.risk_to_alert(rr)
    d = alert.to_dict()
    required = {"alert_id", "severity", "service", "score", "timestamp",
                "evidence_window", "model_name", "threshold", "meta"}
    assert required.issubset(d.keys())
