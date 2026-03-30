# test/test_stage_06_dedup_cooldown.py

# Purpose: Tests for AlertManager deduplication and cooldown logic.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are tests for AlertManager,
# indirectly used by the alerting pipeline)


"""Tests for AlertManager deduplication and cooldown logic."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from typing import Optional

from src.alerts import Alert, AlertManager, AlertPolicy


# ---------------------------------------------------------------------------
# Minimal RiskResult stub
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
        "templates_preview": [],
        "tokens": list(range(50)),
        "window_start_ts": 1_704_067_150.0,
        "window_end_ts": 1_704_067_200.0,
    })
    top_predictions: Optional[list] = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(cooldown: float = 60.0, clock_fn=None) -> AlertManager:
    policy = AlertPolicy(cooldown_seconds=cooldown)
    return AlertManager(policy=policy, clock_fn=clock_fn)


# ---------------------------------------------------------------------------
# Basic emission
# ---------------------------------------------------------------------------

def test_emit_anomaly_returns_alert():
    mgr = _make_manager()
    results = mgr.emit(_FakeRiskResult(is_anomaly=True))
    assert len(results) == 1
    assert isinstance(results[0], Alert)


def test_emit_non_anomaly_returns_empty():
    mgr = _make_manager()
    results = mgr.emit(_FakeRiskResult(is_anomaly=False))
    assert results == []


def test_emit_increments_alert_count():
    mgr = _make_manager()
    mgr.emit(_FakeRiskResult(stream_key="svc-a:"))
    mgr.emit(_FakeRiskResult(stream_key="svc-b:"))  # different key -> both fire
    assert mgr.alert_count == 2


# ---------------------------------------------------------------------------
# Cooldown suppression
# ---------------------------------------------------------------------------

def test_same_key_within_cooldown_suppressed():
    """Second emit for the same stream_key within cooldown is suppressed."""
    now = [0.0]

    def clock():
        return now[0]

    mgr = _make_manager(cooldown=60.0, clock_fn=clock)

    # First emit: should fire
    r1 = mgr.emit(_FakeRiskResult(stream_key="auth:"))
    assert len(r1) == 1

    # Advance time by 30s (within 60s cooldown)
    now[0] = 30.0
    r2 = mgr.emit(_FakeRiskResult(stream_key="auth:"))
    assert r2 == []
    assert mgr.suppressed_count == 1


def test_same_key_after_cooldown_fires_again():
    """Emit fires again once cooldown has elapsed."""
    now = [0.0]

    def clock():
        return now[0]

    mgr = _make_manager(cooldown=60.0, clock_fn=clock)

    mgr.emit(_FakeRiskResult(stream_key="auth:"))

    # Advance past cooldown
    now[0] = 61.0
    r2 = mgr.emit(_FakeRiskResult(stream_key="auth:"))
    assert len(r2) == 1
    assert mgr.suppressed_count == 0


def test_different_keys_both_fire():
    """Different stream keys have independent cooldowns."""
    now = [0.0]
    mgr = _make_manager(cooldown=60.0, clock_fn=lambda: now[0])

    mgr.emit(_FakeRiskResult(stream_key="auth:"))
    mgr.emit(_FakeRiskResult(stream_key="billing:"))  # different key

    assert mgr.alert_count == 2
    assert mgr.suppressed_count == 0


def test_suppression_counted_correctly():
    now = [0.0]
    mgr = _make_manager(cooldown=60.0, clock_fn=lambda: now[0])

    mgr.emit(_FakeRiskResult(stream_key="auth:"))
    now[0] = 10.0
    mgr.emit(_FakeRiskResult(stream_key="auth:"))  # suppressed
    now[0] = 20.0
    mgr.emit(_FakeRiskResult(stream_key="auth:"))  # suppressed
    now[0] = 61.0
    mgr.emit(_FakeRiskResult(stream_key="auth:"))  # fires again

    assert mgr.alert_count == 2
    assert mgr.suppressed_count == 2


# ---------------------------------------------------------------------------
# Zero-cooldown (fire every time)
# ---------------------------------------------------------------------------

def test_zero_cooldown_fires_every_time():
    mgr = _make_manager(cooldown=0.0)
    for _ in range(3):
        r = mgr.emit(_FakeRiskResult(stream_key="auth:"))
        assert len(r) == 1
    assert mgr.alert_count == 3
    assert mgr.suppressed_count == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_counters():
    now = [0.0]
    mgr = _make_manager(cooldown=60.0, clock_fn=lambda: now[0])

    mgr.emit(_FakeRiskResult(stream_key="auth:"))
    now[0] = 10.0
    mgr.emit(_FakeRiskResult(stream_key="auth:"))  # suppressed

    assert mgr.alert_count == 1
    assert mgr.suppressed_count == 1

    mgr.reset()
    assert mgr.alert_count == 0
    assert mgr.suppressed_count == 0
    assert mgr.active_stream_keys == []


def test_reset_allows_re_alert():
    """After reset, the same stream_key can fire immediately."""
    now = [0.0]
    mgr = _make_manager(cooldown=60.0, clock_fn=lambda: now[0])

    mgr.emit(_FakeRiskResult(stream_key="auth:"))
    mgr.reset()

    now[0] = 1.0
    r = mgr.emit(_FakeRiskResult(stream_key="auth:"))
    assert len(r) == 1  # fires again after reset


# ---------------------------------------------------------------------------
# active_stream_keys
# ---------------------------------------------------------------------------

def test_active_stream_keys_populated():
    mgr = _make_manager()
    mgr.emit(_FakeRiskResult(stream_key="auth:"))
    mgr.emit(_FakeRiskResult(stream_key="api:"))
    assert "auth:" in mgr.active_stream_keys
    assert "api:" in mgr.active_stream_keys


def test_non_anomaly_does_not_add_to_active_keys():
    mgr = _make_manager()
    mgr.emit(_FakeRiskResult(stream_key="auth:", is_anomaly=False))
    assert "auth:" not in mgr.active_stream_keys
