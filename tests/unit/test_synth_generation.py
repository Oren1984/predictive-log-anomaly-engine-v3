# test/unit/test_synth_generation.py

# Purpose: Unit tests for synthetic data generation pipeline to verify 
# that it produces expected outputs and adheres to the canonical schema.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are unit tests for the synthetic data generation pipeline, 
# indirectly used by the generation script and
# any downstream models that consume the synthetic data)


"""
Unit tests for synthetic data generation pipeline.

Tests:
- Generator produces correct event count
- Two-service generate_all scenario
- Parquet written and non-empty
- Required columns exist in canonical output
- Label contains at least 2 classes (normal + anomaly)
- LogEvent to_dict / from_dict round-trip
- All four patterns emit events without error
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.synthetic import (
    AuthBruteForcePattern,
    DiskFullPattern,
    MemoryLeakPattern,
    NetworkFlapPattern,
    ScenarioBuilder,
    SyntheticLogGenerator,
)
from src.data.log_event import LogEvent

_PATTERNS = [
    MemoryLeakPattern(),
    DiskFullPattern(),
    AuthBruteForcePattern(),
    NetworkFlapPattern(),
]
_BASE_TS = 1_704_067_200.0


def _make_scenario(service: str, n: int, pattern_name: str, idx: int = 0) -> dict:
    builder = ScenarioBuilder()
    return builder.build_scenario(
        scenario_id=f"test_{pattern_name}_{idx}",
        service=service,
        host=f"host-{idx + 1:02d}",
        start_ts=_BASE_TS + idx * 3600,
        n_events=n,
        pattern_name=pattern_name,
    )


def _events_to_df(events: list) -> pd.DataFrame:
    """Convert events to canonical-schema DataFrame."""
    rows = []
    for ev in events:
        meta = ev.meta or {}
        rows.append({
            "timestamp": float(ev.timestamp) if ev.timestamp is not None else 0.0,
            "service":   ev.service,
            "level":     ev.level,
            "message":   ev.message,
            "meta":      json.dumps(meta),
            "label":     int(ev.label) if ev.label is not None else 0,
        })
    return pd.DataFrame(rows, columns=["timestamp", "service", "level", "message", "meta", "label"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_returns_correct_count():
    gen = SyntheticLogGenerator(_PATTERNS, seed=42)
    sc = _make_scenario("svc-a", 500, "memory_leak", 0)
    events = gen.generate(500, sc)
    assert len(events) == 500


def test_generate_all_two_services():
    """Run generator with n_events=2000 and 2 services; assert result length."""
    gen = SyntheticLogGenerator(_PATTERNS, seed=42)
    builder = ScenarioBuilder()
    scenarios = [
        builder.build_scenario(
            scenario_id="test_svc_a",
            service="svc-a",
            host="host-01",
            start_ts=_BASE_TS,
            n_events=1000,
            pattern_name="memory_leak",
        ),
        builder.build_scenario(
            scenario_id="test_svc_b",
            service="svc-b",
            host="host-02",
            start_ts=_BASE_TS + 3600,
            n_events=1000,
            pattern_name="disk_full",
        ),
    ]
    events = gen.generate_all(scenarios)
    assert len(events) == 2000


def test_parquet_written_and_non_empty():
    """Assert parquet written and non-empty."""
    gen = SyntheticLogGenerator(_PATTERNS, seed=42)
    builder = ScenarioBuilder()
    scenarios = [
        builder.build_scenario(
            scenario_id="test_a",
            service="svc-a",
            host="host-01",
            start_ts=_BASE_TS,
            n_events=1000,
            pattern_name="memory_leak",
        ),
        builder.build_scenario(
            scenario_id="test_b",
            service="svc-b",
            host="host-02",
            start_ts=_BASE_TS + 3600,
            n_events=1000,
            pattern_name="disk_full",
        ),
    ]
    events = gen.generate_all(scenarios)
    assert len(events) == 2000

    df = _events_to_df(events)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_synth.parquet"
        df.to_parquet(out, index=False)
        assert out.exists(), "Parquet file was not created"
        df2 = pd.read_parquet(out)
        assert len(df2) > 0, "Parquet file is empty"


def test_required_columns_exist():
    """Assert required columns exist in canonical output DataFrame."""
    gen = SyntheticLogGenerator(_PATTERNS, seed=42)
    sc = _make_scenario("svc-a", 200, "auth_brute_force", 0)
    events = gen.generate(200, sc)

    df = _events_to_df(events)
    required = {"timestamp", "service", "level", "message", "meta", "label"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_label_contains_at_least_two_classes():
    """Assert label contains at least 2 classes (normal + anomaly) in output."""
    gen = SyntheticLogGenerator(_PATTERNS, seed=42)
    builder = ScenarioBuilder()
    scenarios = [
        builder.build_scenario(
            scenario_id="test_a",
            service="svc-a",
            host="host-01",
            start_ts=_BASE_TS,
            n_events=1000,
            pattern_name="memory_leak",
        ),
        builder.build_scenario(
            scenario_id="test_b",
            service="svc-b",
            host="host-02",
            start_ts=_BASE_TS + 3600,
            n_events=1000,
            pattern_name="network_flap",
        ),
    ]
    events = gen.generate_all(scenarios)
    labels = {int(ev.label) if ev.label is not None else 0 for ev in events}
    assert 0 in labels, "No normal events (label=0) found"
    assert 1 in labels, "No anomaly events (label=1) found"


def test_log_event_to_dict_round_trip():
    """Test src.data.log_event.LogEvent to_dict / from_dict round-trip."""
    ev = LogEvent(
        timestamp=1_704_067_200.0,
        service="auth",
        level="INFO",
        message="test message",
        meta={"host": "host-01", "pattern": "memory_leak"},
        label=0,
    )
    d = ev.to_dict()
    assert d["timestamp"] == 1_704_067_200.0
    assert d["service"] == "auth"
    assert d["label"] == 0
    assert isinstance(d["meta"], str), "meta should be JSON-encoded string"

    ev2 = LogEvent.from_dict(d)
    assert ev2.service == ev.service
    assert ev2.label == ev.label
    assert ev2.meta == ev.meta


def test_all_patterns_emit():
    """Each pattern must emit events for all three phases without error."""
    gen = SyntheticLogGenerator(_PATTERNS, seed=42)
    for pattern in _PATTERNS:
        sc = _make_scenario("svc", 300, pattern.name, 0)
        events = gen.generate(300, sc)
        assert len(events) == 300, f"Pattern {pattern.name} returned wrong count"
        levels = {ev.level for ev in events}
        assert levels, f"No events from pattern {pattern.name}"
        # 300 events with 60/30/10 split should cover all phases
        phases = {(ev.meta or {}).get("phase") for ev in events}
        assert "normal" in phases, f"Pattern {pattern.name}: missing normal phase"
        assert "degradation" in phases, f"Pattern {pattern.name}: missing degradation phase"
        assert "failure" in phases, f"Pattern {pattern.name}: missing failure phase"
