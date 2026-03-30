# tests/helpers_stage_07.py

# Purpose: Shared test helpers for Stage 7 tests.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are test helpers, indirectly used by Stage 7 tests)

"""
Shared test helpers for Stage 7 tests.

Provides a mock Pipeline that never loads ML models, so tests run fast.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alerts import Alert, AlertManager, AlertPolicy, N8nWebhookClient
from src.api.settings import Settings
from src.observability.metrics import MetricsRegistry
from src.runtime.types import RiskResult


# ---------------------------------------------------------------------------
# A minimal stub RiskResult
# ---------------------------------------------------------------------------

def _stub_risk_result(
    stream_key: str = "svc:",
    score: float = 0.5,
    is_anomaly: bool = False,
    threshold: float = 1.0,
) -> RiskResult:
    return RiskResult(
        stream_key=stream_key,
        timestamp=1_704_067_200.0,
        model="ensemble",
        risk_score=score,
        is_anomaly=is_anomaly,
        threshold=threshold,
        evidence_window={
            "tokens": [5, 6, 7],
            "template_ids": [3, 4, 5],
            "templates_preview": ["tid=3: memory_check", "tid=4: disk_io"],
            "window_start_ts": 1_704_067_150.0,
            "window_end_ts": 1_704_067_200.0,
        },
        top_predictions=None,
        meta={"window_size": 50, "label": None, "emit_index": 1},
    )


# ---------------------------------------------------------------------------
# MockInferenceEngine
# ---------------------------------------------------------------------------

class MockInferenceEngine:
    """
    Stub InferenceEngine.

    By default ingest() returns None (no window emitted).
    Set .next_result to a RiskResult to make the next call return a window.
    """

    def __init__(self) -> None:
        self._artifacts_loaded = True
        self.next_result: Optional[RiskResult] = None
        self._emit_counts: dict = {}

    def load_artifacts(self) -> None:
        self._artifacts_loaded = True

    def ingest(self, event: dict) -> Optional[RiskResult]:
        result = self.next_result
        self.next_result = None
        return result


# ---------------------------------------------------------------------------
# MockPipeline
# ---------------------------------------------------------------------------

class MockPipeline:
    """
    Minimal Pipeline substitute for HTTP tests.

    Wraps a MockInferenceEngine and a real AlertManager so the full
    ingest → alert path is exercised without loading ML models.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.metrics = metrics

        self.engine = MockInferenceEngine()
        policy = AlertPolicy(cooldown_seconds=0.0)
        self.manager = AlertManager(policy=policy)
        self.n8n_client = N8nWebhookClient(dry_run=True)
        self._alert_buffer: deque[dict] = deque(maxlen=200)

    def load_models(self) -> None:
        pass  # No-op for tests

    def process_event(self, event: dict) -> dict:
        risk_result = self.engine.ingest(event)

        result: dict = {
            "window_emitted": risk_result is not None,
            "risk_result": None,
            "alert": None,
        }

        if self.metrics:
            self.metrics.ingest_events_total.inc()

        if risk_result is None:
            return result

        if self.metrics:
            self.metrics.ingest_windows_total.inc()

        result["risk_result"] = risk_result.to_dict()

        alerts = self.manager.emit(risk_result)
        if alerts:
            fired = alerts[0]
            alert_dict = fired.to_dict()
            self._alert_buffer.append(alert_dict)
            result["alert"] = alert_dict

            if self.metrics:
                self.metrics.alerts_total.labels(severity=fired.severity).inc()

        return result

    def recent_alerts(self) -> list[dict]:
        return list(self._alert_buffer)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_mock_pipeline(
    settings: Optional[Settings] = None,
    metrics: Optional[MetricsRegistry] = None,
) -> MockPipeline:
    return MockPipeline(settings=settings, metrics=metrics)
