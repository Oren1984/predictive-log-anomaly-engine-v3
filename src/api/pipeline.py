# src/api/pipeline.py

# Purpose: Implement the Pipeline class, which serves as a container for 
# all live runtime components shared across API requests.
# This includes the inference engine, alert manager, n8n client, and metrics registry.

# Input: The Pipeline class is initialized with optional settings and metrics registry. 
# It provides methods to load ML models and process 
# incoming events through the pipeline, which includes scoring and alerting.

# Output: The process_event method returns a dictionary containing the results of processing an event, 
# including whether a window was emitted, the risk result, 
# and any alert that was fired. 
# The recent_alerts method returns a list of the most recent alerts from the buffer.

# Used by: The Pipeline class is used in the main API implementation (src.api.app.py) 
# to process incoming events and manage the inference engine, alert manager, and n8n client. 
# It is also tested in the test file test_stage_07_pipeline.py.

"""Stage 7 — API: Pipeline container (engine + alerts + metrics + buffer)."""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

from ..alerts import Alert, AlertManager, AlertPolicy, N8nWebhookClient
from ..runtime import InferenceEngine
from .settings import Settings

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Container for all live runtime components shared across API requests.

    Attributes
    ----------
    engine          : InferenceEngine (ingest events, emit windows)
    manager         : AlertManager (dedup + cooldown)
    n8n_client      : N8nWebhookClient (dry-run outbox by default)
    metrics         : MetricsRegistry | None
    _alert_buffer   : deque of the most-recent Alert.to_dict() dicts
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        metrics=None,
    ) -> None:
        self.settings: Settings = settings or Settings()
        self.metrics = metrics

        self.engine = InferenceEngine(
            mode=self.settings.model_mode,
            window_size=self.settings.window_size,
            stride=self.settings.stride,
        )
        # Wire demo/fallback behaviour from settings
        self.engine.demo_mode = self.settings.demo_mode
        self.engine.fallback_score = self.settings.demo_score

        policy = AlertPolicy(
            cooldown_seconds=self.settings.alert_cooldown_seconds
        )
        self.manager = AlertManager(policy=policy)
        self.n8n_client = N8nWebhookClient()

        self._alert_buffer: deque[dict] = deque(
            maxlen=self.settings.alert_buffer_size
        )

    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load all ML artifacts. Called once during app lifespan startup."""
        t0 = time.perf_counter()
        self.engine.load_artifacts()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Pipeline: models loaded in %.2fs (mode=%s)",
            elapsed,
            self.settings.model_mode,
        )

    # ------------------------------------------------------------------

    def process_event(self, event: dict) -> dict:
        """
        Feed one event through the full pipeline.

        Returns a dict with keys:
            window_emitted  : bool
            risk_result     : dict | None
            alert           : dict | None
        """
        t_score = time.perf_counter()
        risk_result = self.engine.ingest(event)
        score_elapsed = time.perf_counter() - t_score

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
            self.metrics.scoring_latency_seconds.observe(score_elapsed)

        result["risk_result"] = risk_result.to_dict()

        alerts: list[Alert] = self.manager.emit(risk_result)
        if alerts:
            fired = alerts[0]
            self.n8n_client.send(fired)
            alert_dict = fired.to_dict()
            self._alert_buffer.append(alert_dict)
            result["alert"] = alert_dict

            if self.metrics:
                self.metrics.alerts_total.labels(
                    severity=fired.severity
                ).inc()

        return result

    def recent_alerts(self) -> list[dict]:
        """Return the most-recent alerts from the ring buffer (newest last)."""
        return list(self._alert_buffer)
