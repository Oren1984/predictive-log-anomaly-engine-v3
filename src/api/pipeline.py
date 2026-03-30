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
from ..semantic import (
    RuleBasedExplainer,
    SemanticConfig,
    SemanticEmbedder,
    SemanticModelLoader,
    SemanticSimilarity,
)
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

        # V3 Semantic enrichment — inert when SEMANTIC_ENABLED=false (the default).
        self._semantic_config = SemanticConfig()
        self._semantic_loader = SemanticModelLoader(self._semantic_config)
        self._semantic_embedder = SemanticEmbedder(
            self._semantic_config, self._semantic_loader
        )
        self._explainer = RuleBasedExplainer()
        self._similarity = SemanticSimilarity()
        # Ring-buffer of (label, embedding) pairs for historical similarity lookup.
        self._semantic_history: deque[tuple[str, object]] = deque(maxlen=200)

    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load all ML artifacts. Called once during app lifespan startup."""
        t0 = time.perf_counter()
        self.engine.load_artifacts()
        # Semantic model load is a no-op when SEMANTIC_ENABLED=false.
        self._semantic_loader.load()
        elapsed = time.perf_counter() - t0
        logger.info(
            "Pipeline: models loaded in %.2fs (mode=%s, semantic=%s)",
            elapsed,
            self.settings.model_mode,
            self._semantic_config.semantic_enabled,
        )
        if self.metrics:
            self.metrics.semantic_model_ready.set(
                1.0 if self._semantic_loader.is_ready else 0.0
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
            # V3: enrich with semantic fields after anomaly confirmation.
            # No-op when SEMANTIC_ENABLED=false.
            self._enrich_alert(fired)
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

    def get_alert_by_id(self, alert_id: str) -> Optional[dict]:
        """Look up an alert dict from the ring buffer by alert_id. Returns None if not found."""
        for a in self._alert_buffer:
            if a.get("alert_id") == alert_id:
                return a
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enrich_alert(self, alert: Alert) -> None:
        """
        Attach V3 semantic fields to a fired alert, in-place.

        Gated by SemanticConfig.semantic_enabled; entirely inert when disabled.
        Called only after anomaly confirmation (alert already created).
        Records metrics when enrichment runs.
        """
        if not self._semantic_config.semantic_enabled:
            return

        t0 = time.perf_counter()

        # 1. Rule-based explanation
        enrichment = self._explainer.explain(alert.evidence_window)
        alert.explanation = enrichment["explanation"]
        alert.evidence_tokens = enrichment["evidence_tokens"] or None

        # 2. Semantic embedding + similarity against history
        alert_text = " ".join(
            str(t) for t in alert.evidence_window.get("templates_preview", [])
        ) or alert.service
        embedding = self._semantic_embedder.embed(alert_text)

        if embedding is not None:
            if self._semantic_history:
                top = self._similarity.top_k(
                    embedding,
                    list(self._semantic_history),  # type: ignore[arg-type]
                    k=3,
                )
                if top:
                    alert.semantic_similarity = round(top[0]["score"], 4)
                    alert.top_similar_events = top

            # Store current alert in history for future comparisons.
            label = f"{alert.service}:{alert.alert_id[:8]}"
            self._semantic_history.append((label, embedding))

        # 3. Record metrics
        elapsed = time.perf_counter() - t0
        if self.metrics:
            self.metrics.semantic_enrichments_total.inc()
            self.metrics.semantic_enrichment_latency_seconds.observe(elapsed)
