# src/runtime/inference_engine_v2.py
# Phase 6 — Runtime: v2 Inference Engine
#
# InferenceEngineV2 wraps V2Pipeline to provide:
#   - A stable interface for the API layer (routes_v2.py)
#   - An in-memory alert ring buffer (configurable size)
#   - Alert deduplication / cooldown per stream
#   - Metrics counters (optional)
#
# This is the v2 counterpart of src/runtime/inference_engine.py.
# It is completely isolated from the v1 engine.

from __future__ import annotations

import collections
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .pipeline_v2 import V2Pipeline, V2PipelineConfig, V2Result

logger = logging.getLogger(__name__)


class InferenceEngineV2:
    """
    v2 Inference Engine — wraps V2Pipeline with alerting and buffering.

    Parameters
    ----------
    cfg:
        V2PipelineConfig passed directly to V2Pipeline.
    alert_buffer_size:
        Maximum number of alerts to retain in the in-memory ring buffer.
    alert_cooldown_seconds:
        Minimum time (seconds) between alerts for the same stream.
    """

    def __init__(
        self,
        cfg: Optional[V2PipelineConfig] = None,
        alert_buffer_size: int = 200,
        alert_cooldown_seconds: float = 60.0,
    ) -> None:
        self._pipeline = V2Pipeline(cfg)
        self._alert_buffer: collections.deque = collections.deque(
            maxlen=alert_buffer_size
        )
        self._cooldown_seconds = alert_cooldown_seconds
        # stream_key → last alert timestamp (monotonic)
        self._last_alert_ts: Dict[str, float] = {}

        logger.info(
            "InferenceEngineV2 initialised: buffer=%d cooldown=%.0fs",
            alert_buffer_size, alert_cooldown_seconds,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load all v2 model artifacts. Must be called before process_log."""
        self._pipeline.load_models()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline.is_loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_log(
        self,
        raw_log: str,
        service: str = "default",
        session_id: str = "",
        timestamp: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Process one raw log string and optionally fire an alert.

        Returns
        -------
        dict with keys:
            window_emitted  bool
            result          V2Result (or None if not emitted)
            alert           dict (or None if no alert fired)
        """
        result: V2Result = self._pipeline.process_log(
            raw_log=raw_log,
            service=service,
            session_id=session_id,
            timestamp=timestamp,
        )

        if not result.window_emitted:
            return {"window_emitted": False, "result": None, "alert": None}

        alert = None
        if result.is_anomaly:
            alert = self._maybe_fire_alert(result, service, session_id, timestamp)

        return {
            "window_emitted": True,
            "result": result,
            "alert": alert,
        }

    # ------------------------------------------------------------------
    # Alert logic
    # ------------------------------------------------------------------

    def _maybe_fire_alert(
        self,
        result: V2Result,
        service: str,
        session_id: str,
        timestamp: float,
    ) -> Optional[Dict[str, Any]]:
        """Fire an alert if cooldown has elapsed for this stream."""
        now = time.monotonic()
        key = result.stream_key
        last = self._last_alert_ts.get(key, 0.0)

        if now - last < self._cooldown_seconds:
            logger.debug(
                "Alert suppressed (cooldown): stream=%s score=%.4f",
                key, result.anomaly_score,
            )
            return None

        self._last_alert_ts[key] = now
        alert = {
            "alert_id": str(uuid.uuid4()),
            "severity": result.severity,
            "service": service,
            "session_id": session_id,
            "score": result.anomaly_score,
            "timestamp": timestamp or time.time(),
            "stream_key": key,
            "is_anomaly": result.is_anomaly,
            "severity_confidence": result.severity_confidence,
            "severity_probabilities": result.severity_probabilities,
            "model_name": "v2_pipeline",
        }
        self._alert_buffer.append(alert)
        logger.info(
            "ALERT fired: id=%s severity=%s stream=%s score=%.4f",
            alert["alert_id"], alert["severity"], key, result.anomaly_score,
        )
        return alert

    # ------------------------------------------------------------------
    # Alert retrieval
    # ------------------------------------------------------------------

    def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return the most recent alerts from the ring buffer (newest first).

        Parameters
        ----------
        limit:
            Maximum number of alerts to return (default 50).
        """
        alerts = list(self._alert_buffer)
        alerts.reverse()
        return alerts[:limit]

    # ------------------------------------------------------------------
    # Health / diagnostics
    # ------------------------------------------------------------------

    def health_info(self) -> Dict[str, Any]:
        """Return a summary for the /health endpoint."""
        info = self._pipeline.model_info()
        info["alert_buffer_used"] = len(self._alert_buffer)
        info["alert_buffer_capacity"] = self._alert_buffer.maxlen
        return info
