# src/health/checks.py

# Purpose: Defines the HealthChecker class that performs critical 
# and optional health checks against the live Pipeline object 
# to determine the overall health status of the system.

# Input: - A Pipeline object (optional) that the HealthChecker will 
# use to perform checks against its components.

# Output: - A dictionary containing the overall health status, uptime, 
# and the status of individual components (inference engine, alert manager, alert buffer).

# Used by: - API endpoints that need to report the health status of the system, 
# such as readiness and liveness probes for container orchestration platforms like Kubernetes.

"""Stage 7 — Health: readiness and liveness checks."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.pipeline import Pipeline

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Performs lightweight checks against the live Pipeline object.

    Status values
    -------------
    healthy   — all checks pass
    degraded  — at least one optional check fails
    unhealthy — a critical check fails (model not loaded)
    """

    def __init__(self, pipeline: "Pipeline | None" = None) -> None:
        self._pipeline = pipeline
        self._start_time = time.time()

    def check(self) -> dict:
        """Run all checks and return a health payload dict."""
        components: dict[str, dict] = {}
        overall = "healthy"

        # ----------------------------------------------------------------
        # Critical: inference engine loaded
        # ----------------------------------------------------------------
        engine_ok = (
            self._pipeline is not None
            and self._pipeline.engine is not None
            and self._pipeline.engine._artifacts_loaded
        )
        components["inference_engine"] = {
            "status": "ok" if engine_ok else "unavailable",
            "artifacts_loaded": engine_ok,
        }
        if not engine_ok:
            overall = "unhealthy"

        # ----------------------------------------------------------------
        # Optional: alert manager reachable
        # ----------------------------------------------------------------
        manager_ok = (
            self._pipeline is not None
            and self._pipeline.manager is not None
        )
        components["alert_manager"] = {
            "status": "ok" if manager_ok else "unavailable",
        }
        if not manager_ok and overall == "healthy":
            overall = "degraded"

        # ----------------------------------------------------------------
        # Optional: alert ring buffer accessible
        # ----------------------------------------------------------------
        buffer_ok = (
            self._pipeline is not None
            and hasattr(self._pipeline, "_alert_buffer")
        )
        components["alert_buffer"] = {
            "status": "ok" if buffer_ok else "unavailable",
            "size": len(self._pipeline._alert_buffer) if buffer_ok else 0,
        }

        # ----------------------------------------------------------------
        # V3 Semantic layer — informational, never degrades overall status
        # ----------------------------------------------------------------
        semantic_cfg = getattr(self._pipeline, "_semantic_config", None)
        semantic_loader = getattr(self._pipeline, "_semantic_loader", None)
        components["semantic"] = {
            "enabled": semantic_cfg.semantic_enabled if semantic_cfg else False,
            "model_loaded": semantic_loader.is_ready if semantic_loader else False,
            "model_name": semantic_cfg.semantic_model if semantic_cfg else "n/a",
        }

        uptime_s = round(time.time() - self._start_time, 1)

        return {
            "status": overall,
            "uptime_seconds": uptime_s,
            "components": components,
        }
