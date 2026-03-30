# src/api/routes.py

# Purpose: Define the API routes for the FastAPI application, 
# including endpoints for ingesting events, listing recent alerts, 
# checking health status, and exposing metrics.

# Input: The routes are defined using FastAPI's APIRouter, 
# and they interact with the shared Pipeline instance to process events 
# and retrieve alerts. The /ingest endpoint accepts a tokenised log event, 
# processes it through the pipeline, and returns the results. The /alerts 
# endpoint returns the most recent alerts from the pipeline's buffer. 
# The /health endpoint checks the health status of the application, 
# and the /metrics endpoint exposes Prometheus metrics.

# Output: Each endpoint returns a response model defined in the .schemas module, 
# which includes structured data for the results of processing events, listing alerts, 
# and health status. 
# The /metrics endpoint returns a plain text response in Prometheus format.

# Used by: The routes defined in this file are included in the main FastAPI application instance created in src.api.app.py, 
# and they are used to handle incoming HTTP requests to the API. 
# They are also tested in the test file test_stage_07_api.py.

"""Stage 7 — API: route definitions."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import PlainTextResponse

from .schemas import (
    AlertListResponse,
    AlertSchema,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RiskResultSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def ingest_event(body: IngestRequest, request: Request) -> IngestResponse:
    """
    Feed a single tokenised log event into the inference pipeline.

    Returns a scored window and optional alert whenever a stride boundary
    is crossed; otherwise returns ``window_emitted=False``.
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    event = {
        "timestamp":  body.timestamp,
        "service":    body.service,
        "session_id": body.session_id,
        "token_id":   body.token_id,
        "label":      body.label,
    }

    try:
        result = pipeline.process_event(event)
    except Exception as exc:
        if pipeline.metrics:
            pipeline.metrics.ingest_errors_total.inc()
        logger.exception("Error processing ingest event: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    risk_schema = None
    if result["risk_result"] is not None:
        rr = result["risk_result"]
        risk_schema = RiskResultSchema(
            stream_key=rr["stream_key"],
            timestamp=float(rr["timestamp"]),
            model=rr["model"],
            risk_score=rr["risk_score"],
            is_anomaly=rr["is_anomaly"],
            threshold=rr["threshold"],
            evidence_window=rr["evidence_window"],
            top_predictions=rr.get("top_predictions"),
            meta=rr.get("meta", {}),
        )

    alert_schema = None
    if result["alert"] is not None:
        a = result["alert"]
        alert_schema = AlertSchema(
            alert_id=a["alert_id"],
            severity=a["severity"],
            service=a["service"],
            score=a["score"],
            timestamp=a["timestamp"],
            evidence_window=a["evidence_window"],
            model_name=a["model_name"],
            threshold=a["threshold"],
            meta=a.get("meta", {}),
        )

    return IngestResponse(
        window_emitted=result["window_emitted"],
        risk_result=risk_schema,
        alert=alert_schema,
    )


# ---------------------------------------------------------------------------
# GET /alerts
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=AlertListResponse)
async def list_alerts(request: Request) -> AlertListResponse:
    """Return the most-recent alerts from the in-memory ring buffer."""
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    raw = pipeline.recent_alerts()
    alerts = [
        AlertSchema(
            alert_id=a["alert_id"],
            severity=a["severity"],
            service=a["service"],
            score=a["score"],
            timestamp=a["timestamp"],
            evidence_window=a["evidence_window"],
            model_name=a["model_name"],
            threshold=a["threshold"],
            meta=a.get("meta", {}),
        )
        for a in raw
    ]
    return AlertListResponse(count=len(alerts), alerts=alerts)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

_HEALTH_GAUGE_VALUES = {"healthy": 1.0, "degraded": 0.5}


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness probe. Also updates the service_health Prometheus gauge."""
    checker = getattr(request.app.state, "health_checker", None)
    if checker is None:
        return HealthResponse(
            status="unknown",
            uptime_seconds=0.0,
            components={},
        )
    payload = checker.check()

    # Reflect application health state in Prometheus so Grafana and alert rules
    # can observe real component status rather than mere scrape reachability.
    metrics_reg = getattr(request.app.state, "metrics", None)
    if metrics_reg is not None:
        gauge_value = _HEALTH_GAUGE_VALUES.get(payload["status"], 0.0)
        metrics_reg.service_health.set(gauge_value)

    return HealthResponse(**payload)


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> PlainTextResponse:
    """Prometheus text-format metrics endpoint."""
    registry = getattr(request.app.state, "metrics", None)
    if registry is None:
        return PlainTextResponse("# metrics disabled\n")
    body, content_type = registry.generate_text()
    return PlainTextResponse(content=body, media_type=content_type)
