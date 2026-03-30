# src/api/routes_v2.py
# Phase 7 — API Integration (v2)
#
# Defines the v2 API routes:
#   POST /ingest_v2    feed a raw log string through the v2 pipeline
#   GET  /alerts_v2   list recent v2 alerts
#
# The v2 engine is stored on app.state.engine_v2 (set by app.py at startup).
# These routes are additive — they do not modify or replace the v1 routes.

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router_v2 = APIRouter(prefix="/v2", tags=["v2"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestV2Request(BaseModel):
    """Single raw log string sent to POST /v2/ingest."""

    raw_log: str = Field(..., description="Raw log message text")
    service: str = Field(default="default", description="Service/component name")
    session_id: str = Field(default="", description="Session or block identifier")
    timestamp: float = Field(
        default=0.0, description="Unix epoch of the event (0 = now)"
    )

    model_config = {"extra": "ignore"}


class V2ResultSchema(BaseModel):
    stream_key: str
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    severity: Optional[str] = None
    severity_confidence: Optional[float] = None
    severity_probabilities: Optional[List[float]] = None


class AlertV2Schema(BaseModel):
    alert_id: str
    severity: str
    service: str
    session_id: str
    score: float
    timestamp: float
    stream_key: str
    is_anomaly: bool
    severity_confidence: Optional[float] = None
    severity_probabilities: Optional[List[float]] = None
    model_name: str


class IngestV2Response(BaseModel):
    window_emitted: bool
    result: Optional[V2ResultSchema] = None
    alert: Optional[AlertV2Schema] = None


class AlertListV2Response(BaseModel):
    count: int
    alerts: List[AlertV2Schema]


# ---------------------------------------------------------------------------
# POST /v2/ingest
# ---------------------------------------------------------------------------

@router_v2.post("/ingest", response_model=IngestV2Response)
async def ingest_v2(body: IngestV2Request, request: Request) -> IngestV2Response:
    """
    Feed a raw log string through the v2 ML pipeline.

    The engine buffers log embeddings into a rolling window.  A scored result
    is returned when a full window accumulates; otherwise ``window_emitted=False``.
    """
    engine = getattr(request.app.state, "engine_v2", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "v2 inference engine not initialised. "
                "Set MODEL_MODE=v2 or ensure v2 models are trained."
            ),
        )

    ts = body.timestamp if body.timestamp > 0 else time.time()

    try:
        output = engine.process_log(
            raw_log=body.raw_log,
            service=body.service,
            session_id=body.session_id,
            timestamp=ts,
        )
    except Exception as exc:
        logger.exception("v2 ingest error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result_schema = None
    if output.get("result") is not None:
        r = output["result"]
        result_schema = V2ResultSchema(
            stream_key=r.stream_key,
            anomaly_score=r.anomaly_score,
            is_anomaly=r.is_anomaly,
            severity=r.severity,
            severity_confidence=r.severity_confidence,
            severity_probabilities=r.severity_probabilities,
        )

    alert_schema = None
    if output.get("alert") is not None:
        a = output["alert"]
        alert_schema = AlertV2Schema(
            alert_id=a["alert_id"],
            severity=a["severity"],
            service=a["service"],
            session_id=a.get("session_id", ""),
            score=a["score"],
            timestamp=a["timestamp"],
            stream_key=a["stream_key"],
            is_anomaly=a["is_anomaly"],
            severity_confidence=a.get("severity_confidence"),
            severity_probabilities=a.get("severity_probabilities"),
            model_name=a["model_name"],
        )

    return IngestV2Response(
        window_emitted=output["window_emitted"],
        result=result_schema,
        alert=alert_schema,
    )


# ---------------------------------------------------------------------------
# GET /v2/alerts
# ---------------------------------------------------------------------------

@router_v2.get("/alerts", response_model=AlertListV2Response)
async def alerts_v2(request: Request, limit: int = 50) -> AlertListV2Response:
    """Return the most recent v2 alerts from the in-memory ring buffer."""
    engine = getattr(request.app.state, "engine_v2", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="v2 inference engine not initialised.",
        )

    raw = engine.recent_alerts(limit=limit)
    alerts = [
        AlertV2Schema(
            alert_id=a["alert_id"],
            severity=a["severity"],
            service=a["service"],
            session_id=a.get("session_id", ""),
            score=a["score"],
            timestamp=a["timestamp"],
            stream_key=a["stream_key"],
            is_anomaly=a["is_anomaly"],
            severity_confidence=a.get("severity_confidence"),
            severity_probabilities=a.get("severity_probabilities"),
            model_name=a["model_name"],
        )
        for a in raw
    ]
    return AlertListV2Response(count=len(alerts), alerts=alerts)
