# src/api/routes_v3.py
#
# Phase 8 — V3 Semantic API routes.
#
# New endpoints:
#   GET  /v3/alerts/{alert_id}/explanation  — semantic explanation for a specific alert
#   GET  /v3/models/info                    — inference mode + semantic layer status
#   POST /v3/ingest                         — versioned ingest alias with explicit semantic response
#
# All endpoints are additive and do not modify V1/V2 behaviour.

"""Phase 8 — V3 API routes: explanation, models/info, versioned ingest."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from .schemas import (
    AlertSchema,
    ExplanationResponse,
    IngestRequest,
    IngestResponse,
    ModelsInfoResponse,
    RiskResultSchema,
)

logger = logging.getLogger(__name__)

router_v3 = APIRouter(prefix="/v3", tags=["v3"])


# ---------------------------------------------------------------------------
# GET /v3/alerts/{alert_id}/explanation
# ---------------------------------------------------------------------------

@router_v3.get(
    "/alerts/{alert_id}/explanation",
    response_model=ExplanationResponse,
    summary="Semantic explanation for a specific alert",
)
async def get_alert_explanation(alert_id: str, request: Request) -> ExplanationResponse:
    """
    Return V3 semantic enrichment fields for a specific alert.

    The alert must be present in the in-memory ring buffer (most-recent
    ``alert_buffer_size`` alerts).  Returns 404 when not found.

    When the semantic layer is disabled (``SEMANTIC_ENABLED=false``), the
    response is still returned with ``semantic_enabled=false`` and null fields.
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    # Use get_alert_by_id if available (real Pipeline); fall back to linear scan.
    get_fn = getattr(pipeline, "get_alert_by_id", None)
    if get_fn is not None:
        alert_dict = get_fn(alert_id)
    else:
        alert_dict = next(
            (a for a in pipeline.recent_alerts() if a.get("alert_id") == alert_id),
            None,
        )

    if alert_dict is None:
        raise HTTPException(
            status_code=404,
            detail=f"Alert '{alert_id}' not found in ring buffer.",
        )

    semantic_cfg = getattr(pipeline, "_semantic_config", None)
    semantic_enabled = semantic_cfg.semantic_enabled if semantic_cfg else False

    return ExplanationResponse(
        alert_id=alert_id,
        semantic_enabled=semantic_enabled,
        explanation=alert_dict.get("explanation"),
        evidence_tokens=alert_dict.get("evidence_tokens"),
        semantic_similarity=alert_dict.get("semantic_similarity"),
        top_similar_events=alert_dict.get("top_similar_events"),
    )


# ---------------------------------------------------------------------------
# GET /v3/models/info
# ---------------------------------------------------------------------------

@router_v3.get(
    "/models/info",
    response_model=ModelsInfoResponse,
    summary="Inference engine and semantic layer status",
)
async def get_models_info(request: Request) -> ModelsInfoResponse:
    """
    Return a summary of the loaded inference engine and V3 semantic layer state.

    Useful for diagnostics and integration checks without hitting ``/health``.
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    engine = getattr(pipeline, "engine", None)
    artifacts_loaded = getattr(engine, "_artifacts_loaded", False) if engine else False
    model_mode = getattr(
        getattr(pipeline, "settings", None), "model_mode", "unknown"
    )

    semantic_cfg = getattr(pipeline, "_semantic_config", None)
    semantic_loader = getattr(pipeline, "_semantic_loader", None)

    engine_v2 = getattr(request.app.state, "engine_v2", None)

    return ModelsInfoResponse(
        inference_mode=model_mode,
        artifacts_loaded=artifacts_loaded,
        semantic_enabled=semantic_cfg.semantic_enabled if semantic_cfg else False,
        semantic_model=semantic_cfg.semantic_model if semantic_cfg else "n/a",
        semantic_model_loaded=semantic_loader.is_ready if semantic_loader else False,
        explanation_enabled=semantic_cfg.explanation_enabled if semantic_cfg else False,
        v2_engine_available=engine_v2 is not None,
    )


# ---------------------------------------------------------------------------
# POST /v3/ingest
# ---------------------------------------------------------------------------

@router_v3.post(
    "/ingest",
    response_model=IngestResponse,
    summary="V3 versioned ingest with semantic enrichment fields",
)
async def ingest_v3(body: IngestRequest, request: Request) -> IngestResponse:
    """
    Feed a tokenised log event into the pipeline (V3 versioned path).

    Functionally identical to ``POST /ingest`` but lives under the ``/v3/``
    prefix and explicitly surfaces semantic enrichment fields in the alert
    response when ``SEMANTIC_ENABLED=true``.
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
        if getattr(pipeline, "metrics", None):
            pipeline.metrics.ingest_errors_total.inc()
        logger.exception("v3 ingest error: %s", exc)
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
            explanation=a.get("explanation"),
            semantic_similarity=a.get("semantic_similarity"),
            top_similar_events=a.get("top_similar_events"),
            evidence_tokens=a.get("evidence_tokens"),
        )

    return IngestResponse(
        window_emitted=result["window_emitted"],
        risk_result=risk_schema,
        alert=alert_schema,
    )
