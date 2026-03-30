# src/api/ui.py

# Purpose: Implement a simple demo UI for the API, 
# consisting of a single-page HTML interface and a thin RAG (Retrieval-Augmented Generation) 
# stub that provides answers to user queries based on a built-in knowledge base.

# Input: The ui_router defines two endpoints:
# 1. GET / — serves the demo UI single page (index.html).
# 2. POST /query — accepts a natural language question and returns
# a short answer along with the top-3 relevant knowledge base documents.

# Output: HTML content for the demo UI and JSON responses for RAG queries.

# Used by: The ui_router is included in the main API router (app.py) 
# to provide the demo UI and RAG functionality as part of the overall API.

"""Stage 7.1 — Demo UI: single-page HTML interface + thin RAG stub."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"

ui_router = APIRouter()


# ---------------------------------------------------------------------------
# GET / — serve the demo UI page
# ---------------------------------------------------------------------------

@ui_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the demo UI single page."""
    html_path = _TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        logger.error("UI: templates/index.html not found at %s", html_path)
        return HTMLResponse(
            content=(
                "<html><body><h1>Demo UI unavailable</h1>"
                "<p>templates/index.html was not found. "
                "Ensure the templates/ directory is present at the project root.</p>"
                "</body></html>"
            ),
            status_code=503,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# POST /query — thin RAG stub (built-in knowledge base)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    id: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


# Built-in knowledge base: static docs about the system.
_KB: list[dict[str, Any]] = [
    {
        "id": "runtime-inference",
        "score": 0.95,
        "snippet": (
            "Runtime inference uses a sliding window (default 50 events, stride 10). "
            "The InferenceEngine supports baseline (IsolationForest), transformer, "
            "and ensemble modes."
        ),
    },
    {
        "id": "alert-policy",
        "score": 0.91,
        "snippet": (
            "AlertManager emits alerts when risk_score exceeds the model threshold. "
            "Severity: critical (score/threshold >= 1.5x), high (1.2x), medium (1.0x). "
            "Cooldown is per stream key (env: ALERT_COOLDOWN_SECONDS)."
        ),
    },
    {
        "id": "ingest-api",
        "score": 0.88,
        "snippet": (
            "POST /ingest accepts: {service (required), token_id (required), "
            "session_id, timestamp, label}. "
            "Returns window_emitted, risk_result, and alert when an anomaly is detected."
        ),
    },
    {
        "id": "isolation-forest",
        "score": 0.85,
        "snippet": (
            "IsolationForest (n_estimators=200) trained on 495,405 sessions from HDFS+BGL. "
            "BGL F1=0.96, HDFS F1=0.047. Best overall F1=0.38 at threshold 0.307."
        ),
    },
    {
        "id": "dataset",
        "score": 0.80,
        "snippet": (
            "15.9M log events: HDFS (10.9M normal + 288K anomaly) and "
            "BGL (348K normal + 4.4M anomaly). "
            "Template mining extracted 7,833 unique log templates."
        ),
    },
    {
        "id": "sequence-buffer",
        "score": 0.78,
        "snippet": (
            "SequenceBuffer groups events by stream key (service name). "
            "When the window fills (WINDOW_SIZE events), the engine scores it. "
            "A new score is emitted every STRIDE events thereafter."
        ),
    },
    {
        "id": "docker-deploy",
        "score": 0.76,
        "snippet": (
            "Run: docker compose up. "
            "Services: api (port 8000), prometheus (9090), grafana (3000). "
            "Demo mode is pre-configured (DEMO_MODE=true, DEMO_SCORE=100.0)."
        ),
    },
    {
        "id": "template-mining",
        "score": 0.74,
        "snippet": (
            "Log template mining uses a 9-step regex pipeline "
            "(block IDs, timestamps, IPs, paths, hex values, numbers). "
            "Produces 7,833 templates: 7,792 BGL + 41 HDFS."
        ),
    },
]

# Answer templates keyed by a topic keyword present in the question.
_ANSWERS: dict[str, str] = {
    "alert": (
        "Alerts fire when the risk score exceeds the model threshold. "
        "Severity buckets: critical (score/threshold >= 1.5x), high (1.2x), medium (1.0x). "
        "A per-stream-key cooldown (ALERT_COOLDOWN_SECONDS) prevents duplicate alerts."
    ),
    "model": (
        "The core model is an IsolationForest trained on 495,405 log sessions. "
        "It achieves F1=0.96 on BGL data. "
        "An optional transformer and ensemble mode (baseline + transformer) are also supported."
    ),
    "ingest": (
        "POST /ingest with {service, token_id} (plus optional session_id, timestamp, label). "
        "Events accumulate in a sliding window per stream key. "
        "The engine scores the window at each stride boundary."
    ),
    "window": (
        "The SequenceBuffer groups events by stream key (service name). "
        "When the window is full (default 50 events) the InferenceEngine scores it. "
        "A new score is produced every stride (default 10) events."
    ),
    "dataset": (
        "Trained on two datasets: HDFS (Hadoop log entries) and BGL (Blue Gene/L supercomputer logs). "
        "Total: 15.9M events. BGL anomaly rate: 27.6%; HDFS anomaly rate: 2.6%."
    ),
    "score": (
        "Risk scores from IsolationForest range 0.29–0.44 in practice (higher = more anomalous). "
        "Demo mode uses DEMO_SCORE=100.0 to always trigger alerts without trained models."
    ),
    "threshold": (
        "The anomaly threshold is the F1-optimal cutpoint found during training (~0.307 for baseline). "
        "Alerts fire when risk_score > threshold. "
        "Severity is determined by the score/threshold ratio."
    ),
    "api": (
        "Endpoints: POST /ingest, GET /alerts, GET /health, GET /metrics (Prometheus). "
        "Auth via X-API-Key header; disable with DISABLE_AUTH=true. "
        "Demo mode (DEMO_MODE=true) scores windows without trained models."
    ),
    "template": (
        "Log template mining uses 9-step regex substitution "
        "(block IDs, timestamps, IPs, date strings, node names, paths, hex, numbers). "
        "Extracts 7,833 unique templates: 7,792 from BGL, 41 from HDFS."
    ),
    "docker": (
        "Run: docker compose build && docker compose up. "
        "The stack starts three services: api on :8000, Prometheus on :9090, Grafana on :3000. "
        "Demo mode is pre-configured (no trained models needed)."
    ),
    "grafana": (
        "Grafana is available at http://localhost:3000 (admin/admin). "
        "The Stage 08 dashboard shows: event rate, window rate, alerts by severity, "
        "ingest latency p95, and scoring latency p95."
    ),
    "prometheus": (
        "Prometheus scrapes /metrics on the API every 15 seconds. "
        "Key metrics: ingest_events_total, windows_scored_total, alerts_total, "
        "ingest_latency_seconds, scoring_latency_seconds."
    ),
}

_DEFAULT_ANSWER = (
    "This system performs real-time log anomaly detection. "
    "It ingests tokenised log events, groups them into sliding windows, "
    "scores them with a trained IsolationForest model, and fires alerts on anomalies. "
    "Try asking about: alerts, model, ingest, window, dataset, score, threshold, "
    "api, template, docker, grafana, or prometheus."
)


def _best_answer(question: str) -> str:
    q = question.lower()
    for kw, ans in _ANSWERS.items():
        if kw in q:
            return ans
    return _DEFAULT_ANSWER


def _top_sources(question: str, k: int = 3) -> list[dict[str, Any]]:
    """Rank KB docs by keyword overlap with the question."""
    words = set(question.lower().split())
    ranked = sorted(
        _KB,
        key=lambda d: sum(
            1 for w in words if w in (d["id"] + " " + d["snippet"]).lower()
        ),
        reverse=True,
    )
    return ranked[:k]


@ui_router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    """
    Thin RAG stub: keyword-match the question against the built-in knowledge base.

    Returns:
        answer  — short natural-language answer
        sources — top-3 matched knowledge-base documents
    """
    return QueryResponse(
        answer=_best_answer(body.question),
        sources=[
            SourceDoc(id=d["id"], score=d["score"], snippet=d["snippet"])
            for d in _top_sources(body.question)
        ],
    )
