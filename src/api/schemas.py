# src/api/schemas.py

# Purpose: Define the Pydantic v2 request and response schemas for the API endpoints.
# This includes the schema for the ingest request, 
# as well as the response schemas for ingesting events, 
# listing alerts, and checking health status.

# Input: The schemas are defined as Pydantic models, which provide validation 
# and serialization for the API requests and responses. 
# The IngestRequest schema defines the expected fields for the POST /ingest endpoint, 
# while the IngestResponse, AlertListResponse, and HealthResponse schemas define 
# the structure of the responses for their respective endpoints.

# Output: The defined schemas are used in the API route definitions (src.api.routes.py) 
# to specify the expected request body and response models for each endpoint. 
# They ensure that the API receives and returns data in a consistent and validated format.

# Used by: The schemas defined in this file are used in the route definitions in src.api.routes.py 
# to validate incoming requests and structure outgoing responses. 
# They are also used in the test file test_stage_07_api.py 
# to create test cases for the API endpoints.

"""Stage 7 — API: Pydantic v2 request/response schemas."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Single log event sent to POST /ingest."""

    timestamp: float = Field(default=0.0, description="Unix epoch of the event")
    service: str = Field(..., description="Service/component name, e.g. 'hdfs'")
    session_id: str = Field(default="", description="Session or block identifier")
    token_id: int = Field(..., description="Template token ID (offset-2 encoded)")
    label: Optional[int] = Field(
        default=None, description="Ground-truth label (0=normal,1=anomaly) if known"
    )

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Inline sub-schemas
# ---------------------------------------------------------------------------

class RiskResultSchema(BaseModel):
    stream_key: str
    timestamp: float
    model: str
    risk_score: float
    is_anomaly: bool
    threshold: float
    evidence_window: dict[str, Any]
    top_predictions: Optional[list[dict[str, Any]]] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class AlertSchema(BaseModel):
    alert_id: str
    severity: str
    service: str
    score: float
    timestamp: float
    evidence_window: dict[str, Any]
    model_name: str
    threshold: float
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    """Response from POST /ingest."""

    window_emitted: bool = Field(
        description="True when InferenceEngine emitted a scored window"
    )
    risk_result: Optional[RiskResultSchema] = Field(
        default=None, description="Scoring result (present when window_emitted=True)"
    )
    alert: Optional[AlertSchema] = Field(
        default=None,
        description="Fired alert (present when anomaly detected and cooldown cleared)",
    )


class AlertListResponse(BaseModel):
    """Response from GET /alerts."""

    count: int
    alerts: list[AlertSchema]


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    uptime_seconds: float
    components: dict[str, Any]
