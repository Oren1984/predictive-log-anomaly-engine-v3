#src/alerts/models.py

# Purpose: Define the Alert and AlertPolicy data models used in the alerting system.

# Input: The Alert class represents a fired alert with all relevant information, 
# while the AlertPolicy class defines the rules for when a RiskResult should 
# be converted into an Alert and how to classify its severity. 
# These models are used by the AlertManager to manage the lifecycle of 
# alerts based on the results from the inference engine.

# Output: The Alert and AlertPolicy classes defined in this module are used by the AlertManager class 
# (src/alerts/manager.py) to manage the lifecycle of alerts based on the results from the inference engine.
# They are also used in various test files (e.g., test_stage_06_alert_policy.py, test_stage_06_dedup_cooldown.py) 
# to create mock RiskResult objects and verify that the alerting logic works as expected under different conditions. 
# Additionally, these classes are used in the main API implementation (src.api.app.py) 
# to create and classify alerts when processing events.

# Used by: The Alert and AlertPolicy classes defined in this module are used by the AlertManager class (src/alerts/manager.py) 
# to manage the lifecycle of alerts based on the results from the inference engine. 
# They are also used in various test files (e.g., test_stage_06_alert_policy.py, 
# test_stage_06_dedup_cooldown.py) to create mock RiskResult objects and 
# verify that the alerting logic works as expected under different conditions. 
# Additionally, these classes are used in the main API implementation (src.api.app.py) 
# to create and classify alerts when processing events.

"""Stage 6 — Alerts: domain models (Alert, AlertPolicy)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..runtime.types import RiskResult


@dataclass
class Alert:
    """
    A fired alert derived from a RiskResult window.

    Fields
    ------
    alert_id        : unique UUID string
    severity        : "critical" | "high" | "medium" | "low"
    service         : service name (left side of stream_key)
    score           : raw anomaly score from the scoring model
    timestamp       : wall-clock time of the last event in the window
    evidence_window : lean snapshot of the scored window (templates, counts, timestamps)
    model_name      : which model produced the score
    threshold       : decision threshold that was used
    meta            : free-form extras (stream_key, emit_index, …)
    """

    alert_id: str
    severity: str
    service: str
    score: float
    timestamp: float
    evidence_window: dict
    model_name: str
    threshold: float
    meta: dict = field(default_factory=dict)

    # V3 Semantic enrichment — all optional, None when semantic layer is disabled.
    explanation: Optional[str] = None
    semantic_similarity: Optional[float] = None
    top_similar_events: Optional[list] = None
    evidence_tokens: Optional[list] = None

    def to_dict(self) -> dict:
        """Serialisable dict suitable for JSON / n8n payload."""
        d: dict = {
            "alert_id":       self.alert_id,
            "severity":       self.severity,
            "service":        self.service,
            "score":          self.score,
            "timestamp":      self.timestamp,
            "evidence_window": self.evidence_window,
            "model_name":     self.model_name,
            "threshold":      self.threshold,
            "meta":           self.meta,
            # Semantic fields — included when set (None serialises cleanly to null)
            "explanation":          self.explanation,
            "semantic_similarity":  self.semantic_similarity,
            "top_similar_events":   self.top_similar_events,
            "evidence_tokens":      self.evidence_tokens,
        }
        return d


@dataclass
class AlertPolicy:
    """
    Policy that controls when a RiskResult becomes an Alert and what severity it gets.

    Parameters
    ----------
    threshold                   : additional minimum score filter on top of is_anomaly;
                                  0.0 (default) means trust is_anomaly from the engine.
    cooldown_seconds            : per-stream-key cooldown after each fired alert.
    aggregation_window_seconds  : (metadata) length of aggregation window; not enforced
                                  internally — provided for downstream use.
    min_events                  : minimum event count in the window to allow alerting;
                                  0 means no minimum.
    severity_buckets            : mapping of severity label to score/threshold multiplier.
                                  A score is classified as severity S when
                                  score >= threshold * severity_buckets[S].
                                  Checked highest-multiplier-first.
    """

    threshold: float = 0.0
    cooldown_seconds: float = 60.0
    aggregation_window_seconds: float = 300.0
    min_events: int = 0
    severity_buckets: dict = field(default_factory=lambda: {
        "critical": 1.5,   # score >= 1.5x model threshold
        "high":     1.2,   # score >= 1.2x model threshold
        "medium":   1.0,   # score >= 1.0x model threshold (= any anomaly)
    })

    # ------------------------------------------------------------------
    def should_alert(self, risk_result: "RiskResult") -> bool:
        """Return True when risk_result should fire an alert."""
        if not risk_result.is_anomaly:
            return False
        # Optional additional score filter (0 = disabled)
        if self.threshold > 0 and risk_result.risk_score < self.threshold:
            return False
        return True

    def classify_severity(self, score: float, threshold: float) -> str:
        """
        Classify severity by comparing score/threshold ratio against severity_buckets.

        Evaluated highest-multiplier-first so "critical" beats "high" beats "medium".
        Falls back to "low" if no bucket matches.
        """
        effective = max(threshold, 1e-9)
        ratio = score / effective
        for severity, multiplier in sorted(
            self.severity_buckets.items(), key=lambda x: x[1], reverse=True
        ):
            if ratio >= multiplier:
                return severity
        return "low"

    def risk_to_alert(self, risk_result: "RiskResult") -> Alert:
        """Convert a RiskResult into an Alert using this policy."""
        service = risk_result.stream_key.split(":")[0]

        ts = risk_result.timestamp
        if isinstance(ts, str):
            try:
                ts = float(ts)
            except (ValueError, TypeError):
                ts = 0.0
        else:
            ts = float(ts) if ts is not None else 0.0

        severity = self.classify_severity(risk_result.risk_score, risk_result.threshold)

        ew = risk_result.evidence_window or {}
        evidence = {
            "templates_preview": ew.get("templates_preview", [])[:5],
            "token_count":       len(ew.get("tokens", [])),
            "start_ts":          ew.get("window_start_ts"),
            "end_ts":            ew.get("window_end_ts"),
        }

        return Alert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            service=service,
            score=risk_result.risk_score,
            timestamp=ts,
            evidence_window=evidence,
            model_name=risk_result.model,
            threshold=risk_result.threshold,
            meta={
                "stream_key": risk_result.stream_key,
                "is_anomaly": risk_result.is_anomaly,
                **(risk_result.meta or {}),
            },
        )
