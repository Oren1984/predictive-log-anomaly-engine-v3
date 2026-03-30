# src/runtime/types.py

# Purpose: Define the RiskResult dataclass, which encapsulates the results of 
# anomaly scoring performed by the InferenceEngine during runtime inference. 
# The RiskResult includes fields such as the stream key, 
# timestamp, model used, risk score, anomaly status, and decision threshold.

# Input: - stream_key: The identity of the log stream (service:session_id) for which the anomaly score was produced.
#        - timestamp: The wall-clock or log timestamp of the last event in the window that was scored.
#        - model: The name of the model that produced the score (e.g., "baseline", "transformer", "ensemble").
#        - risk_score: The anomaly score produced by the model, where higher values indicate more anomalous events. 
#          For ensemble models, this score is normalized to approximately 1.0.
#        - is_anomaly: A boolean indicating whether the risk_score exceeds the decision threshold, 
#          thus classifying the window as anomalous.
#        - threshold: The decision threshold used to determine if the risk_score indicates an anomaly.

# Output: - RiskResult: An instance of the RiskResult dataclass that encapsulates 
# the results of anomaly scoring, including the stream key, timestamp, 
# model used, risk score, anomaly status, and decision threshold. 
# This structured format allows for easy serialization and downstream processing of anomaly detection results.

# Used by: The RiskResult dataclass is used by the InferenceEngine in the 
# runtime stage to represent the results of anomaly scoring in a structured format. 
# It allows for easy serialization and downstream processing of anomaly detection results, 
# such as logging, alerting, or further analysis.

"""Stage 5 — Runtime: result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskResult:
    """
    Result of a single anomaly-scoring window emitted by InferenceEngine.

    Fields
    ------
    stream_key        : identity of the log stream (service:session_id)
    timestamp         : wall-clock or log timestamp of the last event in the window
    model             : which model produced the score ("baseline"|"transformer"|"ensemble")
    risk_score        : anomaly score (higher = more anomalous; ensemble normalised to ~1.0)
    is_anomaly        : True when risk_score >= threshold
    threshold         : decision threshold used
    evidence_window   : dict with decoded token/template info
    top_predictions   : optional list of top-k next-token predictions (transformer only)
    meta              : free-form extra metadata
    """

    stream_key: str
    timestamp: float | str
    model: str
    risk_score: float
    is_anomaly: bool
    threshold: float
    evidence_window: dict
    top_predictions: Optional[list] = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialisable dict (suitable for JSON / CSV)."""
        return {
            "stream_key": self.stream_key,
            "timestamp": self.timestamp,
            "model": self.model,
            "risk_score": self.risk_score,
            "is_anomaly": self.is_anomaly,
            "threshold": self.threshold,
            "evidence_window": self.evidence_window,
            "top_predictions": self.top_predictions,
            "meta": self.meta,
        }
