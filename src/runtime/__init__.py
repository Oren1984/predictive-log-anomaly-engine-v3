# src/runtime/__init__.py

# Purpose: Initialize the runtime inference package by importing key components 
# such as the InferenceEngine, SequenceBuffer, and RiskResult types.

# Input: - InferenceEngine: The core class responsible for performing anomaly detection inference on log data.
#        - SequenceBuffer: A utility class that maintains a rolling window of log events for each stream key (service:session_id).
#        - RiskResult: A dataclass that encapsulates the results of anomaly scoring, including the stream key, 
#          timestamp, model used, risk score, and other relevant metadata.

# Output: - InferenceEngine: Available for use in the runtime stage to perform inference on log data.
#         - SequenceBuffer: Available for managing rolling windows of log events during inference.
#         - RiskResult: Available for representing the results of anomaly scoring in a structured format.

# Used by: Other components of the runtime stage that need to perform inference on log data, 
# manage event sequences, or represent anomaly detection results.

"""Stage 5 — Runtime inference package."""
from .inference_engine import InferenceEngine
from .sequence_buffer import SequenceBuffer
from .types import RiskResult

__all__ = ["InferenceEngine", "SequenceBuffer", "RiskResult"]
