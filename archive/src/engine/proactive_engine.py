# src/engine/proactive_engine.py
#
# STATUS: LEGACY — Phase 7 development orchestrator (not wired to production API)
#
# This module is NOT part of the live inference path.
# Active production path: src/runtime/inference_engine.py + src/api/pipeline.py
# Retained for: test coverage (test_proactive_engine.py) and reference architecture.
#
# Phase 7: AIOps Engine Integration

# ProactiveMonitorEngine orchestrates the full AI pipeline built in
# Phases 2-6:

#   LogPreprocessor (Phase 2)
#     -> rolling embedding buffer
#     -> SystemBehaviorModel / LSTM (Phase 4)
#     -> AnomalyDetector / DAE   (Phase 5)
#     -> SeverityClassifier / MLP (Phase 6)
#     -> EngineResult

# Design notes:
#   - Completely isolated from the existing runtime pipeline.
#     InferenceEngine (src/runtime/), Pipeline (src/api/pipeline.py),
#     AlertManager, and all FastAPI routes are unchanged.
#   - All model loading uses try/except with warn-and-continue semantics.
#     The engine never raises on missing or corrupt model files.
#   - When a model is unavailable, downstream stages return safe defaults
#     (anomaly_score=0.0, severity="info", confidence=0.0).
#   - torch is imported lazily — the module is importable without torch.
#   - Rolling embedding buffers (one deque per stream_key) accumulate
#     LogPreprocessor float vectors.  A window is emitted when the buffer
#     is full and the stride interval is met (mirrors SequenceBuffer logic
#     but operates on float embeddings instead of token IDs).
#   - max_stream_keys enforces an LRU-style eviction cap to prevent
#     unbounded memory growth in long-running processes.
#   - The existing Pipeline in src/api/pipeline.py remains the active
#     production path.  This engine is NOT wired to FastAPI yet.

from __future__ import annotations

import datetime
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..modeling.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
from ..modeling.behavior_model import BehaviorModelConfig, SystemBehaviorModel
from ..modeling.severity_classifier import SEVERITY_LABELS, SeverityClassifier
from ..preprocessing.log_preprocessor import LogPreprocessor

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root

# ---------------------------------------------------------------------------
# Optional torch import — required at scoring time
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:                             # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None                                # type: ignore[assignment]


# ---------------------------------------------------------------------------
# EngineResult — named output of the full pipeline
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """
    Structured output of one scored window.

    Produced by :meth:`ProactiveMonitorEngine.score_sequence` and returned
    (when a window is emitted) by :meth:`ProactiveMonitorEngine.process_log`.

    Fields
    ------
    timestamp:
        ISO-8601 string of the moment the window was scored.
    service:
        Stream key / service identifier for the window.
    anomaly_score:
        Reconstruction error from :class:`AnomalyDetector`.
        Higher = more anomalous.  ``0.0`` when the detector is unavailable.
    reconstruction_error:
        Alias of ``anomaly_score`` for output-contract compliance.
    is_anomaly:
        ``True`` when ``anomaly_score`` exceeds the detector's calibrated
        threshold.  ``False`` when the detector is uncalibrated or absent.
    severity:
        One of ``"info"``, ``"warning"``, ``"critical"``
        (lowercase, from :class:`SeverityClassifier`).
    confidence:
        Softmax probability of the predicted severity class (0.0–1.0).
        ``0.0`` when the classifier is unavailable.
    probabilities:
        Full 3-class softmax distribution ``[p_info, p_warning, p_critical]``.
    meta:
        Free-form dict for extra metadata (e.g. latent vector norm).
    """

    timestamp: str
    service: str
    anomaly_score: float
    reconstruction_error: float
    is_anomaly: bool
    severity: str
    confidence: float
    probabilities: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialisable dict suitable for JSON / logging."""
        return {
            "timestamp": self.timestamp,
            "service": self.service,
            "anomaly_score": round(self.anomaly_score, 6),
            "reconstruction_error": round(self.reconstruction_error, 6),
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "probabilities": [round(p, 4) for p in self.probabilities],
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# _EmbeddingBuffer — rolling window of float vectors per stream key
# ---------------------------------------------------------------------------

class _EmbeddingBuffer:
    """
    Lightweight rolling embedding buffer (one per stream key).

    Mirrors the eviction / emit logic of :class:`SequenceBuffer` but
    operates on ``np.ndarray`` float vectors instead of token IDs.

    Emits a window when:
      ``total_events >= window_size``  AND
      ``(total_events - window_size) % stride == 0``

    This produces the first window at event #window_size, then one new
    window every ``stride`` events thereafter.
    """

    def __init__(self, window_size: int, stride: int) -> None:
        self.window_size = window_size
        self.stride = stride
        # deque(maxlen=window_size) auto-discards oldest once full
        self._buf: deque = deque(maxlen=window_size)
        self._total: int = 0

    def push(self, vector: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Add one embedding vector to the buffer.

        Returns
        -------
        list[np.ndarray] | None
            The current window (list of window_size arrays) when an emit
            is due; ``None`` otherwise.
        """
        self._buf.append(vector)
        self._total += 1
        if (
            self._total >= self.window_size
            and (self._total - self.window_size) % self.stride == 0
        ):
            return list(self._buf)   # snapshot; deque is still intact
        return None


# ---------------------------------------------------------------------------
# ProactiveMonitorEngine
# ---------------------------------------------------------------------------

class ProactiveMonitorEngine:
    """
    Phase 7: AIOps Infrastructure Orchestrator.

    Connects the six AI pipeline stages produced in Phases 2-6 into a
    single streaming inference object.

    Pipeline
    --------
    ::

        log_line (str)
            |
            v  LogPreprocessor.process_log()
        embedding [vec_dim]
            |
            v  _EmbeddingBuffer (rolling window, stride-based emit)
        window [window_size, vec_dim]
            |
            v  SystemBehaviorModel.forward() [batch=1, window_size, vec_dim]
        context [1, hidden_dim]
            |
            v  AnomalyDetector.forward()
        AEOutput  (latent [1, latent_dim], error [1])
            |
            v  SeverityClassifier.predict()
        SeverityOutput  (label, class_index, confidence, probabilities)
            |
        EngineResult

    Safety contract
    ---------------
    - A missing or corrupt model file emits a WARNING log and is skipped.
    - A scoring error emits an ERROR log and returns safe defaults.
    - The engine never raises an exception from ``process_log`` or
      ``score_sequence``.

    Parameters
    ----------
    models_dir:
        Directory containing saved model artifacts.  Defaults to
        ``<project_root>/models``.
    window_size:
        Number of consecutive log-line embeddings per scoring window
        (default 20).
    stride:
        Emit a new window every ``stride`` events once the buffer is
        full (default 5).
    max_stream_keys:
        Maximum number of concurrent stream keys tracked.  Oldest key
        is evicted when the limit is reached (default 1000).
    vec_dim:
        Embedding dimensionality used to validate pre-trained preprocessor
        (default 100, must match the saved Word2Vec model).
    alert_buffer_size:
        Capacity of the ring buffer that holds generated alerts
        (default 500).
    """

    # Expected model file names inside models_dir
    _PREPROCESSOR_FILE = "word2vec.model"
    _BEHAVIOR_FILE = "behavior_model.pt"
    _DETECTOR_FILE = "anomaly_detector.pt"
    _CLASSIFIER_FILE = "severity_classifier.pt"

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        window_size: int = 20,
        stride: int = 5,
        max_stream_keys: int = 1000,
        vec_dim: int = 100,
        alert_buffer_size: int = 500,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else _ROOT / "models"
        self.window_size = window_size
        self.stride = stride
        self.max_stream_keys = max_stream_keys
        self.vec_dim = vec_dim

        # Pipeline stage references — populated by initialize_models()
        self._preprocessor: Optional[LogPreprocessor] = None
        self._behavior_model: Optional[SystemBehaviorModel] = None
        self._anomaly_detector: Optional[AnomalyDetector] = None
        self._severity_classifier: Optional[SeverityClassifier] = None

        # Rolling embedding buffers: stream_key -> _EmbeddingBuffer
        # Kept insertion-ordered so the oldest key is easily identified
        # for LRU eviction.
        self._buffers: Dict[str, _EmbeddingBuffer] = {}

        # Recent-alert ring buffer
        self._alert_buffer: deque = deque(maxlen=alert_buffer_size)

        # Initialisation state
        self._loaded: bool = False

        # Counters for metrics_snapshot()
        self._events_total: int = 0
        self._windows_total: int = 0
        self._anomalies_total: int = 0

        logger.debug(
            "ProactiveMonitorEngine: window_size=%d stride=%d "
            "max_stream_keys=%d models_dir=%s",
            self.window_size, self.stride,
            self.max_stream_keys, self.models_dir,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize_models(self) -> None:
        """
        Load all AI pipeline models from :attr:`models_dir`.

        Each model is loaded independently.  A missing file or loading
        error emits a WARNING but does NOT raise — the engine continues
        with that stage disabled and returns safe defaults at inference
        time.

        Loaded models
        -------------
        * ``word2vec.model``          → :class:`LogPreprocessor`
        * ``behavior_model.pt``       → :class:`SystemBehaviorModel`
        * ``anomaly_detector.pt``     → :class:`AnomalyDetector`
        * ``severity_classifier.pt``  → :class:`SeverityClassifier`
        """
        t0 = time.perf_counter()
        d = self.models_dir
        logger.info("ProactiveMonitorEngine: loading models from %s", d)

        # 1 — LogPreprocessor (gensim Word2Vec)
        self._preprocessor = self._load_preprocessor(d / self._PREPROCESSOR_FILE)

        # 2 — SystemBehaviorModel (torch)
        self._behavior_model = self._load_behavior_model(d / self._BEHAVIOR_FILE)

        # 3 — AnomalyDetector (torch)
        self._anomaly_detector = self._load_anomaly_detector(d / self._DETECTOR_FILE)

        # 4 — SeverityClassifier (torch)
        self._severity_classifier = self._load_severity_classifier(
            d / self._CLASSIFIER_FILE
        )

        elapsed = time.perf_counter() - t0
        self._loaded = True
        logger.info(
            "ProactiveMonitorEngine: initialisation complete in %.3fs | "
            "preprocessor=%s behavior=%s detector=%s classifier=%s",
            elapsed,
            self._preprocessor is not None,
            self._behavior_model is not None,
            self._anomaly_detector is not None,
            self._severity_classifier is not None,
        )

    # Alias kept for backward compatibility with existing callers and tests
    def load_models(self) -> None:
        """Alias for :meth:`initialize_models`."""
        self.initialize_models()

    # ------------------------------------------------------------------
    # Individual model loaders (each wraps errors with warn-and-continue)
    # ------------------------------------------------------------------

    def _load_preprocessor(self, path: Path) -> Optional[LogPreprocessor]:
        if not path.exists():
            logger.warning(
                "LogPreprocessor model not found at %s — "
                "embedding stage will be unavailable.",
                path,
            )
            return None
        try:
            prep = LogPreprocessor(vec_dim=self.vec_dim)
            prep.load(path)
            logger.info(
                "LogPreprocessor loaded from %s (vec_dim=%d)",
                path, prep.vec_dim,
            )
            return prep
        except Exception as exc:
            logger.warning(
                "Failed to load LogPreprocessor from %s: %s — "
                "embedding stage disabled.",
                path, exc,
            )
            return None

    def _load_behavior_model(self, path: Path) -> Optional[SystemBehaviorModel]:
        if not _TORCH_AVAILABLE:
            logger.warning(
                "torch not installed — SystemBehaviorModel unavailable."
            )
            return None
        if not path.exists():
            logger.warning(
                "SystemBehaviorModel checkpoint not found at %s — "
                "behavior stage will be unavailable.",
                path,
            )
            return None
        try:
            model = SystemBehaviorModel.load(path)
            model.eval()
            logger.info(
                "SystemBehaviorModel loaded from %s (hidden_dim=%d)",
                path, model.hidden_dim,
            )
            return model
        except Exception as exc:
            logger.warning(
                "Failed to load SystemBehaviorModel from %s: %s — "
                "behavior stage disabled.",
                path, exc,
            )
            return None

    def _load_anomaly_detector(self, path: Path) -> Optional[AnomalyDetector]:
        if not _TORCH_AVAILABLE:
            logger.warning(
                "torch not installed — AnomalyDetector unavailable."
            )
            return None
        if not path.exists():
            logger.warning(
                "AnomalyDetector checkpoint not found at %s — "
                "anomaly stage will be unavailable.",
                path,
            )
            return None
        try:
            detector = AnomalyDetector.load(path)
            detector.eval()
            logger.info(
                "AnomalyDetector loaded from %s "
                "(latent_dim=%d, calibrated=%s, threshold=%.6f)",
                path, detector.latent_dim,
                detector.is_calibrated, detector.threshold,
            )
            return detector
        except Exception as exc:
            logger.warning(
                "Failed to load AnomalyDetector from %s: %s — "
                "anomaly stage disabled.",
                path, exc,
            )
            return None

    def _load_severity_classifier(self, path: Path) -> Optional[SeverityClassifier]:
        if not _TORCH_AVAILABLE:
            logger.warning(
                "torch not installed — SeverityClassifier unavailable."
            )
            return None
        if not path.exists():
            logger.warning(
                "SeverityClassifier checkpoint not found at %s — "
                "severity classification will default to 'info'.",
                path,
            )
            return None
        try:
            clf = SeverityClassifier.load(path)
            clf.eval()
            logger.info(
                "SeverityClassifier loaded from %s "
                "(input_dim=%d, hidden_dim=%d)",
                path, clf.input_dim, clf.hidden_dim,
            )
            return clf
        except Exception as exc:
            logger.warning(
                "Failed to load SeverityClassifier from %s: %s — "
                "severity defaults to 'info'.",
                path, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process_log(
        self,
        log_line: str,
        stream_key: str = "default",
        service: str = "unknown",
        timestamp: Optional[str] = None,
    ) -> Optional[EngineResult]:
        """
        Feed one raw log line through the full AI pipeline.

        The log line is embedded by :class:`LogPreprocessor`, added to
        the rolling buffer for ``stream_key``, and — when the window is
        full and the stride interval is met — scored by
        :meth:`score_sequence`.

        Parameters
        ----------
        log_line:
            Raw log message string.
        stream_key:
            Logical stream identifier (e.g. service name or session id).
            Each key maintains an independent rolling buffer.
        service:
            Human-readable service label included in :class:`EngineResult`.
        timestamp:
            ISO-8601 timestamp string; defaults to the current UTC time.

        Returns
        -------
        EngineResult | None
            :class:`EngineResult` when a window was scored; ``None`` when
            the buffer has not yet accumulated enough events to emit.
        """
        self._events_total += 1
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()

        # -- Stage 1: embed -----------------------------------------------
        vector = self._embed(log_line)
        if vector is None:
            return None     # preprocessor unavailable — cannot proceed

        # -- Rolling buffer -----------------------------------------------
        buf = self._get_buffer(stream_key)
        window_embeddings = buf.push(vector)
        if window_embeddings is None:
            return None     # window not yet full / stride not met

        # -- Stages 2-4: behavior → anomaly → severity --------------------
        self._windows_total += 1
        result = self._score_window(window_embeddings, service=service, timestamp=ts)
        if result.is_anomaly:
            self._anomalies_total += 1
        return result

    def process_batch(
        self,
        log_lines: List[str],
        stream_key: str = "default",
        service: str = "unknown",
    ) -> List[Optional[EngineResult]]:
        """
        Feed a list of log lines through the pipeline sequentially.

        Parameters
        ----------
        log_lines:
            Ordered list of raw log message strings.
        stream_key:
            Stream key shared by all lines in this batch.
        service:
            Human-readable service label.

        Returns
        -------
        list[EngineResult | None]
            One entry per input log line.  ``None`` entries indicate that
            the buffer had not accumulated a full window at that position.
        """
        return [
            self.process_log(line, stream_key=stream_key, service=service)
            for line in log_lines
        ]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_sequence(
        self,
        sequence_tensor: "torch.Tensor",
        service: str = "unknown",
        timestamp: Optional[str] = None,
    ) -> EngineResult:
        """
        Run the behavior model → anomaly detector → severity classifier
        chain directly on a pre-formed sequence tensor.

        This method bypasses the embedding and buffer stages and is
        intended for use with pre-embedded tensors (e.g. from a
        :class:`LogDataset`).

        Parameters
        ----------
        sequence_tensor : torch.FloatTensor
            Shape ``[window_size, vec_dim]``.  Stacked embedding vectors
            for one window.
        service:
            Human-readable service label.
        timestamp:
            ISO-8601 timestamp string; defaults to current UTC time.

        Returns
        -------
        EngineResult
            Always returns a valid :class:`EngineResult`; falls back to
            safe defaults on any scoring error.
        """
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        if not _TORCH_AVAILABLE:
            return self._fallback_result(service=service, timestamp=ts,
                                         reason="torch unavailable")
        embeddings = [sequence_tensor[i].numpy() for i in range(sequence_tensor.shape[0])]
        return self._score_window(embeddings, service=service, timestamp=ts)

    def _score_window(
        self,
        embeddings: List[np.ndarray],
        service: str,
        timestamp: str,
    ) -> EngineResult:
        """
        Internal scoring entry point.  Accepts raw embedding list.
        All exceptions are caught and logged; safe defaults are returned.
        """
        try:
            return self._run_pipeline(embeddings, service=service, timestamp=timestamp)
        except Exception as exc:
            logger.error(
                "Scoring error for service=%s: %s", service, exc, exc_info=True
            )
            return self._fallback_result(service=service, timestamp=timestamp,
                                         reason=str(exc))

    def _run_pipeline(
        self,
        embeddings: List[np.ndarray],
        service: str,
        timestamp: str,
    ) -> EngineResult:
        """
        Execute Stages 3-5 (behavior → anomaly → severity) on a window.

        Returns
        -------
        EngineResult
            Populated result.  Stages that are unavailable produce safe
            defaults and a warning is logged.
        """
        if not _TORCH_AVAILABLE:
            return self._fallback_result(service=service, timestamp=timestamp,
                                         reason="torch unavailable")

        # Stack embeddings into [1, window_size, vec_dim] tensor
        window_arr = np.stack(embeddings, axis=0)           # [W, D]
        x = torch.tensor(window_arr, dtype=torch.float32).unsqueeze(0)  # [1, W, D]

        # -- Stage 3: SystemBehaviorModel ---------------------------------
        if self._behavior_model is None:
            logger.warning("SystemBehaviorModel not loaded; returning fallback.")
            return self._fallback_result(service=service, timestamp=timestamp,
                                         reason="behavior model unavailable")

        self._behavior_model.eval()
        with torch.no_grad():
            context = self._behavior_model(x)               # [1, hidden_dim]

        # -- Stage 4: AnomalyDetector ------------------------------------
        if self._anomaly_detector is None:
            logger.warning("AnomalyDetector not loaded; returning fallback.")
            return self._fallback_result(service=service, timestamp=timestamp,
                                         reason="anomaly detector unavailable")

        self._anomaly_detector.eval()
        with torch.no_grad():
            ae_out = self._anomaly_detector(context)        # AEOutput

        recon_error = float(ae_out.error[0].item())
        is_anomaly = self._anomaly_detector.is_anomaly(recon_error)
        latent = ae_out.latent                              # [1, latent_dim]

        # -- Stage 5: SeverityClassifier ---------------------------------
        if self._severity_classifier is None:
            logger.warning(
                "SeverityClassifier not loaded; defaulting severity to 'info'."
            )
            return EngineResult(
                timestamp=timestamp,
                service=service,
                anomaly_score=recon_error,
                reconstruction_error=recon_error,
                is_anomaly=is_anomaly,
                severity="info",
                confidence=0.0,
                probabilities=[1.0, 0.0, 0.0],
                meta={"latent_norm": float(latent.norm().item())},
            )

        sev_out = self._severity_classifier.predict(
            latent_vector=latent[0],            # [latent_dim]
            reconstruction_error=recon_error,
        )

        return EngineResult(
            timestamp=timestamp,
            service=service,
            anomaly_score=recon_error,
            reconstruction_error=recon_error,
            is_anomaly=is_anomaly,
            severity=sev_out.label,
            confidence=sev_out.confidence,
            probabilities=sev_out.probabilities,
            meta={"latent_norm": float(latent[0].norm().item())},
        )

    # ------------------------------------------------------------------
    # Alert formatting
    # ------------------------------------------------------------------

    def generate_alert(self, result: Optional[EngineResult]) -> Optional[dict]:
        """
        Format an :class:`EngineResult` as an alert dict when anomalous.

        Parameters
        ----------
        result:
            Output of :meth:`process_log` or :meth:`score_sequence`.

        Returns
        -------
        dict | None
            Alert dict when ``result.is_anomaly`` is ``True``; otherwise
            ``None``.  The dict is also appended to the internal alert
            ring buffer.
        """
        if result is None or not result.is_anomaly:
            return None

        alert = {
            "timestamp": result.timestamp,
            "service": result.service,
            "severity": result.severity.upper(),
            "anomaly_score": result.anomaly_score,
            "confidence": result.confidence,
            "message": (
                f"Anomaly detected in service '{result.service}' — "
                f"severity={result.severity.upper()} "
                f"score={result.anomaly_score:.4f} "
                f"confidence={result.confidence:.2%}"
            ),
        }
        self._alert_buffer.append(alert)
        return alert

    # ------------------------------------------------------------------
    # Backward-compatible stub methods
    # ------------------------------------------------------------------

    def process_event(self, event: dict) -> dict:
        """
        Backward-compatible entry point: extract ``message`` from an event
        dict and delegate to :meth:`process_log`.

        Compatible with the Phase 1 stub interface.

        Parameters
        ----------
        event:
            Dict with optional keys ``message`` / ``log_line``,
            ``service``, ``timestamp``.

        Returns
        -------
        dict
            Keys: ``window_emitted`` (bool), ``risk_result`` (dict | None),
            ``alert`` (dict | None).
        """
        log_line = event.get("message") or event.get("log_line") or ""
        service = event.get("service", "unknown")
        timestamp = event.get("timestamp", "")
        stream_key = service or "default"

        result = self.process_log(
            log_line,
            stream_key=stream_key,
            service=service,
            timestamp=timestamp or None,
        )
        alert = self.generate_alert(result) if result is not None else None
        return {
            "window_emitted": result is not None,
            "risk_result": result.to_dict() if result is not None else None,
            "alert": alert,
        }

    def recent_alerts(self) -> List[dict]:
        """Return the most-recent alerts from the ring buffer (newest last)."""
        return list(self._alert_buffer)

    def metrics_snapshot(self) -> Dict:
        """
        Return the current state of each pipeline component.

        Suitable for a ``/pipeline/status`` endpoint in a future phase.

        Returns
        -------
        dict
            Component availability flags and accumulated counters.
        """
        return {
            "loaded": self._loaded,
            "models": {
                "preprocessor": self._preprocessor is not None,
                "behavior_model": self._behavior_model is not None,
                "anomaly_detector": self._anomaly_detector is not None,
                "severity_classifier": self._severity_classifier is not None,
            },
            "config": {
                "window_size": self.window_size,
                "stride": self.stride,
                "max_stream_keys": self.max_stream_keys,
                "vec_dim": self.vec_dim,
                "models_dir": str(self.models_dir),
            },
            "counters": {
                "events_total": self._events_total,
                "windows_total": self._windows_total,
                "anomalies_total": self._anomalies_total,
                "active_streams": len(self._buffers),
                "alert_buffer_size": len(self._alert_buffer),
            },
            "anomaly_threshold": (
                self._anomaly_detector.threshold
                if self._anomaly_detector is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, log_line: str) -> Optional[np.ndarray]:
        """Embed a single raw log line; return None if preprocessor absent."""
        if self._preprocessor is None:
            return None
        try:
            return self._preprocessor.process_log(log_line)
        except Exception as exc:
            logger.error("Embedding error: %s", exc)
            return None

    def _get_buffer(self, stream_key: str) -> _EmbeddingBuffer:
        """
        Return (or create) the embedding buffer for ``stream_key``.
        Evicts the oldest key when :attr:`max_stream_keys` is reached.
        """
        if stream_key not in self._buffers:
            if len(self._buffers) >= self.max_stream_keys:
                # Evict oldest (dict insertion order — Python 3.7+)
                oldest = next(iter(self._buffers))
                del self._buffers[oldest]
                logger.debug(
                    "ProactiveMonitorEngine: evicted buffer for stream_key=%s",
                    oldest,
                )
            self._buffers[stream_key] = _EmbeddingBuffer(
                window_size=self.window_size,
                stride=self.stride,
            )
        return self._buffers[stream_key]

    @staticmethod
    def _fallback_result(
        service: str,
        timestamp: str,
        reason: str = "",
    ) -> EngineResult:
        """Safe default EngineResult for error / unavailable model cases."""
        return EngineResult(
            timestamp=timestamp,
            service=service,
            anomaly_score=0.0,
            reconstruction_error=0.0,
            is_anomaly=False,
            severity="info",
            confidence=0.0,
            probabilities=[1.0, 0.0, 0.0],
            meta={"fallback_reason": reason},
        )
