# src/runtime/pipeline_v2.py
# Phase 6 — Runtime Pipeline (v2)
#
# V2Pipeline orchestrates the four-stage ML inference pipeline:
#
#   raw log string
#       → _V2LogTokenizer  (TemplateMiner-compatible generalisation → template_id → token_id)
#       → Word2Vec KeyedVectors lookup  (str(token_id) → float32[vec_dim])
#       → rolling window buffer
#       → SystemBehaviorModel (LSTM context vector)
#       → AnomalyDetector (reconstruction error + anomaly flag)
#       → SeverityClassifier (Info / Warning / Critical)
#       → V2Result
#
# Embedding mode (token-ID, consistent with training)
# ---------------------------------------------------
# Training used sequences_train.parquet whose 'tokens' column contains integer
# token_ids (template_id + 2).  Word2Vec was trained on str(token_id) sequences.
# Inference must therefore follow the same path:
#
#   raw_log  →  apply _SUBS substitution patterns (identical to TemplateMiner)
#            →  normalised template text
#            →  lookup in templates.csv → template_id   (UNK_ID=1 if absent)
#            →  token_id = template_id + 2             (EventTokenizer._OFFSET)
#            →  wv[str(token_id)]                      (zero vector if OOV)
#            →  float32[vec_dim]
#
# Design:
#   - Completely isolated from the v1 runtime (inference_engine.py).
#   - All four models are loaded lazily on the first call to load_models().
#   - Model paths default to the standard v2 artifact layout but are fully
#     configurable via V2PipelineConfig.
#   - The rolling window buffer is per-stream (keyed by service+session_id),
#     implemented as a collections.deque.
#   - A V2Result is returned for every call to process_log(), but
#     window_emitted=False when the buffer hasn't accumulated enough events.
#   - No threads / asyncio — designed to be called from a sync or async
#     context (FastAPI background task or direct call).

from __future__ import annotations

import collections
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _V2LogTokenizer — single-string TemplateMiner + EventTokenizer for inference
# ---------------------------------------------------------------------------

class _V2LogTokenizer:
    """
    Maps a raw log string to an integer token_id for v2 inference.

    Replicates the two-step encoding used during data preprocessing:
      1. TemplateMiner._generalize() — apply the same 9-step substitution
         pipeline to normalise the log message into a template string.
      2. EventTokenizer encoding — look up the template string in templates.csv
         to get template_id, then token_id = template_id + _OFFSET (2).

    If the normalised string is not in the template vocabulary the UNK token
    id (1) is returned, which maps to a zero embedding vector at inference time.

    Parameters
    ----------
    templates_csv_path:
        Path to data/intermediate/templates.csv.  Must contain columns
        ``template_id`` (int) and ``template_text`` (str).
    """

    # Regex substitution patterns — identical to TemplateMiner._SUBS so that
    # the normalised string matches the pre-computed template_text values.
    _SUBS: list[tuple] = [
        (re.compile(r"blk_-?\d+"),                                     "<BLK>"),
        (re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+"),  "<TS>"),
        (re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?"), "<IP>"),
        (re.compile(r"\d{4}\.\d{2}\.\d{2}"),                           "<DATE>"),
        (re.compile(r"R\d+(?:-[A-Z\d]+)+(?::[A-Z]\d+-[A-Z]\d+)?"),    "<NODE>"),
        (re.compile(r"/[a-zA-Z0-9_./-]+"),                             "<PATH>"),
        (re.compile(r"\b[0-9a-f]{8,}\b"),                              "<HEX>"),
        (re.compile(r"\b\d+\b"),                                        "<NUM>"),
        (re.compile(r"\s+"),                                            " "),
    ]

    # Token-id constants — must match EventTokenizer
    UNK_ID: int = 1
    _OFFSET: int = 2   # token_id = template_id + _OFFSET

    def __init__(self, templates_csv_path: Path) -> None:
        import pandas as pd
        df = pd.read_csv(
            templates_csv_path,
            usecols=["template_id", "template_text"],
        )
        self._template_to_id: dict[str, int] = dict(
            zip(df["template_text"], df["template_id"].astype(int))
        )
        logger.info(
            "_V2LogTokenizer: loaded %d templates from %s",
            len(self._template_to_id),
            templates_csv_path,
        )

    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of known templates (excludes PAD/UNK specials)."""
        return len(self._template_to_id)

    def generalize(self, text: str) -> str:
        """
        Apply the TemplateMiner substitution pipeline to a single string.

        This is the single-string equivalent of TemplateMiner._generalize()
        and produces the same normalised template text.
        """
        for pattern, replacement in self._SUBS:
            text = pattern.sub(replacement, text)
        return text.strip()

    def log_to_token_id(self, raw_log: str) -> int:
        """
        Convert a raw log string to a token_id.

        Steps
        -----
        1. Normalise: apply _SUBS substitutions → template text.
        2. Lookup:    template text → template_id from templates.csv.
        3. Encode:    token_id = template_id + _OFFSET (2).

        Returns UNK_ID (1) when the normalised text is not in the vocabulary.
        """
        normalised = self.generalize(raw_log)
        template_id = self._template_to_id.get(normalised)
        if template_id is None:
            return self.UNK_ID
        return template_id + self._OFFSET

# ---------------------------------------------------------------------------
# Paths — defaults point to models/ layout from REFACTOR_PLAN.md
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_W2V = _PROJECT_ROOT / "models" / "embeddings" / "word2vec.model"
_DEFAULT_BEHAVIOR = _PROJECT_ROOT / "models" / "behavior" / "behavior_model.pt"
_DEFAULT_AE = _PROJECT_ROOT / "models" / "anomaly" / "anomaly_detector.pt"
_DEFAULT_SEV = _PROJECT_ROOT / "models" / "severity" / "severity_classifier.pt"
_DEFAULT_TEMPLATES = _PROJECT_ROOT / "data" / "intermediate" / "templates.csv"


# ---------------------------------------------------------------------------
# V2PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class V2PipelineConfig:
    """
    Configuration for the v2 inference pipeline.

    Parameters
    ----------
    window_size:
        Number of consecutive log embeddings per sequence window.
    w2v_model_path:
        Path to the gensim Word2Vec model saved by train_embeddings.py.
    behavior_model_path:
        Path to the SystemBehaviorModel checkpoint.
    anomaly_model_path:
        Path to the AnomalyDetector checkpoint.
    severity_model_path:
        Path to the SeverityClassifier checkpoint.
    templates_path:
        Path to data/intermediate/templates.csv — the template vocabulary
        used by _V2LogTokenizer to map raw logs to token_ids at inference time.
        Must be the same vocabulary used when producing events_tokenized.parquet.
    """
    window_size: int = 10
    w2v_model_path: Path = field(default_factory=lambda: _DEFAULT_W2V)
    behavior_model_path: Path = field(default_factory=lambda: _DEFAULT_BEHAVIOR)
    anomaly_model_path: Path = field(default_factory=lambda: _DEFAULT_AE)
    severity_model_path: Path = field(default_factory=lambda: _DEFAULT_SEV)
    templates_path: Path = field(default_factory=lambda: _DEFAULT_TEMPLATES)


# ---------------------------------------------------------------------------
# V2Result
# ---------------------------------------------------------------------------

@dataclass
class V2Result:
    """
    Output of a single V2Pipeline.process_log() call.

    Attributes
    ----------
    window_emitted:
        True when a full window was scored.  False when the buffer is still
        accumulating events.
    stream_key:
        ``"<service>/<session_id>"`` identifier for the event stream.
    anomaly_score:
        Reconstruction error from the Autoencoder (higher = more anomalous).
        ``None`` when window_emitted=False.
    is_anomaly:
        True when anomaly_score > calibrated threshold.
        ``None`` when window_emitted=False.
    severity:
        ``"info"`` / ``"warning"`` / ``"critical"``.
        ``None`` when window_emitted=False.
    severity_confidence:
        Probability of the predicted severity class (0.0–1.0).
        ``None`` when window_emitted=False.
    severity_probabilities:
        Full 3-class softmax distribution ``[p_info, p_warning, p_critical]``.
        ``None`` when window_emitted=False.
    """
    window_emitted: bool
    stream_key: str
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    severity: Optional[str] = None
    severity_confidence: Optional[float] = None
    severity_probabilities: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# V2Pipeline
# ---------------------------------------------------------------------------

class V2Pipeline:
    """
    v2 four-stage inference pipeline.

    Parameters
    ----------
    cfg:
        Pipeline configuration.  If omitted, default paths and settings are used.

    Notes
    -----
    Call :meth:`load_models` before :meth:`process_log`.
    All four models are required; an error is raised if any artifact is missing.
    """

    def __init__(self, cfg: Optional[V2PipelineConfig] = None) -> None:
        self.cfg = cfg or V2PipelineConfig()

        # Model instances — set by load_models()
        self._preprocessor = None   # LogPreprocessor (kept for loading; wv extracted from it)
        self._tokenizer: Optional[_V2LogTokenizer] = None  # raw log → token_id
        self._wv = None             # gensim KeyedVectors for str(token_id) → float32[D]
        self._vec_dim: Optional[int] = None
        self._behavior = None       # SystemBehaviorModel
        self._detector = None       # AnomalyDetector
        self._classifier = None     # SeverityClassifier

        # Per-stream rolling window buffers: stream_key → deque of embeddings
        self._buffers: Dict[str, Deque[np.ndarray]] = {}

        self._loaded: bool = False

        logger.info(
            "V2Pipeline created (window_size=%d)", self.cfg.window_size
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True after load_models() completes successfully."""
        return self._loaded

    def load_models(self) -> None:
        """
        Load all four v2 model artifacts from disk.

        Raises
        ------
        FileNotFoundError
            If any model artifact is missing.
        ImportError
            If gensim or torch is not installed.
        """
        cfg = self.cfg

        # --- 1. Word2Vec model + KeyedVectors ---
        if not cfg.w2v_model_path.exists():
            raise FileNotFoundError(
                f"Word2Vec model not found: {cfg.w2v_model_path}\n"
                "Train it with: python -m training.train_embeddings"
            )
        from ..preprocessing.log_preprocessor import LogPreprocessor
        preprocessor = LogPreprocessor()
        preprocessor.load(cfg.w2v_model_path)
        self._preprocessor = preprocessor
        # Extract KeyedVectors for direct str(token_id) → vector lookups.
        # The Word2Vec model was trained on token-ID sequences (not raw text),
        # so inference must use the same vocabulary.
        self._wv = preprocessor._model.wv
        self._vec_dim = preprocessor.vec_dim
        logger.info("Loaded Word2Vec: vec_dim=%d vocab_size=%d", preprocessor.vec_dim, len(self._wv))

        # --- 2. Template tokenizer (raw log → token_id) ---
        if not cfg.templates_path.exists():
            raise FileNotFoundError(
                f"Template vocabulary not found: {cfg.templates_path}\n"
                "Required for v2 inference: run scripts/data_pipeline/stage_02_templates.py "
                "to generate data/intermediate/templates.csv."
            )
        self._tokenizer = _V2LogTokenizer(cfg.templates_path)
        logger.info(
            "Loaded template vocabulary: %d templates", self._tokenizer.vocab_size
        )

        # --- 2. SystemBehaviorModel ---
        if not cfg.behavior_model_path.exists():
            raise FileNotFoundError(
                f"Behavior model not found: {cfg.behavior_model_path}\n"
                "Train it with: python -m training.train_behavior_model"
            )
        from ..modeling.behavior_model import SystemBehaviorModel
        self._behavior = SystemBehaviorModel.load(cfg.behavior_model_path)
        self._behavior.eval()
        logger.info(
            "Loaded SystemBehaviorModel: hidden_dim=%d",
            self._behavior.hidden_dim,
        )

        # --- 3. AnomalyDetector ---
        if not cfg.anomaly_model_path.exists():
            raise FileNotFoundError(
                f"AnomalyDetector not found: {cfg.anomaly_model_path}\n"
                "Train it with: python -m training.train_autoencoder"
            )
        from ..modeling.anomaly_detector import AnomalyDetector
        self._detector = AnomalyDetector.load(cfg.anomaly_model_path)
        self._detector.eval()
        logger.info(
            "Loaded AnomalyDetector: latent_dim=%d threshold=%.6f calibrated=%s",
            self._detector.latent_dim,
            self._detector.threshold,
            self._detector.is_calibrated,
        )

        # --- 4. SeverityClassifier ---
        if not cfg.severity_model_path.exists():
            raise FileNotFoundError(
                f"SeverityClassifier not found: {cfg.severity_model_path}\n"
                "Train it with: python -m training.train_severity_model"
            )
        from ..modeling.severity_classifier import SeverityClassifier
        self._classifier = SeverityClassifier.load(cfg.severity_model_path)
        logger.info(
            "Loaded SeverityClassifier: input_dim=%d",
            self._classifier.input_dim,
        )

        self._loaded = True
        logger.info("V2Pipeline: all models loaded successfully")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_buffer(self, stream_key: str) -> Deque[np.ndarray]:
        if stream_key not in self._buffers:
            self._buffers[stream_key] = collections.deque(
                maxlen=self.cfg.window_size
            )
        return self._buffers[stream_key]

    def process_log(
        self,
        raw_log: str,
        service: str = "default",
        session_id: str = "",
        timestamp: float = 0.0,
    ) -> V2Result:
        """
        Process one raw log string through the full v2 pipeline.

        Parameters
        ----------
        raw_log:
            Raw log message text (message field only or full log line).
        service:
            Service or component name (used as stream key prefix).
        session_id:
            Session / block identifier (used as stream key suffix).
        timestamp:
            Event Unix epoch (stored in result but not used for windowing).

        Returns
        -------
        V2Result
            Contains anomaly score and severity when a full window is ready.

        Raises
        ------
        RuntimeError
            If load_models() has not been called yet.
        """
        if not self._loaded:
            raise RuntimeError(
                "V2Pipeline.process_log called before load_models(). "
                "Call pipeline.load_models() first."
            )

        stream_key = f"{service}/{session_id}" if session_id else service
        buf = self._get_buffer(stream_key)

        # Stage 1: raw log → token_id → Word2Vec embedding
        # Uses the same representation as training:
        #   raw_log → TemplateMiner-compatible generalisation → template_id → token_id
        #   token_id → str(token_id) → wv lookup → float32[vec_dim]
        token_id = self._tokenizer.log_to_token_id(raw_log)
        tok_str = str(token_id)
        if tok_str in self._wv:
            embedding: np.ndarray = self._wv[tok_str].astype(np.float32)
        else:
            # Token is OOV (e.g., UNK or template absent from training corpus).
            # Zero vector is the correct fallback: LSTM has seen this pattern
            # during training on sessions that contained OOV tokens.
            embedding = np.zeros(self._vec_dim, dtype=np.float32)
        buf.append(embedding)

        # Wait until the buffer is full
        if len(buf) < self.cfg.window_size:
            return V2Result(window_emitted=False, stream_key=stream_key)

        # Assemble window tensor [1, window_size, vec_dim]
        import torch
        window = np.stack(list(buf), axis=0).astype(np.float32)   # [T, D]
        x = torch.from_numpy(window).unsqueeze(0)                  # [1, T, D]

        # Stage 2: LSTM context vector [1, hidden_dim]
        with torch.no_grad():
            context = self._behavior(x)

        # Stage 3: Autoencoder anomaly scoring
        ae_out = self._detector(context)
        score = float(ae_out.error[0].item())
        is_anomaly = self._detector.is_anomaly(score)

        # Stage 4: Severity classification
        sev_out = self._classifier.predict(ae_out.latent[0], ae_out.error[0])

        return V2Result(
            window_emitted=True,
            stream_key=stream_key,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            severity=sev_out.label,
            severity_confidence=sev_out.confidence,
            severity_probabilities=sev_out.probabilities,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def model_info(self) -> dict:
        """Return a summary dict for health checks and /health endpoint."""
        return {
            "loaded": self._loaded,
            "window_size": self.cfg.window_size,
            "active_streams": len(self._buffers),
            "embedding_mode": "token-id",
            "template_vocab_size": (
                self._tokenizer.vocab_size if self._tokenizer else None
            ),
            "w2v_vocab_size": len(self._wv) if self._wv is not None else None,
            "models": {
                "word2vec": str(self.cfg.w2v_model_path),
                "templates": str(self.cfg.templates_path),
                "behavior_model": str(self.cfg.behavior_model_path),
                "anomaly_detector": str(self.cfg.anomaly_model_path),
                "severity_classifier": str(self.cfg.severity_model_path),
            },
        }
