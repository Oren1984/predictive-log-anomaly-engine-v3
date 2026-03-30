# src/runtime/inference_engine.py

# Purpose: Implement the InferenceEngine class responsible for orchestrating the rolling buffer,
# model scoring, and thresholding for live log streams.
# This includes loading necessary artifacts,
# scoring sequences using the baseline and transformer models,
# applying decision thresholds,
# and building structured RiskResult objects for emitted windows.

# Input: - InferenceEngine: A class that manages the inference process for log data, 
#          including buffering, scoring, and thresholding.
#        - load_artifacts: A method to load model artifacts, vocabularies, and thresholds from disk.
#        - ingest: A method to feed log events into the buffer and trigger scoring when appropriate.
#        - score_baseline: A method to score a sequence using the baseline model.

# Output: - InferenceEngine: An instance of the InferenceEngine class that 
#           can be used to perform inference on log data streams.
#         - RiskResult: Structured results of anomaly scoring for emitted windows.

# Used by: Other components of the runtime stage that need to perform inference on log data, 
# such as the main application loop that feeds log events into the InferenceEngine 
# and processes the resulting RiskResults for alerting or further analysis.

"""Stage 5 — Runtime: inference engine (buffer + model scoring + thresholding)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..data_layer.models import LogEvent
from ..modeling.baseline.extractor import BaselineFeatureExtractor
from ..modeling.baseline.model import BaselineAnomalyModel
from ..modeling.transformer.model import NextTokenTransformerModel
from ..modeling.transformer.scorer import AnomalyScorer
from ..sequencing.models import Sequence
from .sequence_buffer import SequenceBuffer
from .types import RiskResult

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root


class InferenceEngine:
    """
    Orchestrate SequenceBuffer + model scoring + thresholding for live streams.

    Parameters
    ----------
    mode            : "baseline" | "transformer" | "ensemble"
    window_size     : rolling window length (tokens)
    stride          : emit interval once window is full
    max_stream_keys : buffer LRU cap
    root            : project root (auto-detected when None)
    """

    VALID_MODES = ("baseline", "transformer", "ensemble")

    def __init__(
        self,
        mode: str = "baseline",
        window_size: int = 50,
        stride: int = 10,
        max_stream_keys: int = 5000,
        root: Optional[Path] = None,
        use_runtime_thresholds: bool = False,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got {mode!r}"
            )
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.root = Path(root) if root is not None else _ROOT
        self.use_runtime_thresholds = use_runtime_thresholds

        self.buffer = SequenceBuffer(window_size, stride, max_stream_keys)

        # Runtime artifacts (populated by load_artifacts)
        self._vocab: dict[str, str] = {}            # str(token_id) -> template_text
        self._templates: dict[str, str] = {}        # str(template_id) -> template_text
        # Original thresholds (also serve as normalization denominators in ensemble)
        self._threshold_baseline: float = 0.33
        self._threshold_transformer: float = 0.034
        self._threshold_ensemble: float = 1.0       # normalized; 1.0 = either model votes

        # Runtime overrides for baseline/transformer *decision* only.
        # Ensemble normalization always uses the original _threshold_* above so
        # that the calibrated _threshold_ensemble score space remains consistent.
        self._rt_threshold_baseline: Optional[float] = None
        self._rt_threshold_transformer: Optional[float] = None

        self._extractor: Optional[BaselineFeatureExtractor] = None
        self._baseline_model: Optional[BaselineAnomalyModel] = None
        self._scorer: Optional[AnomalyScorer] = None

        self._artifacts_loaded: bool = False

        # Demo / fallback behaviour
        # When demo_mode=True and scoring fails (missing/incompatible model
        # artifacts), fallback_score is returned instead of crashing.
        # In production (demo_mode=False) the fallback returns 0.0 so no
        # spurious alerts are fired.
        self.demo_mode: bool = False
        self.fallback_score: float = 2.0

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def load_artifacts(self) -> None:
        """Load all vocabulary, thresholds, and model weights."""
        root = self.root
        logger.info("Loading artifacts | mode=%s | root=%s", self.mode, root)

        # Vocabulary and template text
        vocab_path = root / "artifacts" / "vocab.json"
        templates_path = root / "artifacts" / "templates.json"

        if vocab_path.exists():
            with open(vocab_path, encoding="utf-8") as fh:
                self._vocab = json.load(fh)
            logger.info("vocab loaded: %d entries", len(self._vocab))
        else:
            logger.warning("vocab.json not found at %s", vocab_path)

        if templates_path.exists():
            with open(templates_path, encoding="utf-8") as fh:
                self._templates = json.load(fh)
            logger.info("templates loaded: %d entries", len(self._templates))
        else:
            logger.warning("templates.json not found at %s", templates_path)

        # Decision thresholds
        thr_b_path = root / "artifacts" / "threshold.json"
        thr_t_path = root / "artifacts" / "threshold_transformer.json"

        if thr_b_path.exists():
            with open(thr_b_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._threshold_baseline = float(data["threshold"])
            logger.info("Baseline threshold: %.6f", self._threshold_baseline)
        else:
            logger.warning("threshold.json not found; using default %.4f",
                           self._threshold_baseline)

        if thr_t_path.exists():
            with open(thr_t_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._threshold_transformer = float(data["threshold"])
            logger.info("Transformer threshold: %.6f", self._threshold_transformer)
        else:
            logger.warning("threshold_transformer.json not found; using default %.4f",
                           self._threshold_transformer)

        # Optional runtime-calibrated thresholds (override defaults when requested)
        if self.use_runtime_thresholds:
            thr_rt_path = root / "artifacts" / "threshold_runtime.json"
            if thr_rt_path.exists():
                with open(thr_rt_path, encoding="utf-8") as fh:
                    rt_data = json.load(fh)
                rt_thresholds = rt_data.get("thresholds", {})
                # Store in separate attributes so ensemble normalization
                # continues to use the original _threshold_baseline /
                # _threshold_transformer denominators unchanged.
                if "baseline" in rt_thresholds:
                    self._rt_threshold_baseline = float(rt_thresholds["baseline"])
                    logger.info("Runtime baseline decision threshold: %.6f",
                                self._rt_threshold_baseline)
                if "transformer" in rt_thresholds:
                    self._rt_threshold_transformer = float(rt_thresholds["transformer"])
                    logger.info("Runtime transformer decision threshold: %.6f",
                                self._rt_threshold_transformer)
                if "ensemble" in rt_thresholds:
                    self._threshold_ensemble = float(rt_thresholds["ensemble"])
                    logger.info("Runtime ensemble threshold: %.6f", self._threshold_ensemble)
                logger.info("Runtime thresholds loaded from %s", thr_rt_path)
            else:
                logger.warning(
                    "--use-runtime-thresholds requested but %s not found; "
                    "using default thresholds", thr_rt_path,
                )

        # Models
        if self.mode in ("baseline", "ensemble"):
            self._load_baseline_model()
        if self.mode in ("transformer", "ensemble"):
            self._load_transformer_model()

        self._artifacts_loaded = True
        logger.info("Artifacts loaded successfully")

    def _load_baseline_model(self) -> None:
        model_path = self.root / "models" / "baseline.pkl"
        train_path = self.root / "data" / "processed" / "sequences_train.parquet"

        if not model_path.exists():
            logger.warning(
                "Baseline model not found at %s — scoring will use fallback scorer.",
                model_path,
            )
            return

        self._baseline_model = BaselineAnomalyModel.load(model_path)
        logger.info("Baseline model loaded from %s", model_path)

        # Re-fit feature extractor on training sequences to guarantee
        # identical feature column ordering as during model training.
        self._extractor = BaselineFeatureExtractor(top_k=100)
        train_seqs = self._load_sequences_from_parquet(train_path)
        if train_seqs:
            self._extractor.fit(train_seqs)
            logger.info(
                "Feature extractor fitted on %d training sequences "
                "(%d features)", len(train_seqs), self._extractor.n_features
            )
        else:
            logger.warning("No training sequences found; extractor fitted on empty list")
            self._extractor.fit([])

    def _load_transformer_model(self) -> None:
        model_path = self.root / "models" / "transformer.pt"

        if not model_path.exists():
            logger.warning(
                "Transformer model not found at %s — scoring will use fallback scorer.",
                model_path,
            )
            return

        model = NextTokenTransformerModel.load(str(model_path), map_location="cpu")
        cfg = model.cfg
        self._scorer = AnomalyScorer(model, cfg, device="cpu")
        self._scorer.set_threshold(self._threshold_transformer)
        logger.info("Transformer model loaded from %s (vocab=%d, d=%d)",
                    model_path, cfg.vocab_size, cfg.d_model)

    def _load_sequences_from_parquet(self, path: Path) -> list[Sequence]:
        """Parse a sequences parquet file into Sequence objects."""
        if not path.exists():
            logger.warning("Sequence parquet not found: %s", path)
            return []
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            seqs: list[Sequence] = []
            for _, row in df.iterrows():
                raw = row.get("tokens", "[]")
                if isinstance(raw, str):
                    tokens = json.loads(raw)
                elif hasattr(raw, "__iter__"):
                    tokens = [int(t) for t in raw]
                else:
                    tokens = []
                lbl_raw = row.get("label")
                try:
                    lbl = int(lbl_raw) if lbl_raw is not None and lbl_raw == lbl_raw else None
                except (TypeError, ValueError):
                    lbl = None
                seqs.append(Sequence(
                    sequence_id=str(row.get("sequence_id", "")),
                    tokens=[int(t) for t in tokens],
                    label=lbl,
                ))
            logger.info("Loaded %d sequences from %s", len(seqs), path)
            return seqs
        except Exception as exc:
            logger.error("Failed to load sequences from %s: %s", path, exc)
            return []

    # ------------------------------------------------------------------
    # Main ingestion API
    # ------------------------------------------------------------------

    def ingest(self, event: LogEvent | dict) -> Optional[RiskResult]:
        """
        Feed one event into the rolling buffer.

        Returns a RiskResult when a stride boundary is reached and the
        window is full; otherwise returns None.
        """
        if not self._artifacts_loaded:
            self.load_artifacts()

        key = self.buffer.ingest(event)
        if self.buffer.should_emit(key):
            seq = self.buffer.get_window(key)
            return self._build_result(key, seq, event)
        return None

    # ------------------------------------------------------------------
    # Scoring helpers (public so tests can call them directly)
    # ------------------------------------------------------------------

    def score_baseline(self, sequence: Sequence) -> float:
        """Score a Sequence using the baseline IsolationForest model."""
        if self._extractor is None or self._baseline_model is None:
            raise RuntimeError("Baseline model not loaded. Call load_artifacts().")
        X = self._extractor.transform([sequence])
        scores = self._baseline_model.score(X)
        return float(scores[0])

    def score_transformer(self, sequence: Sequence) -> float:
        """Score a Sequence using the transformer NLL scorer."""
        if self._scorer is None:
            raise RuntimeError("Transformer model not loaded. Call load_artifacts().")
        if len(sequence.tokens) < 2:
            return 0.0
        scores = self._scorer.score([sequence])
        return float(scores[0])

    def decide(self, score: float, threshold: float) -> bool:
        """Return True (anomaly) when score >= threshold."""
        return score >= threshold

    def explain(self, sequence: Sequence) -> dict:
        """
        Build an evidence dict with decoded template snippets.

        Keys: tokens, template_ids, templates_preview,
              window_start_ts, window_end_ts
        """
        tokens = sequence.tokens
        # Reverse the offset: token_id - 2 = template_id  (PAD=0, UNK=1)
        template_ids = [t - 2 for t in tokens if t >= 2]

        templates_preview: list[str] = []
        for tid in template_ids[:5]:
            tok_key = str(tid + 2)
            text = self._vocab.get(tok_key) or self._templates.get(str(tid), "<UNK>")
            snippet = (text[:100] + "...") if len(text) > 100 else text
            templates_preview.append(f"tid={tid}: {snippet}")

        start_ts = sequence.timestamps[0] if sequence.timestamps else None
        end_ts   = sequence.timestamps[-1] if sequence.timestamps else None

        return {
            "tokens": tokens[:20],
            "template_ids": template_ids[:20],
            "templates_preview": templates_preview,
            "window_start_ts": start_ts,
            "window_end_ts": end_ts,
        }

    # ------------------------------------------------------------------
    # Fallback scorer
    # ------------------------------------------------------------------

    def _score_fallback(self, sequence: Sequence) -> float:  # noqa: ARG002 (seq reserved for future rules)
        """
        Rule-based fallback used when model artifacts are absent or produce a
        feature-shape mismatch.

        * demo_mode=True  → returns self.fallback_score (crosses threshold → alert)
        * demo_mode=False → returns 0.0 (safe default; no spurious alerts)
        """
        if self.demo_mode:
            return self.fallback_score
        return 0.0

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    def _build_result(
        self,
        key: str,
        seq: Sequence,
        last_event: LogEvent | dict,
    ) -> RiskResult:
        """Score *seq*, apply threshold, build and return a RiskResult."""
        top_predictions: Optional[list] = None

        if self.mode == "baseline":
            try:
                score = self.score_baseline(seq)
            except Exception as exc:
                logger.warning(
                    "Baseline scoring failed (key=%s, n_tokens=%d): %s — using fallback",
                    key, len(seq.tokens), exc,
                )
                score = self._score_fallback(seq)
            threshold = (
                self._rt_threshold_baseline
                if self._rt_threshold_baseline is not None
                else self._threshold_baseline
            )
            model_name = "baseline"

        elif self.mode == "transformer":
            try:
                score = self.score_transformer(seq)
            except Exception as exc:
                logger.warning(
                    "Transformer scoring failed (key=%s): %s — using fallback",
                    key, exc,
                )
                score = self._score_fallback(seq)
            threshold = (
                self._rt_threshold_transformer
                if self._rt_threshold_transformer is not None
                else self._threshold_transformer
            )
            model_name = "transformer"
            top_predictions = self._get_top_predictions(seq)

        else:  # ensemble
            try:
                b_score = self.score_baseline(seq)
            except Exception as exc:
                logger.warning(
                    "Baseline scoring failed in ensemble (key=%s): %s — using fallback",
                    key, exc,
                )
                # Pre-multiply so normalization yields fallback_score as b_norm
                b_score = self._score_fallback(seq) * (self._threshold_baseline + 1e-9)
            try:
                t_score = self.score_transformer(seq)
            except Exception as exc:
                logger.warning(
                    "Transformer scoring failed in ensemble (key=%s): %s — using fallback",
                    key, exc,
                )
                t_score = self._score_fallback(seq) * (self._threshold_transformer + 1e-9)
            # Normalise each score relative to its own threshold so both
            # contribute equally; threshold=1.0 means "anomaly if either
            # model votes anomalous on average".
            b_norm = b_score / (self._threshold_baseline + 1e-9)
            t_norm = t_score / (self._threshold_transformer + 1e-9)
            score = (b_norm + t_norm) / 2.0
            threshold = self._threshold_ensemble
            model_name = "ensemble"
            top_predictions = self._get_top_predictions(seq)

        is_anom = self.decide(score, threshold)
        evidence = self.explain(seq)

        if isinstance(last_event, dict):
            ts = last_event.get("timestamp") or 0.0
        else:
            ts = getattr(last_event, "timestamp", None) or 0.0

        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = 0.0

        return RiskResult(
            stream_key=key,
            timestamp=ts,
            model=model_name,
            risk_score=round(float(score), 6),
            is_anomaly=is_anom,
            threshold=threshold,
            evidence_window=evidence,
            top_predictions=top_predictions,
            meta={
                "window_size": len(seq.tokens),
                "label": seq.label,
                "emit_index": self.buffer._emit_counts.get(key, 0),
            },
        )

    def _get_top_predictions(
        self, seq: Sequence, top_k: int = 5
    ) -> Optional[list]:
        """
        Best-effort: compute top-k next-token predictions from the transformer.
        Returns None on any failure (model not loaded, short sequence, etc.).
        """
        if self._scorer is None or len(seq.tokens) < 2:
            return None
        try:
            import torch
            import torch.nn.functional as F

            model = self._scorer.model
            tokens = seq.tokens[:-1]           # left-shift input
            inp = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                logits = model(inp)            # (1, T, vocab)
            last_logits = logits[0, -1, :]     # last-position logits
            probs = F.softmax(last_logits, dim=-1)
            k = min(top_k, int(probs.size(0)))
            topk_vals, topk_idxs = torch.topk(probs, k=k)

            preds = []
            for prob, tok_id in zip(topk_vals.tolist(), topk_idxs.tolist()):
                tid = tok_id - 2               # reverse PAD/UNK offset
                text = self._vocab.get(str(tok_id), "<UNK>")[:60]
                preds.append({
                    "token_id": tok_id,
                    "template_id": tid,
                    "prob": round(float(prob), 6),
                    "text": text,
                })
            return preds
        except Exception as exc:
            logger.debug("top_predictions skipped: %s", exc)
            return None
