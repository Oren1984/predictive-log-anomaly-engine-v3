# tests/unit/test_proactive_engine.py
# Unit tests for ProactiveMonitorEngine and EngineResult (Phase 7).
#
# Coverage:
#   EngineResult:
#     - field defaults
#     - to_dict() keys and rounding
#   _EmbeddingBuffer:
#     - returns None until window_size reached
#     - emits on window_size, then every stride
#     - returns list of correct length
#   ProactiveMonitorEngine construction:
#     - default config
#     - custom config
#     - starts unloaded
#     - counters initialised to zero
#   initialize_models() / load_models():
#     - warn-and-continue when models_dir is empty
#     - _loaded = True after call
#     - torch-absent: missing models produce None references
#   process_log():
#     - returns None when preprocessor not loaded
#     - returns None when buffer not yet full
#     - returns EngineResult when buffer emits (models mocked)
#     - increments event counter
#     - uses fallback when behavior model missing
#   process_batch():
#     - returns list of same length as input
#     - None entries are correct when buffer not full
#   score_sequence():
#     - returns fallback when torch absent
#     - returns EngineResult with mocked models
#   generate_alert():
#     - returns None for None input
#     - returns None for non-anomalous result
#     - returns alert dict for anomalous result
#     - appends to alert buffer
#   process_event():
#     - backward compat: returns window_emitted / risk_result / alert keys
#   recent_alerts():
#     - returns list from ring buffer
#   metrics_snapshot():
#     - returns expected keys
#     - model booleans reflect loaded state
#   LRU eviction:
#     - oldest key evicted when max_stream_keys exceeded

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.engine.proactive_engine import (
    EngineResult,
    ProactiveMonitorEngine,
    _EmbeddingBuffer,
)
from src.modeling.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
from src.modeling.behavior_model import BehaviorModelConfig, SystemBehaviorModel
from src.modeling.severity_classifier import SeverityClassifier, SeverityClassifierConfig


# ---------------------------------------------------------------------------
# Helpers — build tiny trained models for integration tests
# ---------------------------------------------------------------------------

VEC_DIM = 8
LATENT_DIM = 4
HIDDEN_DIM = 8
WINDOW = 4
STRIDE = 2


def _make_behavior_model(input_dim=VEC_DIM, hidden_dim=HIDDEN_DIM) -> SystemBehaviorModel:
    cfg = BehaviorModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.0,
    )
    model = SystemBehaviorModel(cfg)
    model.eval()
    return model


def _make_anomaly_detector(input_dim=HIDDEN_DIM, latent_dim=LATENT_DIM) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        intermediate_dim=6,
        dropout=0.0,
        noise_std=0.0,
    )
    det = AnomalyDetector(cfg)
    det.fit_threshold([0.01, 0.02, 0.01], percentile=95.0)
    det.eval()
    return det


def _make_severity_classifier(
    input_dim=LATENT_DIM + 1,
    hidden_dim=8,
) -> SeverityClassifier:
    cfg = SeverityClassifierConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=3,
        dropout=0.0,
    )
    clf = SeverityClassifier(cfg)
    clf.eval()
    return clf


def _make_engine(window=WINDOW, stride=STRIDE, max_keys=10) -> ProactiveMonitorEngine:
    """Build an engine with tiny models pre-wired (no disk I/O)."""
    eng = ProactiveMonitorEngine(
        window_size=window,
        stride=stride,
        max_stream_keys=max_keys,
        vec_dim=VEC_DIM,
    )
    eng._behavior_model = _make_behavior_model()
    eng._anomaly_detector = _make_anomaly_detector()
    eng._severity_classifier = _make_severity_classifier()
    eng._loaded = True
    return eng


def _rand_vec(dim=VEC_DIM) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# EngineResult
# ---------------------------------------------------------------------------

class TestEngineResult:
    def test_field_defaults(self):
        r = EngineResult(
            timestamp="2024-01-01T00:00:00",
            service="svc",
            anomaly_score=0.5,
            reconstruction_error=0.5,
            is_anomaly=True,
            severity="warning",
            confidence=0.8,
        )
        assert r.probabilities == [1.0, 0.0, 0.0]
        assert r.meta == {}

    def test_to_dict_keys(self):
        r = EngineResult(
            timestamp="ts", service="s",
            anomaly_score=0.1, reconstruction_error=0.1,
            is_anomaly=False, severity="info", confidence=0.5,
        )
        d = r.to_dict()
        assert set(d) >= {
            "timestamp", "service", "anomaly_score",
            "reconstruction_error", "is_anomaly", "severity",
            "confidence", "probabilities", "meta",
        }

    def test_to_dict_rounding(self):
        r = EngineResult(
            timestamp="ts", service="s",
            anomaly_score=0.123456789,
            reconstruction_error=0.123456789,
            is_anomaly=False, severity="info", confidence=0.999999,
        )
        d = r.to_dict()
        assert d["anomaly_score"] == round(0.123456789, 6)
        assert d["confidence"] == round(0.999999, 4)

    def test_to_dict_severity_preserved(self):
        r = EngineResult(
            timestamp="ts", service="s",
            anomaly_score=0.0, reconstruction_error=0.0,
            is_anomaly=False, severity="critical", confidence=0.9,
        )
        assert r.to_dict()["severity"] == "critical"


# ---------------------------------------------------------------------------
# _EmbeddingBuffer
# ---------------------------------------------------------------------------

class TestEmbeddingBuffer:
    def test_returns_none_until_window_full(self):
        buf = _EmbeddingBuffer(window_size=4, stride=2)
        for _ in range(3):
            assert buf.push(_rand_vec()) is None

    def test_emits_on_window_size(self):
        buf = _EmbeddingBuffer(window_size=4, stride=2)
        result = None
        for _ in range(4):
            result = buf.push(_rand_vec())
        assert result is not None
        assert len(result) == 4

    def test_emits_every_stride_after_full(self):
        buf = _EmbeddingBuffer(window_size=4, stride=2)
        results = [buf.push(_rand_vec()) for _ in range(10)]
        # Emit positions: 4, 6, 8, 10 (indices 3, 5, 7, 9)
        emitted = [i for i, r in enumerate(results) if r is not None]
        assert emitted == [3, 5, 7, 9]

    def test_stride_1_emits_every_event_after_full(self):
        buf = _EmbeddingBuffer(window_size=3, stride=1)
        results = [buf.push(_rand_vec()) for _ in range(6)]
        emitted = [i for i, r in enumerate(results) if r is not None]
        assert emitted == [2, 3, 4, 5]

    def test_window_list_has_correct_length(self):
        buf = _EmbeddingBuffer(window_size=5, stride=3)
        for _ in range(4):
            buf.push(_rand_vec())
        result = buf.push(_rand_vec())
        assert len(result) == 5

    def test_window_contains_numpy_arrays(self):
        buf = _EmbeddingBuffer(window_size=3, stride=1)
        for _ in range(3):
            result = buf.push(_rand_vec())
        for arr in result:
            assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# ProactiveMonitorEngine — construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    def test_default_config(self):
        eng = ProactiveMonitorEngine()
        assert eng.window_size == 20
        assert eng.stride == 5
        assert eng.max_stream_keys == 1000
        assert eng.vec_dim == 100

    def test_custom_config(self):
        eng = ProactiveMonitorEngine(window_size=10, stride=2, max_stream_keys=50)
        assert eng.window_size == 10
        assert eng.stride == 2
        assert eng.max_stream_keys == 50

    def test_starts_unloaded(self):
        eng = ProactiveMonitorEngine()
        assert not eng._loaded

    def test_models_are_none_initially(self):
        eng = ProactiveMonitorEngine()
        assert eng._preprocessor is None
        assert eng._behavior_model is None
        assert eng._anomaly_detector is None
        assert eng._severity_classifier is None

    def test_counters_start_at_zero(self):
        eng = ProactiveMonitorEngine()
        assert eng._events_total == 0
        assert eng._windows_total == 0
        assert eng._anomalies_total == 0


# ---------------------------------------------------------------------------
# initialize_models() — warn-and-continue with empty models dir
# ---------------------------------------------------------------------------

class TestInitializeModels:
    def test_sets_loaded_flag(self, tmp_path):
        eng = ProactiveMonitorEngine(models_dir=tmp_path)
        eng.initialize_models()
        assert eng._loaded is True

    def test_all_models_none_when_dir_empty(self, tmp_path):
        eng = ProactiveMonitorEngine(models_dir=tmp_path)
        eng.initialize_models()
        assert eng._preprocessor is None
        assert eng._behavior_model is None
        assert eng._anomaly_detector is None
        assert eng._severity_classifier is None

    def test_load_models_alias(self, tmp_path):
        eng = ProactiveMonitorEngine(models_dir=tmp_path)
        eng.load_models()
        assert eng._loaded is True

    def test_behavior_model_loaded_from_disk(self, tmp_path):
        model = _make_behavior_model()
        model.save(tmp_path / "behavior_model.pt")
        eng = ProactiveMonitorEngine(
            models_dir=tmp_path, window_size=WINDOW, stride=STRIDE, vec_dim=VEC_DIM
        )
        eng.initialize_models()
        assert eng._behavior_model is not None

    def test_anomaly_detector_loaded_from_disk(self, tmp_path):
        det = _make_anomaly_detector()
        det.save(tmp_path / "anomaly_detector.pt")
        eng = ProactiveMonitorEngine(models_dir=tmp_path)
        eng.initialize_models()
        assert eng._anomaly_detector is not None

    def test_severity_classifier_loaded_from_disk(self, tmp_path):
        clf = _make_severity_classifier()
        clf.save(tmp_path / "severity_classifier.pt")
        eng = ProactiveMonitorEngine(models_dir=tmp_path)
        eng.initialize_models()
        assert eng._severity_classifier is not None


# ---------------------------------------------------------------------------
# process_log() — no preprocessor / buffer not full
# ---------------------------------------------------------------------------

class TestProcessLog:
    def test_returns_none_without_preprocessor(self):
        eng = _make_engine()
        eng._preprocessor = None
        result = eng.process_log("ERROR disk full")
        assert result is None

    def test_returns_none_when_buffer_not_full(self):
        eng = _make_engine(window=4, stride=2)
        # Patch preprocessor to return fixed vector
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        for _ in range(3):
            result = eng.process_log("msg")
            assert result is None

    def test_returns_engine_result_on_emit(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        result = None
        for _ in range(4):
            result = eng.process_log("msg")
        assert isinstance(result, EngineResult)

    def test_increments_event_counter(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        for _ in range(3):
            eng.process_log("msg")
        assert eng._events_total == 3

    def test_increments_windows_counter_on_emit(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        for _ in range(4):
            eng.process_log("msg")
        assert eng._windows_total == 1

    def test_result_has_all_required_fields(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        result = None
        for _ in range(4):
            result = eng.process_log("msg", service="my-svc")
        d = result.to_dict()
        for key in ("timestamp", "service", "anomaly_score",
                    "reconstruction_error", "is_anomaly",
                    "severity", "confidence", "probabilities"):
            assert key in d

    def test_service_propagated(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        result = None
        for _ in range(4):
            result = eng.process_log("msg", service="web-server")
        assert result.service == "web-server"

    def test_fallback_when_behavior_model_missing(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        eng._behavior_model = None
        result = None
        for _ in range(4):
            result = eng.process_log("msg")
        # Returns fallback EngineResult
        assert result.anomaly_score == 0.0
        assert result.severity == "info"

    def test_severity_label_valid(self):
        from src.modeling.severity_classifier import SEVERITY_LABELS
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        result = None
        for _ in range(4):
            result = eng.process_log("msg")
        assert result.severity in SEVERITY_LABELS

    def test_separate_stream_keys_are_independent(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        # Push 4 events to stream A
        for _ in range(4):
            eng.process_log("msg", stream_key="A")
        # Stream B is a fresh buffer — 3 pushes should still return None
        for _ in range(3):
            r = eng.process_log("msg", stream_key="B")
            assert r is None


# ---------------------------------------------------------------------------
# process_batch()
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def test_returns_list_of_same_length(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        results = eng.process_batch(["a", "b", "c", "d"])
        assert len(results) == 4

    def test_first_three_are_none(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        results = eng.process_batch(["a", "b", "c", "d"])
        assert results[0] is None
        assert results[1] is None
        assert results[2] is None

    def test_fourth_is_engine_result(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        results = eng.process_batch(["a", "b", "c", "d"])
        assert isinstance(results[3], EngineResult)


# ---------------------------------------------------------------------------
# score_sequence()
# ---------------------------------------------------------------------------

class TestScoreSequence:
    def test_returns_engine_result(self):
        eng = _make_engine(window=WINDOW, stride=STRIDE)
        x = torch.randn(WINDOW, VEC_DIM)
        result = eng.score_sequence(x, service="test")
        assert isinstance(result, EngineResult)

    def test_service_propagated(self):
        eng = _make_engine()
        x = torch.randn(WINDOW, VEC_DIM)
        result = eng.score_sequence(x, service="svc-x")
        assert result.service == "svc-x"

    def test_fallback_without_behavior_model(self):
        eng = _make_engine()
        eng._behavior_model = None
        x = torch.randn(WINDOW, VEC_DIM)
        result = eng.score_sequence(x)
        assert result.anomaly_score == 0.0
        assert result.severity == "info"

    def test_fallback_without_anomaly_detector(self):
        eng = _make_engine()
        eng._anomaly_detector = None
        x = torch.randn(WINDOW, VEC_DIM)
        result = eng.score_sequence(x)
        assert result.anomaly_score == 0.0

    def test_no_crash_on_torch_absent(self):
        eng = _make_engine()
        x = torch.randn(WINDOW, VEC_DIM)
        with patch("src.engine.proactive_engine._TORCH_AVAILABLE", False):
            result = eng.score_sequence(x)
        assert isinstance(result, EngineResult)
        assert result.anomaly_score == 0.0

    def test_anomaly_score_is_float(self):
        eng = _make_engine()
        x = torch.randn(WINDOW, VEC_DIM)
        result = eng.score_sequence(x)
        assert isinstance(result.anomaly_score, float)

    def test_confidence_in_unit_interval(self):
        eng = _make_engine()
        x = torch.randn(WINDOW, VEC_DIM)
        result = eng.score_sequence(x)
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_one(self):
        eng = _make_engine()
        x = torch.randn(WINDOW, VEC_DIM)
        result = eng.score_sequence(x)
        assert abs(sum(result.probabilities) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# generate_alert()
# ---------------------------------------------------------------------------

class TestGenerateAlert:
    def _anomalous_result(self):
        return EngineResult(
            timestamp="2024-01-01T00:00:00",
            service="svc",
            anomaly_score=0.9,
            reconstruction_error=0.9,
            is_anomaly=True,
            severity="critical",
            confidence=0.95,
        )

    def _normal_result(self):
        return EngineResult(
            timestamp="2024-01-01T00:00:00",
            service="svc",
            anomaly_score=0.01,
            reconstruction_error=0.01,
            is_anomaly=False,
            severity="info",
            confidence=0.5,
        )

    def test_returns_none_for_none_input(self):
        eng = ProactiveMonitorEngine()
        assert eng.generate_alert(None) is None

    def test_returns_none_for_non_anomalous(self):
        eng = ProactiveMonitorEngine()
        assert eng.generate_alert(self._normal_result()) is None

    def test_returns_dict_for_anomalous(self):
        eng = ProactiveMonitorEngine()
        alert = eng.generate_alert(self._anomalous_result())
        assert isinstance(alert, dict)

    def test_alert_has_required_keys(self):
        eng = ProactiveMonitorEngine()
        alert = eng.generate_alert(self._anomalous_result())
        for key in ("timestamp", "service", "severity", "anomaly_score", "message"):
            assert key in alert

    def test_severity_uppercased_in_alert(self):
        eng = ProactiveMonitorEngine()
        alert = eng.generate_alert(self._anomalous_result())
        assert alert["severity"] == "CRITICAL"

    def test_appended_to_alert_buffer(self):
        eng = ProactiveMonitorEngine()
        assert len(eng._alert_buffer) == 0
        eng.generate_alert(self._anomalous_result())
        assert len(eng._alert_buffer) == 1


# ---------------------------------------------------------------------------
# process_event() — backward compat
# ---------------------------------------------------------------------------

class TestProcessEvent:
    def test_returns_required_keys(self):
        eng = _make_engine(window=4, stride=2)
        result = eng.process_event({"message": "test log", "service": "svc"})
        assert "window_emitted" in result
        assert "risk_result" in result
        assert "alert" in result

    def test_window_not_emitted_before_full(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        result = eng.process_event({"message": "msg"})
        assert result["window_emitted"] is False
        assert result["risk_result"] is None

    def test_window_emitted_after_full(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        for _ in range(3):
            eng.process_event({"message": "msg"})
        result = eng.process_event({"message": "msg"})
        assert result["window_emitted"] is True
        assert result["risk_result"] is not None

    def test_risk_result_is_dict(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        for _ in range(4):
            result = eng.process_event({"message": "msg"})
        assert isinstance(result["risk_result"], dict)

    def test_uses_log_line_fallback_key(self):
        """Event dict may use 'log_line' instead of 'message'."""
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        # Should not raise even though 'message' key is absent
        for _ in range(4):
            result = eng.process_event({"log_line": "test"})
        assert result["window_emitted"] is True


# ---------------------------------------------------------------------------
# recent_alerts()
# ---------------------------------------------------------------------------

class TestRecentAlerts:
    def test_empty_initially(self):
        eng = ProactiveMonitorEngine()
        assert eng.recent_alerts() == []

    def test_returns_list_after_alert(self):
        eng = ProactiveMonitorEngine()
        result = EngineResult(
            timestamp="ts", service="s",
            anomaly_score=0.9, reconstruction_error=0.9,
            is_anomaly=True, severity="critical", confidence=0.9,
        )
        eng.generate_alert(result)
        alerts = eng.recent_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) == 1

    def test_alert_buffer_respects_maxlen(self):
        eng = ProactiveMonitorEngine(alert_buffer_size=3)
        r = EngineResult(
            timestamp="ts", service="s",
            anomaly_score=0.9, reconstruction_error=0.9,
            is_anomaly=True, severity="critical", confidence=0.9,
        )
        for _ in range(5):
            eng.generate_alert(r)
        assert len(eng.recent_alerts()) == 3


# ---------------------------------------------------------------------------
# metrics_snapshot()
# ---------------------------------------------------------------------------

class TestMetricsSnapshot:
    def test_returns_expected_top_keys(self):
        eng = ProactiveMonitorEngine()
        snap = eng.metrics_snapshot()
        for key in ("loaded", "models", "config", "counters"):
            assert key in snap

    def test_model_booleans_reflect_state(self):
        eng = ProactiveMonitorEngine()
        snap = eng.metrics_snapshot()
        assert snap["models"]["preprocessor"] is False
        assert snap["models"]["behavior_model"] is False

        eng._behavior_model = _make_behavior_model()
        snap2 = eng.metrics_snapshot()
        assert snap2["models"]["behavior_model"] is True

    def test_counters_match_events(self):
        eng = _make_engine(window=4, stride=2)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        for _ in range(4):
            eng.process_log("msg")
        snap = eng.metrics_snapshot()
        assert snap["counters"]["events_total"] == 4
        assert snap["counters"]["windows_total"] == 1

    def test_config_keys(self):
        eng = ProactiveMonitorEngine(window_size=10, stride=3)
        snap = eng.metrics_snapshot()
        assert snap["config"]["window_size"] == 10
        assert snap["config"]["stride"] == 3

    def test_anomaly_threshold_is_none_without_detector(self):
        eng = ProactiveMonitorEngine()
        assert eng.metrics_snapshot()["anomaly_threshold"] is None

    def test_anomaly_threshold_from_detector(self):
        eng = ProactiveMonitorEngine()
        eng._anomaly_detector = _make_anomaly_detector()
        threshold = eng.metrics_snapshot()["anomaly_threshold"]
        assert isinstance(threshold, float)


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_oldest_key_evicted_at_cap(self):
        eng = ProactiveMonitorEngine(window_size=4, stride=2, max_stream_keys=3)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()

        eng.process_log("msg", stream_key="A")
        eng.process_log("msg", stream_key="B")
        eng.process_log("msg", stream_key="C")
        assert len(eng._buffers) == 3

        # Adding D should evict A (oldest)
        eng.process_log("msg", stream_key="D")
        assert len(eng._buffers) == 3
        assert "A" not in eng._buffers
        assert "D" in eng._buffers

    def test_max_keys_not_exceeded(self):
        eng = ProactiveMonitorEngine(window_size=4, stride=2, max_stream_keys=5)
        eng._preprocessor = MagicMock()
        eng._preprocessor.process_log.return_value = _rand_vec()
        for i in range(10):
            eng.process_log("msg", stream_key=f"svc-{i}")
        assert len(eng._buffers) <= 5


# ---------------------------------------------------------------------------
# __init__ export
# ---------------------------------------------------------------------------

class TestPackageExport:
    def test_importable_from_engine_package(self):
        from src.engine import EngineResult, ProactiveMonitorEngine
        assert ProactiveMonitorEngine is not None
        assert EngineResult is not None
