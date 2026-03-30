# tests/unit/test_severity_classifier.py
# Unit tests for SeverityClassifier and SeverityClassifierConfig (Phase 6).
#
# Coverage:
#   SeverityClassifierConfig:
#     - default values
#     - save() / load() JSON round-trip
#   SeverityClassifier construction:
#     - default config; explicit config
#     - input_dim / hidden_dim / num_classes exposed
#     - _mlp attribute exists
#     - is nn.Module
#   SeverityOutput NamedTuple:
#     - fields accessible by name
#   build_input() static method:
#     - [B] error shape -> [B, input_dim]
#     - [B, 1] error shape -> [B, input_dim]
#     - 1-D latent raises RuntimeError
#     - batch mismatch raises RuntimeError
#   forward():
#     - output shape [B, num_classes]
#     - 1-D input raises
#     - wrong input_dim raises
#   predict():
#     - returns SeverityOutput with valid label
#     - class_index in [0, 1, 2]
#     - confidence in [0, 1]
#     - probabilities sum to 1.0
#     - accepts scalar float error
#     - accepts 1-D latent (auto-unsqueeze)
#   predict_batch():
#     - returns list of length batch_size
#     - all labels valid
#   save() / load():
#     - round-trip restores weights and config
#     - parent directories created automatically
#     - missing file raises FileNotFoundError
#   torch-absent guard (mocked)
#   SEVERITY_LABELS constant

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.modeling.severity_classifier import (
    SEVERITY_LABELS,
    SeverityClassifier,
    SeverityClassifierConfig,
    SeverityOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(input_dim=9, hidden_dim=16, num_classes=3, dropout=0.0):
    return SeverityClassifierConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


def _make_model(**kwargs):
    return SeverityClassifier(_make_cfg(**kwargs))


def _latent(batch=4, latent_dim=8):
    return torch.randn(batch, latent_dim)


def _error(batch=4):
    return torch.rand(batch)


# ---------------------------------------------------------------------------
# SEVERITY_LABELS
# ---------------------------------------------------------------------------

class TestSeverityLabels:
    def test_label_tuple_length(self):
        assert len(SEVERITY_LABELS) == 3

    def test_label_values(self):
        assert SEVERITY_LABELS[0] == "info"
        assert SEVERITY_LABELS[1] == "warning"
        assert SEVERITY_LABELS[2] == "critical"


# ---------------------------------------------------------------------------
# SeverityClassifierConfig
# ---------------------------------------------------------------------------

class TestSeverityClassifierConfig:
    def test_defaults(self):
        cfg = SeverityClassifierConfig()
        assert cfg.input_dim == 33
        assert cfg.hidden_dim == 64
        assert cfg.num_classes == 3
        assert cfg.dropout == 0.3

    def test_custom(self):
        cfg = SeverityClassifierConfig(input_dim=17, hidden_dim=32,
                                       num_classes=3, dropout=0.1)
        assert cfg.input_dim == 17
        assert cfg.hidden_dim == 32

    def test_save_load_roundtrip(self, tmp_path):
        cfg = SeverityClassifierConfig(input_dim=9, hidden_dim=16,
                                       num_classes=3, dropout=0.0)
        p = tmp_path / "cfg.json"
        cfg.save(p)
        loaded = SeverityClassifierConfig.load(p)
        assert loaded.input_dim == 9
        assert loaded.hidden_dim == 16
        assert loaded.num_classes == 3
        assert loaded.dropout == 0.0


# ---------------------------------------------------------------------------
# SeverityOutput NamedTuple
# ---------------------------------------------------------------------------

class TestSeverityOutput:
    def test_fields_by_name(self):
        out = SeverityOutput(
            label="warning",
            class_index=1,
            confidence=0.7,
            probabilities=[0.1, 0.7, 0.2],
        )
        assert out.label == "warning"
        assert out.class_index == 1
        assert out.confidence == 0.7
        assert out.probabilities == [0.1, 0.7, 0.2]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSeverityClassifierConstruction:
    def test_default_config(self):
        model = SeverityClassifier()
        assert model.input_dim == 33
        assert model.hidden_dim == 64
        assert model.num_classes == 3

    def test_custom_config(self):
        cfg = _make_cfg(input_dim=9, hidden_dim=16)
        model = SeverityClassifier(cfg)
        assert model.input_dim == 9
        assert model.hidden_dim == 16

    def test_is_nn_module(self):
        import torch.nn as nn
        model = SeverityClassifier(_make_cfg())
        assert isinstance(model, nn.Module)

    def test_mlp_attribute_exists(self):
        model = SeverityClassifier(_make_cfg())
        assert model._mlp is not None

    def test_cfg_stored(self):
        cfg = _make_cfg(hidden_dim=32)
        model = SeverityClassifier(cfg)
        assert model.cfg is cfg


# ---------------------------------------------------------------------------
# build_input()
# ---------------------------------------------------------------------------

class TestBuildInput:
    def test_1d_error_shape(self):
        lv = _latent(batch=4, latent_dim=8)     # [4, 8]
        err = _error(batch=4)                    # [4]
        features = SeverityClassifier.build_input(lv, err)
        assert features.shape == (4, 9)          # latent_dim + 1

    def test_2d_error_shape(self):
        lv = _latent(batch=3, latent_dim=8)
        err = torch.rand(3, 1)                   # [3, 1]
        features = SeverityClassifier.build_input(lv, err)
        assert features.shape == (3, 9)

    def test_1d_latent_raises(self):
        lv = torch.randn(8)                      # wrong: not 2-D
        err = torch.rand(1)
        with pytest.raises(RuntimeError, match="latent_vector must be 2-D"):
            SeverityClassifier.build_input(lv, err)

    def test_batch_mismatch_raises(self):
        lv = _latent(batch=4, latent_dim=8)
        err = torch.rand(3)                      # different batch size
        with pytest.raises(RuntimeError, match="Batch size mismatch"):
            SeverityClassifier.build_input(lv, err)


# ---------------------------------------------------------------------------
# forward()
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape(self):
        model = _make_model(input_dim=9, hidden_dim=16)
        model.eval()
        x = torch.randn(4, 9)
        logits = model(x)
        assert logits.shape == (4, 3)

    def test_output_dtype_float32(self):
        model = _make_model(input_dim=9, hidden_dim=16)
        model.eval()
        x = torch.randn(2, 9)
        logits = model(x)
        assert logits.dtype == torch.float32

    def test_1d_input_raises(self):
        model = _make_model(input_dim=9, hidden_dim=16)
        with pytest.raises(RuntimeError, match="Expected features tensor"):
            model(torch.randn(9))

    def test_wrong_input_dim_raises(self):
        model = _make_model(input_dim=9, hidden_dim=16)
        with pytest.raises(RuntimeError, match="input_dim"):
            model(torch.randn(4, 5))

    def test_single_sample(self):
        model = _make_model(input_dim=9, hidden_dim=16)
        model.eval()
        logits = model(torch.randn(1, 9))
        assert logits.shape == (1, 3)

    def test_logits_not_probabilities(self):
        # Raw logits do not necessarily sum to 1
        model = _make_model(input_dim=9, hidden_dim=16)
        model.eval()
        logits = model(torch.randn(4, 9))
        row_sums = logits.sum(dim=-1)
        # It would be extremely unlikely for unnormalised logits to sum to 1.0
        # We just verify they are not constrained to [0,1].
        assert not torch.allclose(row_sums, torch.ones(4), atol=1e-3)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def setup_method(self):
        self.model = _make_model(input_dim=9, hidden_dim=16)

    def _lv(self):
        return torch.randn(8)   # 1-D: latent_dim=8

    def test_returns_severity_output(self):
        out = self.model.predict(self._lv(), 0.05)
        assert isinstance(out, SeverityOutput)

    def test_label_is_valid_severity(self):
        out = self.model.predict(self._lv(), 0.05)
        assert out.label in SEVERITY_LABELS

    def test_class_index_in_range(self):
        out = self.model.predict(self._lv(), 0.1)
        assert out.class_index in (0, 1, 2)

    def test_confidence_in_unit_interval(self):
        out = self.model.predict(self._lv(), 0.2)
        assert 0.0 <= out.confidence <= 1.0

    def test_probabilities_sum_to_one(self):
        out = self.model.predict(self._lv(), 0.3)
        assert abs(sum(out.probabilities) - 1.0) < 1e-5

    def test_probabilities_length(self):
        out = self.model.predict(self._lv(), 0.3)
        assert len(out.probabilities) == 3

    def test_confidence_matches_class_index(self):
        out = self.model.predict(self._lv(), 0.4)
        assert abs(out.confidence - out.probabilities[out.class_index]) < 1e-6

    def test_accepts_float_error(self):
        out = self.model.predict(self._lv(), 0.99)
        assert out.label in SEVERITY_LABELS

    def test_accepts_numpy_error(self):
        out = self.model.predict(self._lv(), np.array(0.5, dtype=np.float32))
        assert out.label in SEVERITY_LABELS

    def test_accepts_tensor_error(self):
        out = self.model.predict(self._lv(), torch.tensor(0.5))
        assert out.label in SEVERITY_LABELS

    def test_2d_latent_input(self):
        lv = torch.randn(1, 8)   # already 2-D batch of 1
        out = self.model.predict(lv, 0.1)
        assert out.label in SEVERITY_LABELS

    def test_model_in_eval_mode_after_predict(self):
        self.model.train()
        self.model.predict(self._lv(), 0.1)
        assert not self.model.training


# ---------------------------------------------------------------------------
# predict_batch()
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def setup_method(self):
        self.model = _make_model(input_dim=9, hidden_dim=16)

    def test_returns_list_of_correct_length(self):
        lv = _latent(batch=6, latent_dim=8)
        err = _error(batch=6)
        results = self.model.predict_batch(lv, err)
        assert len(results) == 6

    def test_all_labels_valid(self):
        lv = _latent(batch=4, latent_dim=8)
        err = _error(batch=4)
        for out in self.model.predict_batch(lv, err):
            assert out.label in SEVERITY_LABELS

    def test_all_probs_sum_to_one(self):
        lv = _latent(batch=3, latent_dim=8)
        err = _error(batch=3)
        for out in self.model.predict_batch(lv, err):
            assert abs(sum(out.probabilities) - 1.0) < 1e-5

    def test_single_batch(self):
        lv = _latent(batch=1, latent_dim=8)
        err = _error(batch=1)
        results = self.model.predict_batch(lv, err)
        assert len(results) == 1
        assert results[0].label in SEVERITY_LABELS


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_restores_weights(self, tmp_path):
        model = _make_model(input_dim=9, hidden_dim=16)
        p = tmp_path / "severity.pt"
        model.save(p)

        loaded = SeverityClassifier.load(p)

        x = torch.randn(2, 9)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            out_orig = model(x)
            out_load = loaded(x)
        assert torch.allclose(out_orig, out_load, atol=1e-6)

    def test_roundtrip_restores_config(self, tmp_path):
        cfg = _make_cfg(input_dim=9, hidden_dim=32)
        model = SeverityClassifier(cfg)
        p = tmp_path / "severity.pt"
        model.save(p)

        loaded = SeverityClassifier.load(p)
        assert loaded.cfg.input_dim == 9
        assert loaded.cfg.hidden_dim == 32
        assert loaded.cfg.num_classes == 3

    def test_parent_dirs_created(self, tmp_path):
        model = _make_model(input_dim=9, hidden_dim=16)
        deep = tmp_path / "a" / "b" / "c" / "severity.pt"
        model.save(deep)
        assert deep.exists()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SeverityClassifier.load(tmp_path / "nonexistent.pt")

    def test_loaded_model_produces_valid_predictions(self, tmp_path):
        model = _make_model(input_dim=9, hidden_dim=16)
        p = tmp_path / "severity.pt"
        model.save(p)
        loaded = SeverityClassifier.load(p)
        out = loaded.predict(torch.randn(8), 0.1)
        assert out.label in SEVERITY_LABELS


# ---------------------------------------------------------------------------
# Torch-absent guard (mocked)
# ---------------------------------------------------------------------------

class TestTorchAbsentGuard:
    def _patch_torch(self):
        return patch(
            "src.modeling.severity_classifier._TORCH_AVAILABLE", False
        )

    def test_forward_raises_without_torch(self):
        model = _make_model()
        with self._patch_torch():
            with pytest.raises(RuntimeError, match="torch is required"):
                model.forward(torch.randn(1, 9))

    def test_predict_raises_without_torch(self):
        model = _make_model()
        with self._patch_torch():
            with pytest.raises(RuntimeError, match="torch is required"):
                model.predict(torch.randn(8), 0.1)

    def test_save_raises_without_torch(self, tmp_path):
        model = _make_model()
        with self._patch_torch():
            with pytest.raises(RuntimeError, match="torch is required"):
                model.save(tmp_path / "x.pt")

    def test_load_raises_without_torch(self, tmp_path):
        with self._patch_torch():
            with pytest.raises(RuntimeError, match="torch is required"):
                SeverityClassifier.load(tmp_path / "x.pt")

    def test_build_input_raises_without_torch(self):
        with self._patch_torch():
            with pytest.raises(RuntimeError, match="torch is required"):
                SeverityClassifier.build_input(torch.randn(1, 8), torch.rand(1))
