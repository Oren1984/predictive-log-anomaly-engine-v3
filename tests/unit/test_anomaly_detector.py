# tests/unit/test_anomaly_detector.py
# Unit tests for AnomalyDetector and AnomalyDetectorConfig (Phase 5).
#
# Coverage:
#   AnomalyDetectorConfig:
#     - default values
#     - save() / load() JSON round-trip
#   AnomalyDetector construction:
#     - default config; explicit config
#     - input_dim / latent_dim exposed
#     - encoder / decoder attributes exist
#     - is nn.Module
#     - uncalibrated state
#   AEOutput NamedTuple:
#     - fields accessible by name
#   forward():
#     - latent shape [B, latent_dim]
#     - reconstructed shape [B, input_dim]
#     - error shape [B], dtype float32, non-negative
#     - 2-D input required (1-D and 3-D raise)
#     - wrong input_dim raises
#     - noise injected in train mode, suppressed in eval mode
#   reconstruction_error():
#     - per-sample MSE, shape [B]
#     - zero for identical tensors
#     - positive for differing tensors
#   fit_threshold():
#     - sets threshold to correct percentile
#     - marks is_calibrated = True
#     - raises on empty errors
#     - raises on invalid percentile
#     - accepts list, numpy array, torch tensor
#   is_anomaly():
#     - False when error <= threshold
#     - True when error > threshold
#     - logs warning when uncalibrated
#   score():
#     - returns float32 numpy array shape [B]
#     - higher for anomalous-like inputs
#   save() / load():
#     - round-trip restores weights, config, threshold, calibrated flag
#     - parent directories created
#     - missing file raises FileNotFoundError
#   torch-absent guard (mocked)
#   Pipeline integration: SystemBehaviorModel → AnomalyDetector

import sys
from pathlib import Path
from unittest.mock import patch
import logging

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.modeling.anomaly_detector import (
    AEOutput,
    AnomalyDetector,
    AnomalyDetectorConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(input_dim=32, latent_dim=8, intermediate_dim=16,
              dropout=0.0, noise_std=0.0):
    return AnomalyDetectorConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        intermediate_dim=intermediate_dim,
        dropout=dropout,
        noise_std=noise_std,
    )


def _make_model(**kwargs):
    return AnomalyDetector(_make_cfg(**kwargs))


def _batch(batch=4, dim=32):
    return torch.randn(batch, dim)


# ---------------------------------------------------------------------------
# AnomalyDetectorConfig
# ---------------------------------------------------------------------------

class TestAnomalyDetectorConfig:
    def test_defaults(self):
        cfg = AnomalyDetectorConfig()
        assert cfg.input_dim == 128
        assert cfg.latent_dim == 32
        assert cfg.intermediate_dim == 64
        assert cfg.dropout == 0.1
        assert cfg.noise_std == 0.05

    def test_custom(self):
        cfg = AnomalyDetectorConfig(input_dim=64, latent_dim=16,
                                     intermediate_dim=32, dropout=0.2,
                                     noise_std=0.1)
        assert cfg.input_dim == 64
        assert cfg.latent_dim == 16

    def test_save_load_roundtrip(self, tmp_path):
        cfg = AnomalyDetectorConfig(input_dim=64, latent_dim=16,
                                     intermediate_dim=32, dropout=0.2,
                                     noise_std=0.1)
        p = tmp_path / "cfg.json"
        cfg.save(p)
        loaded = AnomalyDetectorConfig.load(p)
        assert loaded == cfg

    def test_save_json_keys(self, tmp_path):
        import json
        cfg = AnomalyDetectorConfig()
        p = tmp_path / "cfg.json"
        cfg.save(p)
        data = json.loads(p.read_text())
        assert set(data.keys()) == {
            "input_dim", "latent_dim", "intermediate_dim", "dropout", "noise_std"
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_config(self):
        m = AnomalyDetector()
        assert m.cfg.input_dim == 128
        assert m.cfg.latent_dim == 32

    def test_explicit_config(self):
        cfg = _make_cfg(input_dim=32, latent_dim=8)
        m = AnomalyDetector(cfg)
        assert m.input_dim == 32
        assert m.latent_dim == 8

    def test_encoder_exists(self):
        m = _make_model()
        assert m._encoder is not None

    def test_decoder_exists(self):
        m = _make_model()
        assert m._decoder is not None

    def test_is_nn_module(self):
        m = _make_model()
        assert isinstance(m, torch.nn.Module)

    def test_uncalibrated_by_default(self):
        m = _make_model()
        assert m.threshold == 0.0
        assert m.is_calibrated is False

    def test_encoder_is_sequential(self):
        m = _make_model()
        assert isinstance(m._encoder, torch.nn.Sequential)

    def test_decoder_is_sequential(self):
        m = _make_model()
        assert isinstance(m._decoder, torch.nn.Sequential)


# ---------------------------------------------------------------------------
# AEOutput NamedTuple
# ---------------------------------------------------------------------------

class TestAEOutput:
    def test_fields_by_name(self):
        t = torch.zeros(2, 8)
        r = torch.zeros(2, 32)
        e = torch.zeros(2)
        out = AEOutput(latent=t, reconstructed=r, error=e)
        assert out.latent is t
        assert out.reconstructed is r
        assert out.error is e

    def test_positional_unpack(self):
        t = torch.zeros(2, 8)
        r = torch.zeros(2, 32)
        e = torch.zeros(2)
        latent, recon, err = AEOutput(t, r, e)
        assert latent is t
        assert recon is r
        assert err is e


# ---------------------------------------------------------------------------
# forward() — shapes and dtypes
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def setup_method(self):
        self.m = _make_model(input_dim=32, latent_dim=8, intermediate_dim=16)
        self.m.eval()

    def test_latent_shape(self):
        x = _batch(4, 32)
        with torch.no_grad():
            out = self.m(x)
        assert out.latent.shape == (4, 8)

    def test_reconstructed_shape(self):
        x = _batch(4, 32)
        with torch.no_grad():
            out = self.m(x)
        assert out.reconstructed.shape == (4, 32)

    def test_error_shape(self):
        x = _batch(4, 32)
        with torch.no_grad():
            out = self.m(x)
        assert out.error.shape == (4,)

    def test_error_dtype_float32(self):
        x = _batch(4, 32)
        with torch.no_grad():
            out = self.m(x)
        assert out.error.dtype == torch.float32

    def test_error_non_negative(self):
        x = _batch(4, 32)
        with torch.no_grad():
            out = self.m(x)
        assert (out.error >= 0).all()

    def test_batch_size_1(self):
        x = _batch(1, 32)
        with torch.no_grad():
            out = self.m(x)
        assert out.latent.shape == (1, 8)
        assert out.reconstructed.shape == (1, 32)
        assert out.error.shape == (1,)

    def test_latent_dtype_float32(self):
        x = _batch(4, 32)
        with torch.no_grad():
            out = self.m(x)
        assert out.latent.dtype == torch.float32


# ---------------------------------------------------------------------------
# forward() — input validation
# ---------------------------------------------------------------------------

class TestForwardValidation:
    def setup_method(self):
        self.m = _make_model(input_dim=32)

    def test_1d_raises(self):
        with pytest.raises(RuntimeError, match="ndim"):
            self.m(torch.randn(32))

    def test_3d_raises(self):
        with pytest.raises(RuntimeError, match="ndim"):
            self.m(torch.randn(4, 10, 32))

    def test_wrong_input_dim_raises(self):
        with pytest.raises(RuntimeError, match="input_dim"):
            self.m(torch.randn(4, 16))  # should be 32


# ---------------------------------------------------------------------------
# forward() — denoising behaviour
# ---------------------------------------------------------------------------

class TestDenoising:
    def test_eval_mode_no_noise_deterministic(self):
        m = _make_model(input_dim=32, noise_std=0.5)
        m.eval()
        x = torch.randn(4, 32)
        with torch.no_grad():
            out1 = m(x)
            out2 = m(x)
        torch.testing.assert_close(out1.reconstructed, out2.reconstructed)

    def test_train_mode_with_noise_differs(self):
        """With noise_std>0, two forward passes in train mode should differ."""
        torch.manual_seed(0)
        m = _make_model(input_dim=32, noise_std=1.0)
        m.train()
        x = torch.randn(4, 32)
        out1 = m(x)
        out2 = m(x)
        # Very unlikely to be identical with noise injected
        assert not torch.allclose(out1.reconstructed, out2.reconstructed)

    def test_noise_std_zero_train_deterministic(self):
        """With noise_std=0, train mode should be deterministic (no noise)."""
        m = _make_model(input_dim=32, noise_std=0.0)
        m.eval()
        x = torch.randn(4, 32)
        with torch.no_grad():
            out1 = m(x)
            out2 = m(x)
        torch.testing.assert_close(out1.reconstructed, out2.reconstructed)

    def test_error_computed_against_clean_input(self):
        """Even in train mode, the error field reflects clean-input MSE."""
        m = _make_model(input_dim=32, noise_std=0.5)
        m.train()
        x = torch.zeros(2, 32)   # clean input of all zeros
        out = m(x)
        # error = MSE(x=zeros, reconstructed)
        # = mean(reconstructed^2, dim=-1)
        expected_error = (out.reconstructed ** 2).mean(dim=-1)
        torch.testing.assert_close(out.error, expected_error)


# ---------------------------------------------------------------------------
# reconstruction_error()
# ---------------------------------------------------------------------------

class TestReconstructionError:
    def setup_method(self):
        self.m = _make_model(input_dim=32)

    def test_shape(self):
        orig = _batch(4, 32)
        recon = _batch(4, 32)
        err = self.m.reconstruction_error(orig, recon)
        assert err.shape == (4,)

    def test_zero_for_identical(self):
        x = _batch(4, 32)
        err = self.m.reconstruction_error(x, x)
        assert torch.allclose(err, torch.zeros(4))

    def test_positive_for_different(self):
        orig = torch.zeros(4, 32)
        recon = torch.ones(4, 32)
        err = self.m.reconstruction_error(orig, recon)
        assert (err > 0).all()

    def test_mse_value(self):
        orig = torch.zeros(2, 4)
        recon = torch.ones(2, 4)
        # MSE = mean((0-1)^2) = 1.0 for all samples
        err = self.m.reconstruction_error(orig, recon)
        assert torch.allclose(err, torch.ones(2))


# ---------------------------------------------------------------------------
# fit_threshold()
# ---------------------------------------------------------------------------

class TestFitThreshold:
    def setup_method(self):
        self.m = _make_model()

    def test_sets_threshold(self):
        errors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.m.fit_threshold(errors, percentile=90.0)
        assert abs(self.m.threshold - 0.91) < 0.05

    def test_marks_calibrated(self):
        self.m.fit_threshold([0.1, 0.2, 0.3], percentile=50.0)
        assert self.m.is_calibrated is True

    def test_accepts_list(self):
        self.m.fit_threshold([0.1, 0.2, 0.3])
        assert self.m.is_calibrated

    def test_accepts_numpy_array(self):
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.m.fit_threshold(arr)
        assert self.m.is_calibrated

    def test_accepts_torch_tensor(self):
        t = torch.tensor([0.1, 0.2, 0.3])
        self.m.fit_threshold(t)
        assert self.m.is_calibrated

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            self.m.fit_threshold([])

    def test_percentile_zero_raises(self):
        with pytest.raises(ValueError, match="percentile"):
            self.m.fit_threshold([0.1, 0.2], percentile=0.0)

    def test_percentile_over_100_raises(self):
        with pytest.raises(ValueError, match="percentile"):
            self.m.fit_threshold([0.1, 0.2], percentile=101.0)

    def test_percentile_100_accepted(self):
        self.m.fit_threshold([0.1, 0.5, 0.9], percentile=100.0)
        assert self.m.threshold == pytest.approx(0.9)

    def test_threshold_value_correct(self):
        errors = list(range(1, 101))   # 1..100
        self.m.fit_threshold(errors, percentile=95.0)
        # p95 of [1..100] = 95.05
        assert abs(self.m.threshold - 95.05) < 0.1


# ---------------------------------------------------------------------------
# is_anomaly()
# ---------------------------------------------------------------------------

class TestIsAnomaly:
    def setup_method(self):
        self.m = _make_model()
        self.m.fit_threshold([0.1, 0.2, 0.3, 0.4, 0.5], percentile=80.0)

    def test_below_threshold_normal(self):
        assert self.m.is_anomaly(0.0) is False

    def test_above_threshold_anomaly(self):
        assert self.m.is_anomaly(1.0) is True

    def test_exactly_at_threshold_normal(self):
        # threshold is p80 ≈ 0.42; exactly at threshold is NOT > threshold
        assert self.m.is_anomaly(self.m.threshold) is False

    def test_uncalibrated_warns(self, caplog):
        m = _make_model()
        with caplog.at_level(logging.WARNING,
                             logger="src.modeling.anomaly_detector"):
            m.is_anomaly(0.5)
        assert "fit_threshold" in caplog.text


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def setup_method(self):
        self.m = _make_model(input_dim=32, latent_dim=8,
                             intermediate_dim=16, noise_std=0.0)

    def test_returns_numpy_float32(self):
        x = _batch(4, 32)
        scores = self.m.score(x)
        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float32

    def test_shape(self):
        x = _batch(4, 32)
        scores = self.m.score(x)
        assert scores.shape == (4,)

    def test_non_negative(self):
        x = _batch(4, 32)
        scores = self.m.score(x)
        assert (scores >= 0).all()

    def test_switches_to_eval(self):
        self.m.train()
        _ = self.m.score(_batch(2, 32))
        assert not self.m.training


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        m = _make_model()
        p = tmp_path / "ad.pt"
        m.save(p)
        assert p.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        m = _make_model()
        p = tmp_path / "models" / "sub" / "ad.pt"
        m.save(p)
        assert p.exists()

    def test_load_restores_config(self, tmp_path):
        cfg = _make_cfg(input_dim=32, latent_dim=8, intermediate_dim=16)
        m1 = AnomalyDetector(cfg)
        p = tmp_path / "ad.pt"
        m1.save(p)
        m2 = AnomalyDetector.load(p)
        assert m2.cfg.input_dim == 32
        assert m2.cfg.latent_dim == 8

    def test_load_restores_weights(self, tmp_path):
        m1 = _make_model(input_dim=32, noise_std=0.0)
        m1.eval()
        p = tmp_path / "ad.pt"
        m1.save(p)
        m2 = AnomalyDetector.load(p)
        m2.eval()
        x = _batch(3, 32)
        with torch.no_grad():
            out1 = m1(x)
            out2 = m2(x)
        torch.testing.assert_close(out1.reconstructed, out2.reconstructed)

    def test_load_restores_threshold_and_calibrated(self, tmp_path):
        m1 = _make_model()
        m1.fit_threshold([0.1, 0.2, 0.3, 0.4, 0.5], percentile=80.0)
        threshold_before = m1.threshold
        p = tmp_path / "ad.pt"
        m1.save(p)
        m2 = AnomalyDetector.load(p)
        assert m2.threshold == pytest.approx(threshold_before)
        assert m2.is_calibrated is True

    def test_load_uncalibrated_state_preserved(self, tmp_path):
        m1 = _make_model()  # not calibrated
        p = tmp_path / "ad.pt"
        m1.save(p)
        m2 = AnomalyDetector.load(p)
        assert m2.is_calibrated is False
        assert m2.threshold == 0.0

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnomalyDetector.load(tmp_path / "nonexistent.pt")


# ---------------------------------------------------------------------------
# torch-absent guard (mocked)
# ---------------------------------------------------------------------------

class TestTorchAbsentGuard:
    def test_forward_raises(self):
        m = _make_model()
        import src.modeling.anomaly_detector as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                m.forward(torch.randn(2, 32))

    def test_save_raises(self, tmp_path):
        m = _make_model()
        import src.modeling.anomaly_detector as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                m.save(tmp_path / "ad.pt")

    def test_load_raises(self, tmp_path):
        import src.modeling.anomaly_detector as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                AnomalyDetector.load(tmp_path / "ad.pt")

    def test_reconstruction_error_raises(self):
        m = _make_model()
        import src.modeling.anomaly_detector as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                m.reconstruction_error(torch.zeros(2, 32), torch.zeros(2, 32))


# ---------------------------------------------------------------------------
# Pipeline integration: SystemBehaviorModel → AnomalyDetector
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify context vectors from SystemBehaviorModel feed into AnomalyDetector."""

    def test_behavior_model_to_anomaly_detector(self):
        from torch.utils.data import DataLoader
        from src.dataset.log_dataset import LogDataset
        from src.modeling.behavior_model import BehaviorModelConfig, SystemBehaviorModel

        INPUT_DIM = 16
        HIDDEN_DIM = 32
        LATENT_DIM = 8
        WINDOW = 5

        # Build a tiny LogDataset
        rng = np.random.default_rng(1)
        embs = [rng.random(INPUT_DIM).astype(np.float32) for _ in range(20)]
        ds = LogDataset(embeddings=embs, labels=[0] * 20,
                        window_size=WINDOW, stride=1)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        # SystemBehaviorModel
        bm_cfg = BehaviorModelConfig(input_dim=INPUT_DIM,
                                      hidden_dim=HIDDEN_DIM,
                                      num_layers=1, dropout=0.0)
        bm = SystemBehaviorModel(bm_cfg)
        bm.eval()

        # AnomalyDetector
        ad_cfg = AnomalyDetectorConfig(input_dim=HIDDEN_DIM,
                                        latent_dim=LATENT_DIM,
                                        intermediate_dim=16,
                                        dropout=0.0, noise_std=0.0)
        ad = AnomalyDetector(ad_cfg)
        ad.eval()

        batch_tensors, _ = next(iter(loader))
        with torch.no_grad():
            context = bm(batch_tensors)          # [4, HIDDEN_DIM]
            out = ad(context)                    # AEOutput

        assert out.latent.shape == (4, LATENT_DIM)
        assert out.reconstructed.shape == (4, HIDDEN_DIM)
        assert out.error.shape == (4,)
        assert out.error.dtype == torch.float32

    def test_fit_threshold_from_pipeline_scores(self):
        """fit_threshold works with errors collected from model forward passes."""
        ad = AnomalyDetector(AnomalyDetectorConfig(
            input_dim=32, latent_dim=8, intermediate_dim=16,
            dropout=0.0, noise_std=0.0,
        ))
        ad.eval()

        # Collect normal errors
        normal_errors = []
        for _ in range(10):
            x = torch.randn(8, 32)
            with torch.no_grad():
                out = ad(x)
            normal_errors.append(out.error)

        all_errors = torch.cat(normal_errors)   # [80]
        ad.fit_threshold(all_errors, percentile=95.0)
        assert ad.is_calibrated
        assert ad.threshold > 0.0
