# tests/unit/test_behavior_model.py
# Unit tests for SystemBehaviorModel and BehaviorModelConfig (Phase 4).
#
# Coverage:
#   BehaviorModelConfig:
#     - default values
#     - save() / load() JSON round-trip
#   SystemBehaviorModel construction:
#     - default config
#     - explicit config
#     - input_dim/hidden_dim/num_layers/bidirectional stored
#     - lstm and proj attributes created
#   forward():
#     - output shape [batch, hidden_dim] (unidirectional)
#     - output shape [batch, hidden_dim] (bidirectional — projected)
#     - output dtype float32
#     - single-layer (dropout=0 guard)
#     - input dimension mismatch raises RuntimeError
#     - non-3D input raises RuntimeError
#   save() / load():
#     - round-trip restores weights and config
#     - parent directories created automatically
#     - missing checkpoint raises FileNotFoundError
#   torch-absent guard (mocked)

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.modeling.behavior_model import BehaviorModelConfig, SystemBehaviorModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(batch=4, seq=10, vec=100):
    """Return a random float32 input tensor [batch, seq, vec]."""
    return torch.randn(batch, seq, vec)


def _make_model(input_dim=16, hidden_dim=32, num_layers=2,
                dropout=0.0, bidirectional=False):
    cfg = BehaviorModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    return SystemBehaviorModel(cfg)


# ---------------------------------------------------------------------------
# BehaviorModelConfig
# ---------------------------------------------------------------------------

class TestBehaviorModelConfig:
    def test_default_values(self):
        cfg = BehaviorModelConfig()
        assert cfg.input_dim == 100
        assert cfg.hidden_dim == 128
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.2
        assert cfg.bidirectional is False

    def test_custom_values(self):
        cfg = BehaviorModelConfig(input_dim=50, hidden_dim=64,
                                   num_layers=1, dropout=0.1,
                                   bidirectional=True)
        assert cfg.input_dim == 50
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 1
        assert cfg.bidirectional is True

    def test_save_and_load_roundtrip(self, tmp_path):
        cfg = BehaviorModelConfig(input_dim=50, hidden_dim=64,
                                   num_layers=3, dropout=0.15,
                                   bidirectional=True)
        p = tmp_path / "config.json"
        cfg.save(p)
        loaded = BehaviorModelConfig.load(p)
        assert loaded == cfg

    def test_save_creates_valid_json(self, tmp_path):
        import json
        cfg = BehaviorModelConfig()
        p = tmp_path / "cfg.json"
        cfg.save(p)
        data = json.loads(p.read_text())
        assert data["hidden_dim"] == 128
        assert "bidirectional" in data


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_config_used_when_none(self):
        m = SystemBehaviorModel()
        assert m.cfg.hidden_dim == 128
        assert m.cfg.input_dim == 100

    def test_explicit_config_stored(self):
        cfg = BehaviorModelConfig(input_dim=32, hidden_dim=64)
        m = SystemBehaviorModel(cfg)
        assert m.input_dim == 32
        assert m.hidden_dim == 64

    def test_lstm_attribute_exists(self):
        m = _make_model()
        assert m._lstm is not None

    def test_proj_is_none_for_unidirectional(self):
        m = _make_model(bidirectional=False)
        assert m._proj is None

    def test_proj_exists_for_bidirectional(self):
        import torch.nn as nn
        m = _make_model(bidirectional=True)
        assert isinstance(m._proj, nn.Linear)

    def test_proj_dimensions_bidirectional(self):
        m = _make_model(hidden_dim=32, bidirectional=True)
        # proj: 2*hidden_dim → hidden_dim = 64 → 32
        assert m._proj.in_features == 64
        assert m._proj.out_features == 32

    def test_is_nn_module(self):
        import torch.nn as nn
        m = _make_model()
        assert isinstance(m, nn.Module)

    def test_single_layer_no_dropout_warning(self):
        # num_layers=1 should not raise; dropout is forced to 0.0
        m = _make_model(num_layers=1, dropout=0.5)
        assert m._lstm is not None


# ---------------------------------------------------------------------------
# forward() — unidirectional
# ---------------------------------------------------------------------------

class TestForwardUnidirectional:
    def setup_method(self):
        self.model = _make_model(input_dim=16, hidden_dim=32, num_layers=2)
        self.model.eval()

    def test_output_shape(self):
        x = _make_batch(batch=4, seq=10, vec=16)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (4, 32)

    def test_output_dtype_float32(self):
        x = _make_batch(batch=4, seq=10, vec=16)
        with torch.no_grad():
            out = self.model(x)
        assert out.dtype == torch.float32

    def test_batch_size_1(self):
        x = _make_batch(batch=1, seq=10, vec=16)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, 32)

    def test_different_seq_len(self):
        # LSTM handles variable sequence lengths; window_size may differ
        x = _make_batch(batch=8, seq=20, vec=16)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (8, 32)

    def test_single_layer(self):
        m = _make_model(input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        m.eval()
        x = _make_batch(batch=3, seq=5, vec=16)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (3, 32)


# ---------------------------------------------------------------------------
# forward() — bidirectional
# ---------------------------------------------------------------------------

class TestForwardBidirectional:
    def setup_method(self):
        self.model = _make_model(
            input_dim=16, hidden_dim=32, num_layers=2, bidirectional=True
        )
        self.model.eval()

    def test_output_shape_is_hidden_dim(self):
        # Projection maps 2*hidden_dim → hidden_dim
        x = _make_batch(batch=4, seq=10, vec=16)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (4, 32)   # NOT (4, 64)

    def test_dtype_float32(self):
        x = _make_batch(batch=2, seq=8, vec=16)
        with torch.no_grad():
            out = self.model(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# forward() — input validation
# ---------------------------------------------------------------------------

class TestForwardValidation:
    def setup_method(self):
        self.model = _make_model(input_dim=16, hidden_dim=32)

    def test_wrong_vec_dim_raises(self):
        x = _make_batch(batch=2, seq=5, vec=8)   # should be 16
        with pytest.raises(RuntimeError, match="input_dim"):
            self.model(x)

    def test_2d_input_raises(self):
        x = torch.randn(4, 16)   # missing seq dim
        with pytest.raises(RuntimeError, match="ndim"):
            self.model(x)

    def test_4d_input_raises(self):
        x = torch.randn(4, 10, 16, 2)
        with pytest.raises(RuntimeError, match="ndim"):
            self.model(x)


# ---------------------------------------------------------------------------
# Output consistency (determinism)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output_eval_mode(self):
        m = _make_model(input_dim=16, hidden_dim=32)
        m.eval()
        x = _make_batch(batch=2, seq=5, vec=16)
        with torch.no_grad():
            out1 = m(x)
            out2 = m(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        m = _make_model(input_dim=16, hidden_dim=32)
        p = tmp_path / "model.pt"
        m.save(p)
        assert p.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        m = _make_model(input_dim=16, hidden_dim=32)
        p = tmp_path / "models" / "sub" / "bm.pt"
        m.save(p)
        assert p.exists()

    def test_load_restores_config(self, tmp_path):
        cfg = BehaviorModelConfig(input_dim=16, hidden_dim=32,
                                   num_layers=1, dropout=0.0)
        m1 = SystemBehaviorModel(cfg)
        p = tmp_path / "bm.pt"
        m1.save(p)

        m2 = SystemBehaviorModel.load(p)
        assert m2.cfg.input_dim == 16
        assert m2.cfg.hidden_dim == 32
        assert m2.cfg.num_layers == 1

    def test_load_restores_weights(self, tmp_path):
        m1 = _make_model(input_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
        m1.eval()
        p = tmp_path / "bm.pt"
        m1.save(p)

        m2 = SystemBehaviorModel.load(p)
        m2.eval()

        x = _make_batch(batch=2, seq=5, vec=16)
        with torch.no_grad():
            out1 = m1(x)
            out2 = m2(x)
        torch.testing.assert_close(out1, out2)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SystemBehaviorModel.load(tmp_path / "nonexistent.pt")

    def test_load_bidirectional_roundtrip(self, tmp_path):
        m1 = _make_model(input_dim=16, hidden_dim=32,
                         num_layers=2, bidirectional=True)
        m1.eval()
        p = tmp_path / "bi.pt"
        m1.save(p)

        m2 = SystemBehaviorModel.load(p)
        m2.eval()
        assert m2.cfg.bidirectional is True

        x = _make_batch(batch=3, seq=7, vec=16)
        with torch.no_grad():
            out1 = m1(x)
            out2 = m2(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# torch-absent guard (mocked)
# ---------------------------------------------------------------------------

class TestTorchAbsentGuard:
    def test_forward_raises_without_torch(self):
        m = _make_model(input_dim=16, hidden_dim=32)
        import src.modeling.behavior_model as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                m.forward(torch.randn(2, 5, 16))

    def test_save_raises_without_torch(self, tmp_path):
        m = _make_model(input_dim=16, hidden_dim=32)
        import src.modeling.behavior_model as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                m.save(tmp_path / "model.pt")

    def test_load_raises_without_torch(self, tmp_path):
        import src.modeling.behavior_model as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                SystemBehaviorModel.load(tmp_path / "model.pt")


# ---------------------------------------------------------------------------
# Integration: LogDataset → SystemBehaviorModel
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify that LogDataset output feeds directly into SystemBehaviorModel."""

    def test_dataloader_to_model(self):
        from torch.utils.data import DataLoader
        from src.dataset.log_dataset import LogDataset

        VEC_DIM = 16
        WINDOW = 5
        HIDDEN = 32

        rng = np.random.default_rng(0)
        embs = [rng.random(VEC_DIM).astype(np.float32) for _ in range(30)]
        lbls = [0] * 30

        ds = LogDataset(embeddings=embs, labels=lbls,
                        window_size=WINDOW, stride=1)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        model = _make_model(input_dim=VEC_DIM, hidden_dim=HIDDEN,
                            num_layers=1, dropout=0.0)
        model.eval()

        batch_tensors, batch_labels = next(iter(loader))
        # batch_tensors: [4, WINDOW, VEC_DIM]
        assert batch_tensors.shape == (4, WINDOW, VEC_DIM)

        with torch.no_grad():
            context = model(batch_tensors)
        assert context.shape == (4, HIDDEN)
        assert context.dtype == torch.float32
