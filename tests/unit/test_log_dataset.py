# tests/unit/test_log_dataset.py
# Unit tests for LogDataset (Stage 2: Sequence Dataset).
#
# Coverage:
#   - Construction: valid inputs, shape validation, labels alignment
#   - __len__: window count for various window_size/stride combinations
#   - __getitem__: tensor shape, dtype, label, IndexError on out-of-range
#   - _build_windows: window index arithmetic, label propagation
#   - label_counts: normal/anomaly distribution
#   - has_labels / num_windows properties
#   - Validation: empty embeddings, window_size/stride < 1, shape mismatch
#   - from_csv: successful load, missing file, untrained preprocessor,
#     missing message column, absent label column defaults to 0
#   - torch absent guard (mocked)

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.log_dataset import LogDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VEC_DIM = 16
WINDOW = 5
STRIDE = 1


def _make_embeddings(n: int, vec_dim: int = VEC_DIM) -> list:
    """Generate n deterministic float32 embedding vectors."""
    rng = np.random.default_rng(42)
    return [rng.random(vec_dim).astype(np.float32) for _ in range(n)]


def _make_labels(n: int, anomaly_indices=()) -> list:
    labels = [0] * n
    for i in anomaly_indices:
        labels[i] = 1
    return labels


def _make_dataset(n=20, window=WINDOW, stride=STRIDE, anomaly_indices=()):
    embs = _make_embeddings(n)
    lbls = _make_labels(n, anomaly_indices=anomaly_indices)
    return LogDataset(embeddings=embs, labels=lbls, window_size=window, stride=stride)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_creates_without_labels(self):
        embs = _make_embeddings(10)
        ds = LogDataset(embeddings=embs, labels=None, window_size=5, stride=1)
        assert len(ds) > 0

    def test_creates_with_labels(self):
        ds = _make_dataset(n=20)
        assert len(ds) > 0

    def test_vec_dim_stored(self):
        embs = _make_embeddings(10, vec_dim=32)
        ds = LogDataset(embeddings=embs, window_size=5)
        assert ds.vec_dim == 32

    def test_window_size_stored(self):
        ds = _make_dataset(window=7)
        assert ds.window_size == 7

    def test_stride_stored(self):
        ds = _make_dataset(stride=3)
        assert ds.stride == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_embeddings_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LogDataset(embeddings=[])

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            LogDataset(embeddings=_make_embeddings(5), window_size=0)

    def test_window_size_negative_raises(self):
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            LogDataset(embeddings=_make_embeddings(5), window_size=-1)

    def test_stride_zero_raises(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            LogDataset(embeddings=_make_embeddings(5), window_size=3, stride=0)

    def test_mismatched_shapes_raises(self):
        embs = _make_embeddings(5, vec_dim=16)
        embs[2] = np.zeros(8, dtype=np.float32)   # wrong shape
        with pytest.raises(ValueError, match="same shape"):
            LogDataset(embeddings=embs, window_size=3)

    def test_labels_length_mismatch_raises(self):
        embs = _make_embeddings(10)
        with pytest.raises(ValueError, match="labels length"):
            LogDataset(embeddings=embs, labels=[0, 1, 0], window_size=3)


# ---------------------------------------------------------------------------
# Window count (__len__)
# ---------------------------------------------------------------------------

class TestLen:
    def test_stride_1(self):
        # n=10, window=5, stride=1 -> windows starting at 0,1,2,3,4,5 -> 6
        ds = _make_dataset(n=10, window=5, stride=1)
        assert len(ds) == 6

    def test_stride_equals_window(self):
        # n=10, window=5, stride=5 -> windows at 0,5 -> 2
        ds = _make_dataset(n=10, window=5, stride=5)
        assert len(ds) == 2

    def test_stride_larger_than_window(self):
        # n=12, window=4, stride=6 -> windows at 0,6 (end=10<=12) -> 2
        ds = _make_dataset(n=12, window=4, stride=6)
        assert len(ds) == 2

    def test_window_equals_n(self):
        # n=5, window=5, stride=1 -> one window at 0
        ds = _make_dataset(n=5, window=5, stride=1)
        assert len(ds) == 1

    def test_window_larger_than_n(self):
        # n=3, window=5, stride=1 -> no complete windows
        ds = _make_dataset(n=3, window=5, stride=1)
        assert len(ds) == 0

    def test_no_labels(self):
        embs = _make_embeddings(10)
        ds = LogDataset(embeddings=embs, labels=None, window_size=5, stride=1)
        assert len(ds) == 6


# ---------------------------------------------------------------------------
# __getitem__ tensor shape and dtype
# ---------------------------------------------------------------------------

class TestGetItem:
    def setup_method(self):
        self.ds = _make_dataset(n=20, window=WINDOW, stride=STRIDE)

    def test_tensor_shape(self):
        tensor, _ = self.ds[0]
        assert tensor.shape == (WINDOW, VEC_DIM)

    def test_tensor_dtype_float32(self):
        import torch
        tensor, _ = self.ds[0]
        assert tensor.dtype == torch.float32

    def test_label_is_int(self):
        _, label = self.ds[0]
        assert isinstance(label, int)

    def test_all_normal_window_label_zero(self):
        ds = _make_dataset(n=20, anomaly_indices=())
        _, label = ds[0]
        assert label == 0

    def test_window_containing_anomaly_label_one(self):
        # Anomaly at index 2, window 0 covers indices 0-4 -> label=1
        ds = _make_dataset(n=20, window=5, stride=1, anomaly_indices=(2,))
        _, label = ds[0]
        assert label == 1

    def test_window_outside_anomaly_label_zero(self):
        # Anomaly at index 2, window at start=6 covers 6-10 -> label=0
        ds = _make_dataset(n=20, window=5, stride=1, anomaly_indices=(2,))
        _, label = ds[6]
        assert label == 0

    def test_index_error_negative(self):
        with pytest.raises(IndexError):
            self.ds[-1]

    def test_index_error_out_of_range(self):
        with pytest.raises(IndexError):
            self.ds[len(self.ds)]

    def test_last_valid_index_works(self):
        tensor, _ = self.ds[len(self.ds) - 1]
        assert tensor.shape == (WINDOW, VEC_DIM)

    def test_values_match_embeddings(self):
        import torch
        embs = _make_embeddings(10)
        ds = LogDataset(embeddings=embs, window_size=3, stride=1)
        tensor, _ = ds[0]
        expected = np.stack(embs[0:3], axis=0)
        np.testing.assert_allclose(tensor.numpy(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------

class TestLabelPropagation:
    def test_max_label_used(self):
        # All windows over a sequence with one anomaly at index 3
        ds = _make_dataset(n=10, window=4, stride=1, anomaly_indices=(3,))
        # Windows covering index 3: start=0(0-3), 1(1-4), 2(2-5), 3(3-6)
        for start in range(4):
            _, label = ds[start]
            assert label == 1, f"window {start} should be anomalous"
        # Window at start=4 covers 4-7: should be normal
        _, label = ds[4]
        assert label == 0

    def test_no_labels_defaults_to_zero(self):
        embs = _make_embeddings(10)
        ds = LogDataset(embeddings=embs, labels=None, window_size=3, stride=1)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_has_labels_true(self):
        ds = _make_dataset(n=10)
        assert ds.has_labels is True

    def test_has_labels_false(self):
        embs = _make_embeddings(10)
        ds = LogDataset(embeddings=embs, labels=None, window_size=3)
        assert ds.has_labels is False

    def test_num_windows_matches_len(self):
        ds = _make_dataset(n=20)
        assert ds.num_windows == len(ds)

    def test_label_counts_all_normal(self):
        ds = _make_dataset(n=10, anomaly_indices=())
        counts = ds.label_counts()
        assert counts["normal"] == len(ds)
        assert counts["anomaly"] == 0

    def test_label_counts_with_anomalies(self):
        # Anomaly at index 2; with window=5, stride=1, n=10:
        # windows at start 0-5; anomaly index 2 falls in windows 0,1,2
        ds = _make_dataset(n=10, window=5, stride=1, anomaly_indices=(2,))
        counts = ds.label_counts()
        assert counts["anomaly"] == 3
        assert counts["normal"] + counts["anomaly"] == len(ds)


# ---------------------------------------------------------------------------
# Torch-absent guard (mocked)
# ---------------------------------------------------------------------------

class TestTorchAbsentGuard:
    def test_getitem_raises_runtime_error_without_torch(self):
        ds = _make_dataset(n=10)
        import src.dataset.log_dataset as mod
        with patch.object(mod, "_TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="torch is required"):
                ds[0]


# ---------------------------------------------------------------------------
# DataLoader integration (requires torch)
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:
    def test_dataloader_batch_shape(self):
        import torch
        from torch.utils.data import DataLoader

        ds = _make_dataset(n=30, window=5, stride=1)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_tensors, batch_labels = next(iter(loader))
        # batch_size may be smaller for last batch; first batch should be 4
        assert batch_tensors.shape == (4, 5, VEC_DIM)
        assert batch_tensors.dtype == torch.float32
        assert len(batch_labels) == 4


# ---------------------------------------------------------------------------
# from_csv factory
# ---------------------------------------------------------------------------

class TestFromCsv:
    def _make_preprocessor(self, vec_dim=16):
        """Return a trained LogPreprocessor with a small synthetic corpus."""
        from src.preprocessing.log_preprocessor import LogPreprocessor
        p = LogPreprocessor(vec_dim=vec_dim, min_count=1, epochs=2)
        messages = [
            "error disk full node down",
            "warning block replication failed",
            "info checkpoint complete success",
            "critical memory exhausted abort",
            "debug connection timeout retry",
        ]
        corpus = [p.tokenize(p.clean(m)) for m in messages * 10]
        p.train_embeddings(corpus)
        return p

    def _write_csv(self, path, n=15, include_label=True):
        import pandas as pd
        messages = [
            "disk error on node",
            "block replicated successfully",
            "connection reset by peer",
        ]
        rows = {"message": [messages[i % 3] for i in range(n)]}
        if include_label:
            rows["label"] = [i % 2 for i in range(n)]
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_from_csv_returns_dataset(self, tmp_path):
        p = self._make_preprocessor()
        csv = tmp_path / "logs.csv"
        self._write_csv(csv, n=20)
        ds = LogDataset.from_csv(csv, preprocessor=p, window_size=5, stride=1)
        assert isinstance(ds, LogDataset)
        assert len(ds) == 16   # 20 rows, window=5, stride=1 -> 16 windows

    def test_from_csv_tensor_shape(self, tmp_path):
        import torch
        p = self._make_preprocessor(vec_dim=16)
        csv = tmp_path / "logs.csv"
        self._write_csv(csv, n=10)
        ds = LogDataset.from_csv(csv, preprocessor=p, window_size=4, stride=1)
        tensor, _ = ds[0]
        assert tensor.shape == (4, 16)
        assert tensor.dtype == torch.float32

    def test_from_csv_no_label_col_defaults_zero(self, tmp_path):
        p = self._make_preprocessor()
        csv = tmp_path / "logs.csv"
        self._write_csv(csv, n=10, include_label=False)
        ds = LogDataset.from_csv(csv, preprocessor=p, window_size=3, stride=1)
        assert ds.has_labels is False
        _, label = ds[0]
        assert label == 0

    def test_from_csv_missing_file_raises(self, tmp_path):
        p = self._make_preprocessor()
        with pytest.raises(FileNotFoundError):
            LogDataset.from_csv(tmp_path / "nonexistent.csv", preprocessor=p)

    def test_from_csv_untrained_preprocessor_raises(self, tmp_path):
        from src.preprocessing.log_preprocessor import LogPreprocessor
        p = LogPreprocessor()
        csv = tmp_path / "logs.csv"
        self._write_csv(csv, n=5)
        with pytest.raises(RuntimeError, match="must have a loaded embedding model"):
            LogDataset.from_csv(csv, preprocessor=p)

    def test_from_csv_missing_message_col_raises(self, tmp_path):
        import pandas as pd
        p = self._make_preprocessor()
        csv = tmp_path / "logs.csv"
        pd.DataFrame({"text": ["hello"], "label": [0]}).to_csv(csv, index=False)
        with pytest.raises(KeyError, match="message"):
            LogDataset.from_csv(csv, preprocessor=p)
