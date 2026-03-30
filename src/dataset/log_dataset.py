# src/dataset/log_dataset.py
# Stage 2: Sequence Dataset
#
# LogDataset wraps embedded log sequences into a PyTorch Dataset that
# returns sliding-window tensors for DataLoader batching.
#
# Design notes:
#   - Inherits from torch.utils.data.Dataset so it integrates directly
#     with DataLoader without any adapter layer.
#   - torch is imported lazily so the module remains importable in CI
#     environments where torch is not installed (e.g. tests targeting
#     only the API layer).
#   - Sliding-window logic is adapted from src/sequencing/builders.py
#     (SlidingWindowSequenceBuilder) but operates on float embedding
#     vectors (np.ndarray) rather than discrete token IDs.
#   - Window label policy: max(labels[start:end]).  A window is labelled
#     anomalous (1) if any of its constituent log lines is anomalous.
#   - Windows that are shorter than window_size (trailing incomplete
#     windows) are silently discarded to guarantee consistent tensor
#     shapes for the LSTM.
#   - This class is completely isolated from the existing runtime
#     pipeline.  It is not imported by any existing module and changes
#     no behaviour.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional torch import — only required for tensor output
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:                             # pragma: no cover
    _TORCH_AVAILABLE = False
    _TorchDataset = object                      # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# LogDataset
# ---------------------------------------------------------------------------

class LogDataset(_TorchDataset):  # type: ignore[misc]
    """
    Stage 2: Sequence Data Preparation.

    Wraps a sequence of float embedding vectors and optional labels into a
    ``torch.utils.data.Dataset`` that returns sliding-window tensors
    ready for LSTM training.

    Each item returned by ``__getitem__`` is a tuple::

        (FloatTensor[window_size, vec_dim], int_label)

    A ``DataLoader`` over this dataset produces batches of shape::

        [batch_size, window_size, vec_dim]

    Parameters
    ----------
    embeddings:
        Ordered list of 1-D ``np.ndarray`` float vectors, one per log
        line.  All arrays must have the same shape ``(vec_dim,)``.
    labels:
        Optional list of integer labels aligned to ``embeddings``
        (``0`` = normal, ``1`` = anomaly).  If ``None``, all window
        labels default to ``0``.
    window_size:
        Number of consecutive log-line embeddings in each window.
        Windows shorter than ``window_size`` (trailing remainder) are
        discarded.  Default: 20.
    stride:
        Step size between consecutive window start positions.
        Adapted from ``SlidingWindowSequenceBuilder.stride``.
        Default: 1.

    Raises
    ------
    ValueError
        If ``embeddings`` is empty, ``window_size < 1``, ``stride < 1``,
        all arrays do not share the same shape, or ``labels`` length
        does not match ``embeddings`` length.
    RuntimeError
        If ``__getitem__`` is called and torch is not installed.
    """

    def __init__(
        self,
        embeddings: List[np.ndarray],
        labels: Optional[List[int]] = None,
        window_size: int = 20,
        stride: int = 1,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if not embeddings:
            raise ValueError("embeddings must not be empty")

        # Validate that all embeddings share the same shape
        vec_dim = embeddings[0].shape
        for i, emb in enumerate(embeddings):
            if emb.shape != vec_dim:
                raise ValueError(
                    f"All embeddings must have the same shape. "
                    f"embeddings[0].shape={vec_dim}, "
                    f"embeddings[{i}].shape={emb.shape}"
                )

        if labels is not None and len(labels) != len(embeddings):
            raise ValueError(
                f"labels length ({len(labels)}) must match "
                f"embeddings length ({len(embeddings)})"
            )

        self.embeddings: List[np.ndarray] = embeddings
        self._labels: Optional[List[int]] = labels
        self.window_size: int = window_size
        self.stride: int = stride
        self.vec_dim: int = embeddings[0].shape[0]

        # Build the window index table once at construction time.
        # Each entry is (start, end, label) where start:end indexes into
        # self.embeddings.  Storing indices rather than pre-stacked arrays
        # avoids duplicating the embedding data in memory.
        self._windows: List[Tuple[int, int, int]] = self._build_windows()

        logger.info(
            "LogDataset: %d embeddings, window_size=%d, stride=%d -> %d windows",
            len(self.embeddings), self.window_size, self.stride, len(self._windows),
        )

    # ------------------------------------------------------------------
    # Window construction (adapted from SlidingWindowSequenceBuilder)
    # ------------------------------------------------------------------

    def _build_windows(self) -> List[Tuple[int, int, int]]:
        """
        Build a list of ``(start, end, label)`` window index tuples.

        Mirrors the loop in ``SlidingWindowSequenceBuilder.build()``
        but operates on embedding indices instead of token IDs.
        Trailing incomplete windows (``end > len(embeddings)``) are
        discarded so every window has exactly ``window_size`` rows.
        """
        n = len(self.embeddings)
        windows: List[Tuple[int, int, int]] = []

        for start in range(0, n, self.stride):
            end = start + self.window_size
            if end > n:
                break  # discard incomplete trailing window
            if self._labels is not None:
                label = int(max(self._labels[start:end]))
            else:
                label = 0
            windows.append((start, end, label))

        return windows

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of sliding windows in the dataset."""
        return len(self._windows)

    def __getitem__(self, idx: int):
        """
        Return the ``idx``-th window as a ``(tensor, label)`` tuple.

        Parameters
        ----------
        idx:
            Integer index in ``[0, len(self))``.

        Returns
        -------
        tensor : torch.FloatTensor of shape ``[window_size, vec_dim]``
            Stacked embedding vectors for the window.
        label : int
            Window label (``0`` = normal, ``1`` = anomaly).

        Raises
        ------
        RuntimeError
            If torch is not installed.
        IndexError
            If ``idx`` is out of range.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for LogDataset.__getitem__. "
                "Install it with: pip install torch"
            )
        if idx < 0 or idx >= len(self._windows):
            raise IndexError(
                f"index {idx} is out of range for dataset of size {len(self._windows)}"
            )

        start, end, label = self._windows[idx]
        # Stack embedding rows into a 2-D array [window_size, vec_dim]
        window_array = np.stack(self.embeddings[start:end], axis=0)   # (W, D)
        tensor = torch.tensor(window_array, dtype=torch.float32)
        return tensor, label

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_windows(self) -> int:
        """Alias for ``len(self)`` — number of sliding windows."""
        return len(self._windows)

    @property
    def has_labels(self) -> bool:
        """``True`` when per-line labels were supplied at construction."""
        return self._labels is not None

    def label_counts(self) -> dict:
        """
        Return a dict with the count of normal and anomalous windows.

        Useful for checking class balance before training.

        Returns
        -------
        dict
            ``{"normal": int, "anomaly": int}``
        """
        anomaly = sum(1 for _, _, lbl in self._windows if lbl == 1)
        return {"normal": len(self._windows) - anomaly, "anomaly": anomaly}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        preprocessor,
        window_size: int = 20,
        stride: int = 1,
        message_col: str = "message",
        label_col: str = "label",
    ) -> "LogDataset":
        """
        Load a log CSV, embed each row via ``preprocessor``, return a
        ``LogDataset``.

        The CSV must contain a ``message`` column (configurable via
        ``message_col``).  If a ``label`` column is present it is read
        as integer labels; otherwise all labels default to ``0``.

        Parameters
        ----------
        csv_path:
            Path to a CSV file with at least a message column.
        preprocessor:
            A trained ``LogPreprocessor`` instance.  Must have
            ``is_trained == True``.
        window_size:
            Passed through to ``LogDataset.__init__``.
        stride:
            Passed through to ``LogDataset.__init__``.
        message_col:
            Column name containing raw log message strings.
        label_col:
            Column name for integer labels.  If absent, labels default
            to ``0``.

        Raises
        ------
        RuntimeError
            If ``preprocessor.is_trained`` is ``False``.
        FileNotFoundError
            If ``csv_path`` does not exist.
        KeyError
            If ``message_col`` is not present in the CSV.
        """
        import pandas as pd  # lazy import — pandas only needed here

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not preprocessor.is_trained:
            raise RuntimeError(
                "preprocessor must have a loaded embedding model before "
                "calling from_csv.  Call preprocessor.train_embeddings() "
                "or preprocessor.load() first."
            )

        df = pd.read_csv(csv_path)
        if message_col not in df.columns:
            raise KeyError(
                f"Column {message_col!r} not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

        logger.info(
            "LogDataset.from_csv: embedding %d rows from %s",
            len(df), csv_path,
        )
        embeddings = [preprocessor.process_log(str(msg)) for msg in df[message_col]]

        labels: Optional[List[int]] = None
        if label_col in df.columns:
            labels = [int(v) for v in df[label_col]]

        return cls(
            embeddings=embeddings,
            labels=labels,
            window_size=window_size,
            stride=stride,
        )
