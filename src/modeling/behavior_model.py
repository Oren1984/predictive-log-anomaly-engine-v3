# src/modeling/behavior_model.py
# Stage 3: Sequence Modeling
#
# SystemBehaviorModel learns temporal behavioral patterns in log windows
# using a stacked LSTM.  Given a batch of embedded log sequences, it
# returns a single context vector per sequence summarising the window's
# behavioural pattern.  This vector is the input to AnomalyDetector in
# Phase 5.
#
# Design notes:
#   - Inherits from nn.Module so it integrates with standard PyTorch
#     training loops (optimizer.step, DataLoader, etc.).
#   - torch is imported lazily so the module remains importable in
#     environments where torch is not installed (CI fast-suite).
#   - Config is a JSON-serialisable dataclass, mirroring the pattern
#     established by src/modeling/transformer/config.py.
#   - Save/load mirrors NextTokenTransformerModel exactly:
#     torch.save({"state_dict": ..., "cfg": ...}) + safe_globals load.
#   - A linear projection layer normalises the output to hidden_dim
#     regardless of bidirectionality, so the output contract is always
#     [batch, hidden_dim] and downstream phases never need to branch.
#   - bidirectional defaults to False for simplicity and alignment with
#     the agreed architecture; it can be enabled for experiments.
#   - This class is completely isolated from the existing runtime
#     pipeline.  It is not imported by any existing module.

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional torch import — required at forward / save / load time
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
    _TorchModule = nn.Module
except ImportError:                             # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None                                # type: ignore[assignment]
    nn = None                                   # type: ignore[assignment]
    _TorchModule = object                       # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# BehaviorModelConfig
# ---------------------------------------------------------------------------

@dataclass
class BehaviorModelConfig:
    """
    Hyperparameters for SystemBehaviorModel.

    Mirrors the pattern of ``TransformerConfig`` in
    ``src/modeling/transformer/config.py``.

    Parameters
    ----------
    input_dim:
        Size of each embedding vector — must match ``LogPreprocessor.vec_dim``
        (default 100).
    hidden_dim:
        LSTM hidden state size.  Also the final output dimension after
        projection (default 128).
    num_layers:
        Number of stacked LSTM layers (default 2).
    dropout:
        Dropout probability applied between LSTM layers.  Ignored
        (forced to 0.0) when ``num_layers == 1`` per PyTorch LSTM rules.
        Default 0.2.
    bidirectional:
        When ``True``, a BiLSTM is used and the forward/backward final
        hidden states are concatenated then projected back to
        ``hidden_dim``.  Default ``False`` (simple unidirectional LSTM).
    """

    input_dim: int = 100
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Serialise config to a JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "BehaviorModelConfig":
        """Load config from a JSON file."""
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# SystemBehaviorModel
# ---------------------------------------------------------------------------

class SystemBehaviorModel(_TorchModule):  # type: ignore[misc]
    """
    Stage 3: Sequence Modeling (LSTM).

    Learns normal system behavioural patterns from sequences of embedded
    log vectors produced by ``LogPreprocessor``.

    Architecture
    ------------
    ::

        Input  [B, T, input_dim]
            |
            v  nn.LSTM(batch_first=True)
        h_n  [num_layers * num_directions, B, hidden_dim]
            |
            | extract final-layer hidden state(s)
            v
        h    [B, hidden_dim]          (unidirectional)
          or [B, 2 * hidden_dim]      (bidirectional)
            |
            v  nn.Linear (only when bidirectional=True)
        context  [B, hidden_dim]

    The projection layer normalises the output to ``hidden_dim`` in all
    cases, so the output contract is always::

        torch.FloatTensor [batch_size, hidden_dim]

    Parameters
    ----------
    cfg:
        A ``BehaviorModelConfig`` instance.  If omitted, default
        hyperparameters are used.

    Raises
    ------
    RuntimeError
        If ``forward``, ``save``, or ``load`` is called and torch is not
        installed.
    """

    def __init__(self, cfg: Optional[BehaviorModelConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = BehaviorModelConfig()
        self.cfg = cfg

        # Expose config fields directly for convenience
        self.input_dim: int = cfg.input_dim
        self.hidden_dim: int = cfg.hidden_dim
        self.num_layers: int = cfg.num_layers
        self.bidirectional: bool = cfg.bidirectional

        if _TORCH_AVAILABLE:
            self._build_network()
        else:
            # Placeholder attributes — usable for inspection, not for compute
            self._lstm: Optional[object] = None
            self._proj: Optional[object] = None

        logger.debug(
            "SystemBehaviorModel: input_dim=%d hidden_dim=%d "
            "num_layers=%d bidirectional=%s",
            self.input_dim, self.hidden_dim,
            self.num_layers, self.bidirectional,
        )

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(self) -> None:
        """Instantiate LSTM and optional projection layer."""
        # PyTorch LSTM requires dropout=0 when num_layers == 1
        lstm_dropout = self.cfg.dropout if self.cfg.num_layers > 1 else 0.0

        self._lstm = nn.LSTM(
            input_size=self.cfg.input_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.cfg.bidirectional,
        )

        # Projection: maps BiLSTM output [2 * hidden_dim] → [hidden_dim]
        # For unidirectional LSTM the output already has shape [hidden_dim]
        # so no projection is needed.
        lstm_out_dim = self.cfg.hidden_dim * (2 if self.cfg.bidirectional else 1)
        self._proj = (
            nn.Linear(lstm_out_dim, self.cfg.hidden_dim)
            if self.cfg.bidirectional
            else None
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Run the LSTM over a batch of log-window sequences.

        Parameters
        ----------
        x : torch.FloatTensor
            Shape ``[batch_size, seq_len, input_dim]``.  Produced by
            ``LogDataset`` / ``DataLoader`` (Phase 3).

        Returns
        -------
        torch.FloatTensor
            Context vector of shape ``[batch_size, hidden_dim]``.

        Raises
        ------
        RuntimeError
            If torch is not installed or if the input shape is invalid.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SystemBehaviorModel.forward. "
                "Install it with: pip install torch"
            )
        if x.ndim != 3:
            raise RuntimeError(
                f"Expected input tensor of shape [batch, seq_len, input_dim], "
                f"got {tuple(x.shape)} (ndim={x.ndim})"
            )
        if x.size(2) != self.cfg.input_dim:
            raise RuntimeError(
                f"Input vector size {x.size(2)} does not match "
                f"model input_dim {self.cfg.input_dim}"
            )

        # _lstm: output (B, T, num_directions*hidden_dim), h_n (layers*dirs, B, hidden_dim)
        _, (h_n, _) = self._lstm(x)

        if self.cfg.bidirectional:
            # h_n[-2]: last-layer forward direction  [B, hidden_dim]
            # h_n[-1]: last-layer backward direction [B, hidden_dim]
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2 * hidden_dim]
            context = self._proj(h)                     # [B, hidden_dim]
        else:
            context = h_n[-1]                           # [B, hidden_dim]

        return context

    # ------------------------------------------------------------------
    # Persistence — mirrors NextTokenTransformerModel exactly
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """
        Save model checkpoint (state dict + config) to disk.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``models/behavior_model.pt``).
            Parent directories are created automatically.

        Raises
        ------
        RuntimeError
            If torch is not installed or no network has been built.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SystemBehaviorModel.save. "
                "Install it with: pip install torch"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)
        logger.info("Saved SystemBehaviorModel to %s", path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        map_location: str = "cpu",
    ) -> "SystemBehaviorModel":
        """
        Load a previously saved model checkpoint from disk.

        Parameters
        ----------
        path:
            Path to a checkpoint saved by :meth:`save`.
        map_location:
            Torch device string (default ``"cpu"``).

        Returns
        -------
        SystemBehaviorModel
            A new instance with weights and config restored.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SystemBehaviorModel.load. "
                "Install it with: pip install torch"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # safe_globals allows BehaviorModelConfig to be deserialised
        # safely under weights_only=True (torch >= 2.0 requirement)
        with torch.serialization.safe_globals([BehaviorModelConfig]):
            ckpt = torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["state_dict"])
        logger.info(
            "Loaded SystemBehaviorModel from %s (hidden_dim=%d)",
            path, ckpt["cfg"].hidden_dim,
        )
        return model
