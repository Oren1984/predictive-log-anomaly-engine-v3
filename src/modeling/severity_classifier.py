# src/modeling/severity_classifier.py
# Phase 6: Severity Classification
#
# SeverityClassifier is a trained MLP that takes the latent vector and
# reconstruction error from AnomalyDetector (Phase 5) and classifies
# anomaly severity into one of three levels: Info, Warning, or Critical.
#
# Design notes:
#   - Inherits from nn.Module using the same lazy-torch and save/load
#     patterns established in Phase 4 (SystemBehaviorModel) and Phase 5
#     (AnomalyDetector).
#   - SeverityClassifierConfig mirrors BehaviorModelConfig / AnomalyDetectorConfig.
#   - MLP is a 3-layer feed-forward network:
#       Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
#   - forward() returns raw logits [batch, 3].  Call softmax externally
#     for probabilities, or use predict() for convenience.
#   - predict() runs in eval mode with no_grad, returns predicted class
#     index, label string, and probability confidence.
#   - Input contract: concat(latent_vector [latent_dim], reconstruction_error [1])
#     => FloatTensor [batch_size, latent_dim + 1]
#   - Output contract: FloatTensor [batch_size, 3] (raw logits)
#     Label mapping: 0 = Info, 1 = Warning, 2 = Critical
#   - save() / load() mirror AnomalyDetector exactly:
#     torch.save({"state_dict": ..., "cfg": ...}) + safe_globals load.
#   - This class is completely isolated from the existing runtime pipeline.
#     No AlertPolicy or engine integration is performed here.

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np

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
# Label mapping — single source of truth
# ---------------------------------------------------------------------------

SEVERITY_LABELS = ("info", "warning", "critical")
#   index 0 = "info"
#   index 1 = "warning"
#   index 2 = "critical"


# ---------------------------------------------------------------------------
# SeverityOutput — named return type for predict()
# ---------------------------------------------------------------------------

class SeverityOutput(NamedTuple):
    """
    Named output of :meth:`SeverityClassifier.predict`.

    Attributes
    ----------
    label : str
        Human-readable severity label: ``"info"``, ``"warning"``, or
        ``"critical"``.
    class_index : int
        Predicted class index: 0 = Info, 1 = Warning, 2 = Critical.
    confidence : float
        Softmax probability of the predicted class (0.0 – 1.0).
    probabilities : list[float]
        Full 3-class softmax distribution ``[p_info, p_warning, p_critical]``.
    """
    label: str
    class_index: int
    confidence: float
    probabilities: List[float]


# ---------------------------------------------------------------------------
# SeverityClassifierConfig
# ---------------------------------------------------------------------------

@dataclass
class SeverityClassifierConfig:
    """
    Hyperparameters for :class:`SeverityClassifier`.

    Mirrors the pattern of ``AnomalyDetectorConfig`` / ``BehaviorModelConfig``.

    Parameters
    ----------
    input_dim:
        Size of the combined input vector: ``latent_dim + 1``.
        The ``+1`` accounts for the scalar reconstruction error appended
        to the latent vector from :class:`AnomalyDetector`.
        Default 33 (latent_dim=32 from AnomalyDetectorConfig + 1).
    hidden_dim:
        Hidden layer size in the MLP (default 64).
    num_classes:
        Number of output classes (default 3: Info / Warning / Critical).
        Changing this requires re-training; exposed for completeness.
    dropout:
        Dropout probability applied after each hidden ReLU (default 0.3).
    """

    input_dim: int = 33       # latent_dim (32) + 1 reconstruction error
    hidden_dim: int = 64
    num_classes: int = 3      # Info=0, Warning=1, Critical=2
    dropout: float = 0.3

    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Serialise config to a JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "SeverityClassifierConfig":
        """Load config from a JSON file."""
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# SeverityClassifier
# ---------------------------------------------------------------------------

class SeverityClassifier(_TorchModule):  # type: ignore[misc]
    """
    Phase 6: Severity Classification (MLP).

    Classifies the severity of an anomalous log window into one of three
    levels — Info, Warning, or Critical — using features derived from
    the :class:`AnomalyDetector` (Phase 5) output:

    * ``latent_vector``  — the bottleneck representation ``[B, latent_dim]``
    * ``reconstruction_error`` — the scalar anomaly score ``[B]`` or ``[B, 1]``

    These are concatenated to form the combined input
    ``[B, latent_dim + 1]`` before being fed through the MLP.

    Architecture
    ------------
    ::

        Input  [B, input_dim]           (input_dim = latent_dim + 1)
            |
            v  Linear(input_dim -> hidden_dim)
            v  ReLU
            v  Dropout(dropout)
            |
            v  Linear(hidden_dim -> hidden_dim)
            v  ReLU
            v  Dropout(dropout)
            |
            v  Linear(hidden_dim -> num_classes)
            |
        logits  [B, 3]

    ``forward()`` returns **raw logits**.  Softmax is applied only inside
    :meth:`predict` to avoid double-softmax during training with
    ``nn.CrossEntropyLoss`` (which applies log-softmax internally).

    Label Mapping
    -------------
    ::

        0 = "info"
        1 = "warning"
        2 = "critical"

    Parameters
    ----------
    cfg:
        A :class:`SeverityClassifierConfig` instance.  If omitted,
        default hyperparameters are used.

    Raises
    ------
    RuntimeError
        If ``forward``, ``predict``, ``save``, or ``load`` is called
        and torch is not installed.
    """

    def __init__(self, cfg: Optional[SeverityClassifierConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SeverityClassifierConfig()
        self.cfg = cfg

        # Expose key config fields for external inspection
        self.input_dim: int = cfg.input_dim
        self.hidden_dim: int = cfg.hidden_dim
        self.num_classes: int = cfg.num_classes

        if _TORCH_AVAILABLE:
            self._build_network()
        else:
            self._mlp: Optional[object] = None  # pragma: no cover

        logger.debug(
            "SeverityClassifier: input_dim=%d hidden_dim=%d "
            "num_classes=%d dropout=%.2f",
            self.input_dim, self.hidden_dim,
            self.num_classes, cfg.dropout,
        )

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(self) -> None:
        """Instantiate the MLP as a single nn.Sequential block."""
        cfg = self.cfg
        self._mlp = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    # ------------------------------------------------------------------
    # Input preparation helper
    # ------------------------------------------------------------------

    @staticmethod
    def build_input(
        latent_vector: "torch.Tensor",
        reconstruction_error: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Concatenate ``latent_vector`` and ``reconstruction_error`` into
        the combined MLP input tensor.

        Parameters
        ----------
        latent_vector : torch.FloatTensor
            Shape ``[batch_size, latent_dim]``.  Bottleneck output from
            :class:`AnomalyDetector`.
        reconstruction_error : torch.FloatTensor
            Shape ``[batch_size]`` or ``[batch_size, 1]``.  Per-sample
            MSE score from :class:`AnomalyDetector`.

        Returns
        -------
        torch.FloatTensor
            Shape ``[batch_size, latent_dim + 1]``.

        Raises
        ------
        RuntimeError
            If torch is not installed or shapes are incompatible.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SeverityClassifier.build_input. "
                "Install it with: pip install torch"
            )
        if latent_vector.ndim != 2:
            raise RuntimeError(
                f"latent_vector must be 2-D [batch, latent_dim], "
                f"got shape {tuple(latent_vector.shape)}"
            )
        err = reconstruction_error
        if err.ndim == 1:
            err = err.unsqueeze(1)          # [B] -> [B, 1]
        if err.ndim != 2 or err.size(1) != 1:
            raise RuntimeError(
                f"reconstruction_error must be [B] or [B, 1], "
                f"got shape {tuple(reconstruction_error.shape)}"
            )
        if latent_vector.size(0) != err.size(0):
            raise RuntimeError(
                f"Batch size mismatch: latent_vector has {latent_vector.size(0)} "
                f"samples, reconstruction_error has {err.size(0)}"
            )
        return torch.cat([latent_vector, err], dim=1)   # [B, latent_dim + 1]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        """
        Run the MLP over a batch of combined feature vectors.

        Parameters
        ----------
        features : torch.FloatTensor
            Shape ``[batch_size, input_dim]``.  Typically produced by
            :meth:`build_input` from AnomalyDetector outputs.

        Returns
        -------
        torch.FloatTensor
            Raw logits of shape ``[batch_size, num_classes]``.
            Apply ``torch.softmax(logits, dim=-1)`` to get probabilities.
            Use ``torch.argmax(logits, dim=-1)`` for class predictions.

        Raises
        ------
        RuntimeError
            If torch is not installed or the input shape is invalid.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SeverityClassifier.forward. "
                "Install it with: pip install torch"
            )
        if features.ndim != 2:
            raise RuntimeError(
                f"Expected features tensor of shape [batch, input_dim], "
                f"got {tuple(features.shape)} (ndim={features.ndim})"
            )
        if features.size(1) != self.cfg.input_dim:
            raise RuntimeError(
                f"Feature vector size {features.size(1)} does not match "
                f"model input_dim {self.cfg.input_dim}. "
                f"Use SeverityClassifier.build_input() to prepare features."
            )
        return self._mlp(features)      # [B, num_classes]

    # ------------------------------------------------------------------
    # Inference convenience
    # ------------------------------------------------------------------

    def predict(
        self,
        latent_vector: "torch.Tensor",
        reconstruction_error: Union["torch.Tensor", float, "np.ndarray"],
    ) -> SeverityOutput:
        """
        Classify the severity of one or more anomalous windows.

        Switches the model to eval mode, disables gradients, builds the
        combined input, runs the forward pass, and returns a
        :class:`SeverityOutput` named tuple for the **first sample** in
        the batch (or the single sample if ``latent_vector`` is 1-D).

        For batch prediction of all samples, call :meth:`predict_batch`.

        Parameters
        ----------
        latent_vector : torch.FloatTensor
            Shape ``[latent_dim]`` (single sample) or
            ``[1, latent_dim]`` (batch of one).
        reconstruction_error : float | np.ndarray | torch.FloatTensor
            Scalar reconstruction error for the window.

        Returns
        -------
        SeverityOutput
            Named tuple with ``label``, ``class_index``, ``confidence``,
            and ``probabilities``.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SeverityClassifier.predict. "
                "Install it with: pip install torch"
            )

        # Normalise latent_vector to [1, latent_dim]
        lv = latent_vector
        if lv.ndim == 1:
            lv = lv.unsqueeze(0)    # [latent_dim] -> [1, latent_dim]

        # Normalise reconstruction_error to torch scalar tensor [1]
        if isinstance(reconstruction_error, (int, float)):
            re = torch.tensor([float(reconstruction_error)], dtype=torch.float32)
        elif isinstance(reconstruction_error, np.ndarray):
            re = torch.from_numpy(
                np.asarray(reconstruction_error, dtype=np.float32).reshape(-1)
            )[:1]
        else:
            re = reconstruction_error.float().reshape(-1)[:1]

        self.eval()
        with torch.no_grad():
            features = self.build_input(lv, re)         # [1, input_dim]
            logits = self(features)                     # [1, num_classes]
            probs = torch.softmax(logits, dim=-1)[0]    # [num_classes]
            class_idx = int(probs.argmax().item())
            confidence = float(probs[class_idx].item())
            prob_list = probs.cpu().tolist()

        return SeverityOutput(
            label=SEVERITY_LABELS[class_idx],
            class_index=class_idx,
            confidence=confidence,
            probabilities=prob_list,
        )

    def predict_batch(
        self,
        latent_vectors: "torch.Tensor",
        reconstruction_errors: "torch.Tensor",
    ) -> List[SeverityOutput]:
        """
        Classify severity for a batch of anomalous windows.

        Parameters
        ----------
        latent_vectors : torch.FloatTensor [batch_size, latent_dim]
        reconstruction_errors : torch.FloatTensor [batch_size] or [batch_size, 1]

        Returns
        -------
        list[SeverityOutput]
            One :class:`SeverityOutput` per sample in the batch.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SeverityClassifier.predict_batch. "
                "Install it with: pip install torch"
            )

        self.eval()
        with torch.no_grad():
            features = self.build_input(latent_vectors, reconstruction_errors)
            logits = self(features)                             # [B, num_classes]
            probs_all = torch.softmax(logits, dim=-1)          # [B, num_classes]
            class_indices = probs_all.argmax(dim=-1).tolist()  # [B]

        results = []
        for i, class_idx in enumerate(class_indices):
            prob_row = probs_all[i].cpu().tolist()
            results.append(SeverityOutput(
                label=SEVERITY_LABELS[class_idx],
                class_index=class_idx,
                confidence=prob_row[class_idx],
                probabilities=prob_row,
            ))
        return results

    # ------------------------------------------------------------------
    # Persistence — mirrors AnomalyDetector / SystemBehaviorModel exactly
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """
        Save model checkpoint (state dict + config) to disk.

        The saved checkpoint contains:

        * ``state_dict`` — all MLP layer weights
        * ``cfg``        — :class:`SeverityClassifierConfig` dataclass

        Parameters
        ----------
        path:
            Destination file path (e.g. ``models/severity_classifier.pt``).
            Parent directories are created automatically.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for SeverityClassifier.save. "
                "Install it with: pip install torch"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)
        logger.info("Saved SeverityClassifier to %s", path)

    @classmethod
    def load(
        cls,
        path: Path | str,
        map_location: str = "cpu",
    ) -> "SeverityClassifier":
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
        SeverityClassifier
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
                "torch is required for SeverityClassifier.load. "
                "Install it with: pip install torch"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # safe_globals allows SeverityClassifierConfig to be deserialised
        # safely under weights_only=True (torch >= 2.0 requirement)
        with torch.serialization.safe_globals([SeverityClassifierConfig]):
            ckpt = torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["state_dict"])
        logger.info(
            "Loaded SeverityClassifier from %s "
            "(input_dim=%d, hidden_dim=%d, num_classes=%d)",
            path,
            ckpt["cfg"].input_dim,
            ckpt["cfg"].hidden_dim,
            ckpt["cfg"].num_classes,
        )
        return model
