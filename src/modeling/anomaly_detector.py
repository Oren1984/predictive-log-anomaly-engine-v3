# src/modeling/anomaly_detector.py
# Stage 4: Anomaly Detection
#
# AnomalyDetector is a Denoising Autoencoder trained on context vectors
# produced by SystemBehaviorModel (Phase 4).  It learns to compress and
# reconstruct *normal* system behaviour.  At inference time, abnormal
# sequences produce a high reconstruction error — the anomaly score.
#
# Design notes:
#   - Inherits from nn.Module with the same lazy-torch and save/load
#     patterns established in Phase 4 (SystemBehaviorModel).
#   - AnomalyDetectorConfig mirrors BehaviorModelConfig / TransformerConfig.
#   - Encoder and decoder are each 2-layer nn.Sequential blocks with ReLU
#     and Dropout, matching the depth used elsewhere in the pipeline.
#   - Denoising: Gaussian noise is injected into the input at training time
#     only (noise_std > 0); the reconstruction target is always the clean
#     original input.  noise_std = 0 reduces to a plain autoencoder.
#   - forward() returns an AEOutput NamedTuple: (latent, reconstructed, error)
#     so callers can unpack by name without positional confusion.
#   - reconstruction_error() is a standalone helper accepting separate
#     original/reconstructed tensors (e.g. for external training loops).
#   - fit_threshold() calibrates self.threshold from a collection of
#     normal-sequence errors using a percentile rule; the threshold is
#     persisted in the checkpoint so calibration survives save/load.
#   - The existing IsolationForest baseline (src/modeling/baseline/) is
#     unchanged and remains the active fallback.
#   - This class is completely isolated from the existing runtime pipeline.

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Union

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
# AEOutput — named return type for forward()
# ---------------------------------------------------------------------------

class AEOutput(NamedTuple):
    """
    Named output of :meth:`AnomalyDetector.forward`.

    Attributes
    ----------
    latent : torch.FloatTensor [batch, latent_dim]
        Compressed bottleneck representation.
    reconstructed : torch.FloatTensor [batch, input_dim]
        Decoder output — the model's attempt to reconstruct the input.
    error : torch.FloatTensor [batch]
        Per-sample mean-squared error between the clean input and
        ``reconstructed``.  Used directly as the anomaly score.
    """
    latent: object          # torch.Tensor [B, latent_dim]
    reconstructed: object   # torch.Tensor [B, input_dim]
    error: object           # torch.Tensor [B]


# ---------------------------------------------------------------------------
# AnomalyDetectorConfig
# ---------------------------------------------------------------------------

@dataclass
class AnomalyDetectorConfig:
    """
    Hyperparameters for :class:`AnomalyDetector`.

    Mirrors the pattern of ``BehaviorModelConfig`` and ``TransformerConfig``.

    Parameters
    ----------
    input_dim:
        Size of the context vector from ``SystemBehaviorModel``.
        Must match ``BehaviorModelConfig.hidden_dim`` (default 128).
    latent_dim:
        Bottleneck dimension (default 32).  Compression ratio with
        default settings: 128 → 64 → 32 (encoder).
    intermediate_dim:
        Hidden-layer size in both encoder and decoder (default 64).
        Must satisfy ``latent_dim < intermediate_dim < input_dim`` for
        meaningful compression.
    dropout:
        Dropout probability applied after the first linear layer in
        both encoder and decoder (default 0.1).
    noise_std:
        Standard deviation of the Gaussian noise added to the input
        during training to implement denoising.  Set to ``0.0`` to
        use a plain (non-denoising) autoencoder (default 0.05).
    """

    input_dim: int = 128
    latent_dim: int = 32
    intermediate_dim: int = 64
    dropout: float = 0.1
    noise_std: float = 0.05

    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Serialise config to a JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "AnomalyDetectorConfig":
        """Load config from a JSON file."""
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector(_TorchModule):  # type: ignore[misc]
    """
    Stage 4: Anomaly Detection (Denoising Autoencoder).

    Self-supervised: trained **exclusively on normal sequences**.
    Anomalous sequences produce high reconstruction error because the
    model has never seen their distribution during training.

    Architecture
    ------------
    ::

        Input  [B, input_dim]
            |
            | [Denoising — training only, noise_std > 0]
            | x_enc = x + N(0, noise_std²)
            v
        Encoder  Linear(input_dim → intermediate_dim) → ReLU → Dropout
                 Linear(intermediate_dim → latent_dim)
            |
        latent  [B, latent_dim]
            |
        Decoder  Linear(latent_dim → intermediate_dim) → ReLU → Dropout
                 Linear(intermediate_dim → input_dim)
            |
        reconstructed  [B, input_dim]
            |
        error = MSE(x, reconstructed).mean(dim=-1)  [B]

    Note: reconstruction error is always computed against the **clean**
    input ``x``, not the noisy ``x_enc``.

    Parameters
    ----------
    cfg:
        An :class:`AnomalyDetectorConfig` instance.  If omitted, default
        hyperparameters are used.

    Raises
    ------
    RuntimeError
        If ``forward``, ``score``, ``save``, or ``load`` is called
        and torch is not installed.
    """

    def __init__(self, cfg: Optional[AnomalyDetectorConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = AnomalyDetectorConfig()
        self.cfg = cfg

        # Expose config fields directly for inspection
        self.input_dim: int = cfg.input_dim
        self.latent_dim: int = cfg.latent_dim

        # Anomaly threshold — set by fit_threshold(); 0.0 = uncalibrated
        self.threshold: float = 0.0
        self._calibrated: bool = False

        if _TORCH_AVAILABLE:
            self._build_network()
        else:
            self._encoder: Optional[object] = None
            self._decoder: Optional[object] = None

        logger.debug(
            "AnomalyDetector: input_dim=%d latent_dim=%d "
            "intermediate_dim=%d noise_std=%.3f",
            self.input_dim, self.latent_dim,
            cfg.intermediate_dim, cfg.noise_std,
        )

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(self) -> None:
        """Instantiate encoder and decoder nn.Sequential blocks."""
        cfg = self.cfg

        # Encoder: input_dim → intermediate_dim → latent_dim
        self._encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.intermediate_dim, cfg.latent_dim),
        )

        # Decoder: latent_dim → intermediate_dim → input_dim
        # No activation on output layer — reconstruction is unbounded
        self._decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.intermediate_dim, cfg.input_dim),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: "torch.Tensor") -> AEOutput:
        """
        Run the denoising autoencoder over a batch of context vectors.

        Parameters
        ----------
        x : torch.FloatTensor
            Shape ``[batch_size, input_dim]``.  Context vectors produced
            by ``SystemBehaviorModel.forward`` (Phase 4).

        Returns
        -------
        AEOutput
            Named tuple ``(latent, reconstructed, error)`` where:

            - ``latent``        — ``[B, latent_dim]``   bottleneck vector
            - ``reconstructed`` — ``[B, input_dim]``    decoder output
            - ``error``         — ``[B]``               per-sample MSE
              between **clean** ``x`` and ``reconstructed``

        Raises
        ------
        RuntimeError
            If torch is not installed or input shape is invalid.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for AnomalyDetector.forward. "
                "Install it with: pip install torch"
            )
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected input tensor of shape [batch, input_dim], "
                f"got {tuple(x.shape)} (ndim={x.ndim})"
            )
        if x.size(1) != self.cfg.input_dim:
            raise RuntimeError(
                f"Input vector size {x.size(1)} does not match "
                f"model input_dim {self.cfg.input_dim}"
            )

        # Denoising: inject Gaussian noise during training only
        if self.training and self.cfg.noise_std > 0.0:
            noise = torch.randn_like(x) * self.cfg.noise_std
            x_enc = x + noise
        else:
            x_enc = x

        latent = self._encoder(x_enc)           # [B, latent_dim]
        reconstructed = self._decoder(latent)   # [B, input_dim]

        # Error is MSE against the CLEAN original x — not the noisy x_enc
        error = self.reconstruction_error(x, reconstructed)  # [B]

        return AEOutput(latent=latent, reconstructed=reconstructed, error=error)

    # ------------------------------------------------------------------
    # Reconstruction error helper
    # ------------------------------------------------------------------

    def reconstruction_error(
        self,
        original: "torch.Tensor",
        reconstructed: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Per-sample mean-squared error between ``original`` and
        ``reconstructed``.

        Parameters
        ----------
        original : torch.FloatTensor [batch, input_dim]
        reconstructed : torch.FloatTensor [batch, input_dim]

        Returns
        -------
        torch.FloatTensor [batch]
            One MSE scalar per sample.  Higher = more anomalous.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for AnomalyDetector.reconstruction_error."
            )
        return ((original - reconstructed) ** 2).mean(dim=-1)

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """``True`` after :meth:`fit_threshold` has been called."""
        return self._calibrated

    def fit_threshold(
        self,
        normal_errors: Union[List[float], "np.ndarray", "torch.Tensor"],
        percentile: float = 95.0,
    ) -> None:
        """
        Calibrate the anomaly threshold from a set of reconstruction
        errors collected on **normal** validation sequences.

        The threshold is set to the ``percentile``-th percentile of the
        provided errors.  Windows whose error exceeds the threshold are
        flagged as anomalous by :meth:`is_anomaly`.

        Parameters
        ----------
        normal_errors:
            1-D collection of per-sample reconstruction errors from
            normal (label=0) sequences.  May be a Python list, a
            numpy array, or a 1-D torch tensor.
        percentile:
            Percentile to use for the threshold (default 95.0).
            A value of 95 means that up to 5 % of normal windows may
            be flagged as false positives.

        Raises
        ------
        ValueError
            If ``normal_errors`` is empty or ``percentile`` is not in
            ``(0, 100]``.
        """
        if not (0.0 < percentile <= 100.0):
            raise ValueError(
                f"percentile must be in (0, 100], got {percentile}"
            )

        # Normalise to numpy for percentile computation
        if _TORCH_AVAILABLE and isinstance(normal_errors, torch.Tensor):
            arr = normal_errors.detach().cpu().numpy()
        else:
            arr = np.asarray(normal_errors, dtype=np.float32)

        if arr.size == 0:
            raise ValueError("normal_errors must not be empty")

        self.threshold = float(np.percentile(arr, percentile))
        self._calibrated = True
        logger.info(
            "AnomalyDetector threshold calibrated: %.6f "
            "(p%.0f of %d normal errors)",
            self.threshold, percentile, arr.size,
        )

    def is_anomaly(self, error: float) -> bool:
        """
        Return ``True`` if ``error`` exceeds the calibrated threshold.

        Parameters
        ----------
        error:
            Scalar reconstruction error for a single window, e.g.
            ``output.error[i].item()``.

        Returns
        -------
        bool

        .. warning::
            Returns ``False`` for all inputs when ``threshold == 0.0``
            (uncalibrated default).  Call :meth:`fit_threshold` first
            for meaningful anomaly detection.
        """
        if not self._calibrated:
            logger.warning(
                "AnomalyDetector.is_anomaly called before fit_threshold(); "
                "threshold is 0.0 — results are not meaningful."
            )
        return float(error) > self.threshold

    # ------------------------------------------------------------------
    # Inference convenience
    # ------------------------------------------------------------------

    def score(
        self,
        context_vector: "torch.Tensor",
    ) -> "np.ndarray":
        """
        Run inference and return per-sample reconstruction errors as a
        float32 numpy array.

        Switches the model to eval mode, disables gradients, and
        suppresses noise injection.

        Parameters
        ----------
        context_vector : torch.FloatTensor [batch, input_dim]
            Context vectors from ``SystemBehaviorModel``.

        Returns
        -------
        np.ndarray [batch], dtype float32
            Per-sample reconstruction errors.  Higher = more anomalous.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for AnomalyDetector.score. "
                "Install it with: pip install torch"
            )
        self.eval()
        with torch.no_grad():
            output = self(context_vector)
        return output.error.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence — mirrors SystemBehaviorModel / NextTokenTransformerModel
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """
        Save model checkpoint (state dict + config + threshold) to disk.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``models/anomaly_detector.pt``).
            Parent directories are created automatically.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for AnomalyDetector.save. "
                "Install it with: pip install torch"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "cfg": self.cfg,
                "threshold": self.threshold,
                "calibrated": self._calibrated,
            },
            path,
        )
        logger.info(
            "Saved AnomalyDetector to %s (threshold=%.6f, calibrated=%s)",
            path, self.threshold, self._calibrated,
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
        map_location: str = "cpu",
    ) -> "AnomalyDetector":
        """
        Load a previously saved model checkpoint from disk.

        The calibrated threshold (if any) is restored from the
        checkpoint so calibration survives the save/load cycle.

        Parameters
        ----------
        path:
            Path to a checkpoint saved by :meth:`save`.
        map_location:
            Torch device string (default ``"cpu"``).

        Returns
        -------
        AnomalyDetector
            A new instance with weights, config, and threshold restored.

        Raises
        ------
        RuntimeError
            If torch is not installed.
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is required for AnomalyDetector.load. "
                "Install it with: pip install torch"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with torch.serialization.safe_globals([AnomalyDetectorConfig]):
            ckpt = torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["state_dict"])
        model.threshold = ckpt.get("threshold", 0.0)
        model._calibrated = ckpt.get("calibrated", False)
        logger.info(
            "Loaded AnomalyDetector from %s "
            "(latent_dim=%d, threshold=%.6f, calibrated=%s)",
            path, ckpt["cfg"].latent_dim,
            model.threshold, model._calibrated,
        )
        return model
