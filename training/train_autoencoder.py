#!/usr/bin/env python
# training/train_autoencoder.py
# Phase 4 — Anomaly Detection: train the Denoising Autoencoder.
#
# Strategy:
#   1. Load Word2Vec + LSTM behavior model (both previously trained).
#   2. Run training sequences through the LSTM to generate context vectors.
#   3. Train the AnomalyDetector (Denoising Autoencoder) on context vectors
#      from NORMAL sequences only.
#   4. Calibrate the anomaly threshold at a configurable percentile.
#   5. Save the trained detector to models/anomaly/anomaly_detector.pt.
#
# Input:
#   models/embeddings/word2vec.model
#   models/behavior/behavior_model.pt
#   data/processed/sequences_train.parquet  or  events_tokenized.parquet
#
# Output:
#   models/anomaly/anomaly_detector.pt   (checkpoint + calibrated threshold)
#
# Usage:
#   python -m training.train_autoencoder
#   python -m training.train_autoencoder --latent-dim 32 --epochs 30
#
# Environment variables:
#   LATENT_DIM          autoencoder bottleneck dim   (default: 32)
#   AE_INTERMEDIATE     intermediate hidden dim       (default: 64)
#   AE_DROPOUT          dropout probability           (default: 0.1)
#   AE_NOISE_STD        denoising noise std dev       (default: 0.05)
#   AE_EPOCHS           training epochs               (default: 30)
#   AE_LR               learning rate                 (default: 1e-3)
#   AE_BATCH            batch size                    (default: 128)
#   AE_THRESHOLD_PCT    calibration percentile        (default: 95)
#   WINDOW_SIZE         rolling window length         (default: 10)

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training.train_autoencoder")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_W2V_MODEL = _PROJECT_ROOT / "models" / "embeddings" / "word2vec.model"
_BEHAVIOR_MODEL = _PROJECT_ROOT / "models" / "behavior" / "behavior_model.pt"
_SEQ_TRAIN = _PROJECT_ROOT / "data" / "processed" / "sequences_train.parquet"
_EVENTS_TOK = _PROJECT_ROOT / "data" / "processed" / "events_tokenized.parquet"
_MODEL_OUT = _PROJECT_ROOT / "models" / "anomaly" / "anomaly_detector.pt"


# ---------------------------------------------------------------------------
# Context vector generation
# ---------------------------------------------------------------------------

def _embed_token_id(wv, token_id: int, vec_dim: int) -> "np.ndarray":
    """
    Embed a single token_id via direct Word2Vec KeyedVectors lookup.

    Returns a zero vector for token_ids absent from the vocabulary.
    """
    import numpy as np
    tok = str(token_id)
    return wv[tok].astype(np.float32) if tok in wv else np.zeros(vec_dim, dtype=np.float32)


def _generate_context_vectors(
    w2v_trainer,
    behavior_model,
    window_size: int,
    max_windows: int = 50_000,
):
    """
    Embed log token_id sequences and run them through the LSTM.

    Architecture note
    -----------------
    The training data (sequences_train.parquet / events_tokenized.parquet) stores
    integer token_ids, not raw message text.  Embedding is performed by direct
    Word2Vec KeyedVectors lookup (str(token_id) → vec_dim vector).

    For the autoencoder, training on NORMAL sequences only (label=0) is preferred.
    sequences_train.parquet provides these labels; events_tokenized.parquet is used
    as fallback without label filtering.

    Returns numpy array of shape [N, hidden_dim].
    """
    import numpy as np
    import pandas as pd
    import torch

    wv = w2v_trainer.word_vectors
    vec_dim = w2v_trainer.vec_dim

    # Build per-event embedding array from token_ids
    if _SEQ_TRAIN.exists():
        df = pd.read_parquet(_SEQ_TRAIN)
        if "tokens" in df.columns:
            # Prefer normal-only sequences for autoencoder training
            if "label" in df.columns:
                normal_df = df[df["label"] == 0]
                if len(normal_df) > 0:
                    logger.info(
                        "Using %d normal sequences from sequences_train.parquet "
                        "(filtered label=0)",
                        len(normal_df),
                    )
                    df = normal_df
            all_vecs: list = []
            for tids in df["tokens"]:
                all_vecs.extend([_embed_token_id(wv, tid, vec_dim) for tid in tids])
            vecs_arr = np.array(all_vecs, dtype=np.float32)
            logger.info("Embedded %d token events from sequences_train", len(vecs_arr))
        else:
            raise ValueError(
                f"sequences_train.parquet has no 'tokens' column; "
                f"found: {list(df.columns)}"
            )
    else:
        logger.info("sequences_train not found — loading from %s", _EVENTS_TOK)
        df = pd.read_parquet(_EVENTS_TOK, columns=["token_id"]).iloc[:200_000]
        vecs_arr = np.array(
            [_embed_token_id(wv, tid, vec_dim) for tid in df["token_id"].tolist()],
            dtype=np.float32,
        )
        logger.info("Embedded %d events from events_tokenized", len(vecs_arr))

    # Build rolling windows
    n_windows = min(len(vecs_arr) - window_size + 1, max_windows)
    if n_windows <= 0:
        raise ValueError(
            f"Not enough events ({len(vecs_arr)}) to build windows of size {window_size}."
        )
    windows = np.stack(
        [vecs_arr[i : i + window_size] for i in range(n_windows)],
        axis=0,
    ).astype(np.float32)
    logger.info("Running %d windows through LSTM ...", len(windows))

    device = next(behavior_model.parameters()).device
    behavior_model.eval()
    context_vecs = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
            ctx = behavior_model(batch)
            context_vecs.append(ctx.cpu().numpy())

    context_arr = np.vstack(context_vecs).astype(np.float32)
    logger.info("Context vectors: %s", context_arr.shape)
    return context_arr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_autoencoder(
    context_vectors,
    latent_dim: int,
    intermediate_dim: int,
    dropout: float,
    noise_std: float,
    epochs: int,
    lr: float,
    batch_size: int,
    threshold_percentile: float,
):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.modeling.anomaly.autoencoder import AnomalyDetector, AnomalyDetectorConfig

    input_dim = context_vectors.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training AnomalyDetector: input_dim=%d latent_dim=%d device=%s",
        input_dim, latent_dim, device,
    )

    cfg = AnomalyDetectorConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        intermediate_dim=intermediate_dim,
        dropout=dropout,
        noise_std=noise_std,
    )
    model = AnomalyDetector(cfg).to(device)

    x_t = torch.from_numpy(context_vectors).to(device)
    dataset = TensorDataset(x_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = output.error.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info("Epoch %3d/%d  recon_loss=%.6f", epoch, epochs, avg)

    # Calibrate threshold on all training errors (normal data)
    model.eval()
    all_errors = model.score(x_t)
    model.fit_threshold(all_errors, percentile=threshold_percentile)
    logger.info(
        "Threshold calibrated at p%.0f: %.6f",
        threshold_percentile, model.threshold,
    )

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    latent_dim: int,
    intermediate_dim: int,
    dropout: float,
    noise_std: float,
    epochs: int,
    lr: float,
    batch_size: int,
    threshold_percentile: float,
    window_size: int,
) -> None:
    for p in (_W2V_MODEL, _BEHAVIOR_MODEL):
        if not p.exists():
            raise FileNotFoundError(
                f"Required model not found: {p}. "
                "Run train_embeddings.py and train_behavior_model.py first."
            )

    import torch

    from src.modeling.behavior.lstm_model import SystemBehaviorModel
    from src.modeling.embeddings.word2vec_trainer import Word2VecTrainer

    # Load prerequisites
    w2v = Word2VecTrainer()
    w2v.load(_W2V_MODEL)
    logger.info("Word2Vec loaded: vec_dim=%d", w2v.vec_dim)

    behavior_model = SystemBehaviorModel.load(_BEHAVIOR_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    behavior_model = behavior_model.to(device)
    logger.info("Behavior model loaded: hidden_dim=%d", behavior_model.hidden_dim)

    # Generate context vectors
    context_vecs = _generate_context_vectors(
        w2v, behavior_model, window_size
    )

    # Train autoencoder
    ae_model = _train_autoencoder(
        context_vectors=context_vecs,
        latent_dim=latent_dim,
        intermediate_dim=intermediate_dim,
        dropout=dropout,
        noise_std=noise_std,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        threshold_percentile=threshold_percentile,
    )

    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    ae_model.save(_MODEL_OUT)
    logger.info("AnomalyDetector saved to %s", _MODEL_OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Denoising Autoencoder for v2 anomaly detection."
    )
    parser.add_argument("--latent-dim", type=int,
                        default=int(os.environ.get("LATENT_DIM", "32")))
    parser.add_argument("--intermediate-dim", type=int,
                        default=int(os.environ.get("AE_INTERMEDIATE", "64")))
    parser.add_argument("--dropout", type=float,
                        default=float(os.environ.get("AE_DROPOUT", "0.1")))
    parser.add_argument("--noise-std", type=float,
                        default=float(os.environ.get("AE_NOISE_STD", "0.05")))
    parser.add_argument("--epochs", type=int,
                        default=int(os.environ.get("AE_EPOCHS", "30")))
    parser.add_argument("--lr", type=float,
                        default=float(os.environ.get("AE_LR", "1e-3")))
    parser.add_argument("--batch-size", type=int,
                        default=int(os.environ.get("AE_BATCH", "128")))
    parser.add_argument("--threshold-pct", type=float,
                        default=float(os.environ.get("AE_THRESHOLD_PCT", "95")))
    parser.add_argument("--window-size", type=int,
                        default=int(os.environ.get("WINDOW_SIZE", "10")))
    args = parser.parse_args()
    main(
        latent_dim=args.latent_dim,
        intermediate_dim=args.intermediate_dim,
        dropout=args.dropout,
        noise_std=args.noise_std,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        threshold_percentile=args.threshold_pct,
        window_size=args.window_size,
    )
