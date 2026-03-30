#!/usr/bin/env python
# training/train_severity_model.py
# Phase 5 — Severity Prediction: train the MLP severity classifier.
#
# Strategy:
#   1. Load all previously trained v2 models (Word2Vec, LSTM, Autoencoder).
#   2. Run the full pipeline on all training sequences to produce:
#        - latent vectors  [N, latent_dim]
#        - anomaly scores  [N]
#   3. Generate synthetic severity labels from the anomaly score distribution:
#        - score < p33  → "info"   (class 0)
#        - p33 ≤ score < p66 → "warning" (class 1)
#        - score ≥ p66  → "critical" (class 2)
#      This is a training bootstrap; replace with real labels when available.
#   4. Train the SeverityClassifier MLP on (latent + score) → label.
#   5. Save to models/severity/severity_classifier.pt.
#
# Input:
#   models/embeddings/word2vec.model
#   models/behavior/behavior_model.pt
#   models/anomaly/anomaly_detector.pt
#
# Output:
#   models/severity/severity_classifier.pt
#
# Usage:
#   python -m training.train_severity_model
#
# Environment variables:
#   SEV_HIDDEN_DIM   MLP hidden dim            (default: 64)
#   SEV_DROPOUT      dropout                   (default: 0.3)
#   SEV_EPOCHS       training epochs           (default: 30)
#   SEV_LR           learning rate             (default: 1e-3)
#   SEV_BATCH        batch size                (default: 128)
#   WINDOW_SIZE      rolling window length     (default: 10)

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
logger = logging.getLogger("training.train_severity_model")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_W2V_MODEL = _PROJECT_ROOT / "models" / "embeddings" / "word2vec.model"
_BEHAVIOR_MODEL = _PROJECT_ROOT / "models" / "behavior" / "behavior_model.pt"
_AE_MODEL = _PROJECT_ROOT / "models" / "anomaly" / "anomaly_detector.pt"
_EVENTS_TOK = _PROJECT_ROOT / "data" / "processed" / "events_tokenized.parquet"
_SEQ_TRAIN = _PROJECT_ROOT / "data" / "processed" / "sequences_train.parquet"
_MODEL_OUT = _PROJECT_ROOT / "models" / "severity" / "severity_classifier.pt"


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _embed_token_id(wv, token_id: int, vec_dim: int) -> "np.ndarray":
    """
    Embed a single token_id via direct Word2Vec KeyedVectors lookup.

    Returns a zero vector for token_ids absent from the vocabulary.
    """
    import numpy as np
    tok = str(token_id)
    return wv[tok].astype(np.float32) if tok in wv else np.zeros(vec_dim, dtype=np.float32)


def _extract_features(
    w2v_trainer,
    behavior_model,
    ae_model,
    window_size: int,
    max_windows: int = 20_000,
):
    """
    Run the full v2 pipeline to produce (latent_vectors, anomaly_scores).

    Architecture note
    -----------------
    The training data stores integer token_ids, not raw message text.
    Embedding uses direct Word2Vec KeyedVectors lookup (str(token_id) → vector).
    sequences_train.parquet is preferred; events_tokenized.parquet is the fallback.

    Returns:
        latent_vectors  numpy [N, latent_dim]
        anomaly_scores  numpy [N]
    """
    import numpy as np
    import pandas as pd
    import torch

    wv = w2v_trainer.word_vectors
    vec_dim = w2v_trainer.vec_dim

    # Build per-event embedding array from token_ids
    src = _SEQ_TRAIN if _SEQ_TRAIN.exists() else _EVENTS_TOK
    logger.info("Loading token data from %s", src)
    df = pd.read_parquet(src)

    if "tokens" in df.columns:
        # sequences_train.parquet: each row is a sequence (list of token_ids)
        all_vecs: list = []
        for tids in df["tokens"]:
            all_vecs.extend([_embed_token_id(wv, tid, vec_dim) for tid in tids])
        vecs_arr = np.array(all_vecs, dtype=np.float32)
    elif "token_id" in df.columns:
        # events_tokenized.parquet: one token_id per row
        vecs_arr = np.array(
            [_embed_token_id(wv, tid, vec_dim) for tid in df["token_id"].tolist()],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"No usable column in {src}. Expected 'tokens' or 'token_id'; "
            f"found: {list(df.columns)}"
        )

    logger.info("Embedded %d events → array %s", len(vecs_arr), vecs_arr.shape)

    # Build windows
    n_windows = min(len(vecs_arr) - window_size + 1, max_windows)
    if n_windows <= 0:
        raise ValueError(
            f"Not enough events ({len(vecs_arr)}) to build windows of size {window_size}."
        )
    windows = np.stack(
        [vecs_arr[i : i + window_size] for i in range(n_windows)],
        axis=0,
    ).astype(np.float32)
    logger.info("Built %d windows", n_windows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    behavior_model.eval()
    ae_model.eval()

    latents = []
    scores = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
            ctx = behavior_model(batch)                     # [B, hidden_dim]
            ae_out = ae_model(ctx)                          # AEOutput
            latents.append(ae_out.latent.cpu().numpy())
            scores.append(ae_out.error.cpu().numpy())

    latent_arr = np.vstack(latents).astype(np.float32)
    score_arr = np.concatenate(scores).astype(np.float32)
    logger.info(
        "Features extracted: latent=%s scores=%s",
        latent_arr.shape, score_arr.shape,
    )
    return latent_arr, score_arr


def _make_synthetic_labels(scores) -> "np.ndarray":
    """
    Bootstrap severity labels from score percentiles.

    p0–p33  → 0 (info)
    p33–p66 → 1 (warning)
    p66–p100 → 2 (critical)
    """
    import numpy as np

    p33 = float(np.percentile(scores, 33))
    p66 = float(np.percentile(scores, 66))
    labels = np.zeros(len(scores), dtype=np.int64)
    labels[scores >= p33] = 1
    labels[scores >= p66] = 2
    counts = [int((labels == i).sum()) for i in range(3)]
    logger.info(
        "Synthetic labels: info=%d warning=%d critical=%d (p33=%.4f p66=%.4f)",
        counts[0], counts[1], counts[2], p33, p66,
    )
    return labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_classifier(
    latent_vectors,
    anomaly_scores,
    labels,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.modeling.severity.severity_classifier import (
        SeverityClassifier,
        SeverityClassifierConfig,
    )

    latent_dim = latent_vectors.shape[1]
    input_dim = latent_dim + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training SeverityClassifier: input_dim=%d hidden_dim=%d device=%s",
        input_dim, hidden_dim, device,
    )

    cfg = SeverityClassifierConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=3,
        dropout=dropout,
    )
    model = SeverityClassifier(cfg).to(device)

    # Build combined feature tensor [N, latent_dim + 1]
    scores_col = anomaly_scores.reshape(-1, 1)
    features = np.concatenate([latent_vectors, scores_col], axis=1).astype(np.float32)

    x_t = torch.from_numpy(features).to(device)
    y_t = torch.from_numpy(labels).long().to(device)

    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info("Epoch %3d/%d  cross_entropy=%.6f", epoch, epochs, avg)

    # Quick accuracy check
    model.eval()
    with torch.no_grad():
        logits_all = model(x_t)
        preds = logits_all.argmax(dim=-1).cpu().numpy()
    acc = (preds == labels).mean()
    logger.info("Training accuracy: %.2f%%", acc * 100)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    window_size: int,
) -> None:
    for p in (_W2V_MODEL, _BEHAVIOR_MODEL, _AE_MODEL):
        if not p.exists():
            raise FileNotFoundError(
                f"Required model not found: {p}. "
                "Run train_embeddings.py, train_behavior_model.py, "
                "and train_autoencoder.py first."
            )

    import torch

    from src.modeling.anomaly.autoencoder import AnomalyDetector
    from src.modeling.behavior.lstm_model import SystemBehaviorModel
    from src.modeling.embeddings.word2vec_trainer import Word2VecTrainer

    # Load prerequisites
    w2v = Word2VecTrainer()
    w2v.load(_W2V_MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    behavior_model = SystemBehaviorModel.load(_BEHAVIOR_MODEL).to(device)
    ae_model = AnomalyDetector.load(_AE_MODEL).to(device)

    logger.info(
        "Models loaded: vec_dim=%d hidden_dim=%d latent_dim=%d",
        w2v.vec_dim, behavior_model.hidden_dim, ae_model.latent_dim,
    )

    # Extract features
    latent_vecs, scores = _extract_features(
        w2v, behavior_model, ae_model, window_size
    )

    # Synthetic labels (bootstrap)
    labels = _make_synthetic_labels(scores)

    # Train classifier
    clf = _train_classifier(
        latent_vectors=latent_vecs,
        anomaly_scores=scores,
        labels=labels,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    clf.save(_MODEL_OUT)
    logger.info("SeverityClassifier saved to %s", _MODEL_OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the MLP severity classifier for v2 pipeline."
    )
    parser.add_argument("--hidden-dim", type=int,
                        default=int(os.environ.get("SEV_HIDDEN_DIM", "64")))
    parser.add_argument("--dropout", type=float,
                        default=float(os.environ.get("SEV_DROPOUT", "0.3")))
    parser.add_argument("--epochs", type=int,
                        default=int(os.environ.get("SEV_EPOCHS", "30")))
    parser.add_argument("--lr", type=float,
                        default=float(os.environ.get("SEV_LR", "1e-3")))
    parser.add_argument("--batch-size", type=int,
                        default=int(os.environ.get("SEV_BATCH", "128")))
    parser.add_argument("--window-size", type=int,
                        default=int(os.environ.get("WINDOW_SIZE", "10")))
    args = parser.parse_args()
    main(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        window_size=args.window_size,
    )
