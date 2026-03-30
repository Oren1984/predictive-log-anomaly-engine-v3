#!/usr/bin/env python
# training/train_behavior_model.py
# Phase 3 — Behavior Modeling: train the LSTM on embedded log sequences.
#
# Input:
#   models/embeddings/word2vec.model           (trained by train_embeddings.py)
#   data/processed/sequences_train.parquet     (preferred)
#   data/processed/events_tokenized.parquet    (fallback — uses synthetic windows)
#
# Output:
#   models/behavior/behavior_model.pt          (PyTorch checkpoint)
#
# Usage:
#   python -m training.train_behavior_model
#   python -m training.train_behavior_model --hidden-dim 128 --epochs 20
#
# Environment variables:
#   HIDDEN_DIM     LSTM hidden state size        (default: 128)
#   NUM_LAYERS     stacked LSTM layers           (default: 2)
#   LSTM_DROPOUT   dropout between LSTM layers   (default: 0.2)
#   LSTM_EPOCHS    training epochs               (default: 20)
#   LSTM_LR        learning rate                 (default: 1e-3)
#   LSTM_BATCH     batch size                    (default: 64)
#   WINDOW_SIZE    sequence window length        (default: 10)

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
logger = logging.getLogger("training.train_behavior_model")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_W2V_MODEL = _PROJECT_ROOT / "models" / "embeddings" / "word2vec.model"
_SEQ_TRAIN = _PROJECT_ROOT / "data" / "processed" / "sequences_train.parquet"
_EVENTS_TOK = _PROJECT_ROOT / "data" / "processed" / "events_tokenized.parquet"
_MODEL_OUT = _PROJECT_ROOT / "models" / "behavior" / "behavior_model.pt"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _embed_token_id(wv, token_id: int, vec_dim: int) -> "np.ndarray":
    """
    Embed a single token_id via direct Word2Vec KeyedVectors lookup.

    The Word2Vec model was trained on string representations of token_ids
    (e.g., token_id=5413 → vocabulary word "5413"), so the lookup key is
    simply str(token_id).  Returns a zero vector for out-of-vocabulary ids.
    """
    import numpy as np
    tok = str(token_id)
    return wv[tok].astype(np.float32) if tok in wv else np.zeros(vec_dim, dtype=np.float32)


def _load_sequences_from_parquet(
    w2v_trainer,
    window_size: int,
) -> "list[np.ndarray]":
    """
    Load embedded log sequences for LSTM training.

    Architecture note
    -----------------
    Both sequences_train.parquet and events_tokenized.parquet store integer
    token_ids rather than raw message text.  This function embeds each token_id
    via direct Word2Vec KeyedVectors lookup (str(token_id) → vec_dim vector)
    and builds rolling windows of shape [window_size, vec_dim].

    Source priority
    ---------------
    1. sequences_train.parquet — 'tokens' column (list of ints per row, length
       12–42).  Produces ≥ (len-window_size+1) windows per sequence.
    2. events_tokenized.parquet — global windowing over first 200K events if
       sequences_train is absent.

    Returns
    -------
    list of np.ndarray, each shaped [window_size, vec_dim]
    """
    import numpy as np
    import pandas as pd

    wv = w2v_trainer.word_vectors
    vec_dim = w2v_trainer.vec_dim

    if _SEQ_TRAIN.exists():
        logger.info("Building sequences from %s", _SEQ_TRAIN)
        df = pd.read_parquet(_SEQ_TRAIN)
        if "tokens" in df.columns:
            sequences: list = []
            for tids in df["tokens"]:
                if len(tids) < window_size:
                    continue
                vecs = np.array(
                    [_embed_token_id(wv, tid, vec_dim) for tid in tids],
                    dtype=np.float32,
                )
                for i in range(len(vecs) - window_size + 1):
                    sequences.append(vecs[i : i + window_size])
            if sequences:
                logger.info(
                    "Built %d windows from sequences_train.parquet (window=%d)",
                    len(sequences), window_size,
                )
                return sequences
            logger.warning(
                "sequences_train.parquet 'tokens' produced 0 windows "
                "(all sequences shorter than window_size=%d).",
                window_size,
            )

    # Fallback: global windowing over events_tokenized.parquet
    logger.info(
        "sequences_train.parquet not usable — global windowing over %s",
        _EVENTS_TOK,
    )
    df = pd.read_parquet(_EVENTS_TOK, columns=["token_id"]).iloc[:200_000]
    all_tids = df["token_id"].tolist()
    all_vecs = np.array(
        [_embed_token_id(wv, tid, vec_dim) for tid in all_tids],
        dtype=np.float32,
    )
    stride = max(1, (len(all_vecs) - window_size) // 10_000)
    sequences = []
    for i in range(0, len(all_vecs) - window_size + 1, stride):
        sequences.append(all_vecs[i : i + window_size])
    logger.info(
        "Global windowing: %d windows (stride=%d, window=%d)",
        len(sequences), stride, window_size,
    )
    return sequences


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    sequences,
    vec_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
) -> "SystemBehaviorModel":
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.modeling.behavior.lstm_model import BehaviorModelConfig, SystemBehaviorModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # Stack sequences into a single tensor [N, T, D]
    x = np.stack(sequences, axis=0).astype(np.float32)   # [N, window, vec_dim]
    logger.info("Training tensor shape: %s", x.shape)
    x_t = torch.from_numpy(x).to(device)

    dataset = TensorDataset(x_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cfg = BehaviorModelConfig(
        input_dim=vec_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = SystemBehaviorModel(cfg).to(device)

    # Self-supervised objective: predict the mean embedding of the window
    # (sequence-to-vector regression).  This trains the LSTM to produce a
    # meaningful context vector without requiring anomaly labels.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            context = model(batch)                   # [B, hidden_dim]
            # Target: mean of the input sequence projected to hidden_dim
            # via a simple average (unsupervised proxy target).
            # We regress the context onto the mean of the first hidden_dim
            # dimensions of the input to force the LSTM to summarise the window.
            target_raw = batch.mean(dim=1)           # [B, vec_dim]
            # Trim or zero-pad target to hidden_dim
            if vec_dim >= hidden_dim:
                target = target_raw[:, :hidden_dim]
            else:
                pad = torch.zeros(
                    target_raw.size(0), hidden_dim - vec_dim, device=device
                )
                target = torch.cat([target_raw, pad], dim=1)
            loss = criterion(context, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info("Epoch %3d/%d  loss=%.6f", epoch, epochs, avg_loss)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    window_size: int,
) -> None:
    if not _W2V_MODEL.exists():
        raise FileNotFoundError(
            f"Word2Vec model not found at {_W2V_MODEL}. "
            "Run training/train_embeddings.py first."
        )

    from src.modeling.embeddings.word2vec_trainer import Word2VecTrainer

    trainer = Word2VecTrainer()
    trainer.load(_W2V_MODEL)
    vec_dim = trainer.vec_dim
    logger.info("Loaded Word2Vec: vec_dim=%d", vec_dim)

    sequences = _load_sequences_from_parquet(trainer, window_size)
    if not sequences:
        raise ValueError("No training sequences available.")

    model = _train(
        sequences=sequences,
        vec_dim=vec_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(_MODEL_OUT)
    logger.info("Behavior model saved to %s", _MODEL_OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the LSTM behavior model for the v2 pipeline."
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(os.environ.get("HIDDEN_DIM", "128")),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=int(os.environ.get("NUM_LAYERS", "2")),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=float(os.environ.get("LSTM_DROPOUT", "0.2")),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("LSTM_EPOCHS", "20")),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(os.environ.get("LSTM_LR", "1e-3")),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("LSTM_BATCH", "64")),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.environ.get("WINDOW_SIZE", "10")),
    )
    args = parser.parse_args()
    main(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        window_size=args.window_size,
    )
