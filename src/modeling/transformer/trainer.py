# src/modeling/transformer/trainer.py

# Purpose: Define the Trainer class, which handles the training loop for a NextTokenTransformerModel.

# Input: The Trainer takes a TransformerConfig object and a device string ("cpu" or "cuda") as input.

# Output: The Trainer class provides a train method that takes in training and validation sequences, 
# and an optional save path for the model. 
# It runs the training loop with early stopping based on validation loss.

# Used by: The Trainer class is used to train a NextTokenTransformerModel on a list of Sequence objects.


"""Stage 4B — Transformer: training loop."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ...sequencing.models import Sequence
from .config import TransformerConfig
from .model import NextTokenTransformerModel

log = logging.getLogger(__name__)


def _make_batches(
    sequences: list[Sequence],
    batch_size: int,
    pad_id: int,
    max_seq_len: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Yield (input_ids, target_ids, key_padding_mask) batches.

    input_ids  : tokens[:-1] left-shifted
    target_ids : tokens[1:]  right-shifted (next token labels)
    Both are padded to the longest sequence in the batch, capped at max_seq_len.
    """
    n = len(sequences)
    indices = list(range(n))
    for start in range(0, n, batch_size):
        batch_seqs = [sequences[i] for i in indices[start:start + batch_size]]
        # Truncate to max_seq_len + 1 so we can shift by 1
        raw = [s.tokens[: max_seq_len + 1] for s in batch_seqs]
        max_len = max(len(r) for r in raw)
        if max_len < 2:
            continue  # can't form a (input, target) pair

        inp_list, tgt_list, mask_list = [], [], []
        for r in raw:
            inp = r[:-1]                    # input: all but last
            tgt = r[1:]                     # target: all but first
            pad_len = (max_len - 1) - len(inp)
            inp_pad = inp + [pad_id] * pad_len
            tgt_pad = tgt + [-100] * pad_len   # -100 = ignore_index in CE loss
            mask    = [False] * len(inp) + [True] * pad_len
            inp_list.append(inp_pad)
            tgt_list.append(tgt_pad)
            mask_list.append(mask)

        yield (
            torch.tensor(inp_list, dtype=torch.long),
            torch.tensor(tgt_list, dtype=torch.long),
            torch.tensor(mask_list, dtype=torch.bool),
        )


class Trainer:
    """
    Trains a NextTokenTransformerModel on a list of Sequences.

    Parameters
    ----------
    cfg      : TransformerConfig
    device   : "cpu" | "cuda" (default "cpu")
    """

    def __init__(self, cfg: TransformerConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = NextTokenTransformerModel(cfg).to(self.device)

    # ------------------------------------------------------------------
    def train(
        self,
        train_seqs: list[Sequence],
        val_seqs: list[Sequence],
        save_path: Optional[Path | str] = None,
    ) -> dict:
        """
        Run the training loop with early stopping on val loss.

        Returns a metrics dict with train_loss / val_loss history.
        """
        cfg = self.cfg
        optimizer = AdamW(self.model.parameters(),
                          lr=cfg.learning_rate,
                          weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        best_val_loss = float("inf")
        patience_left = cfg.patience
        history: dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, cfg.max_epochs + 1):
            # ---- train ----
            self.model.train()
            train_loss, train_tokens = 0.0, 0
            for inp, tgt, mask in _make_batches(
                    train_seqs, cfg.batch_size, cfg.pad_id, cfg.max_seq_len):
                inp, tgt, mask = inp.to(self.device), tgt.to(self.device), mask.to(self.device)
                optimizer.zero_grad()
                logits = self.model(inp, key_padding_mask=mask)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), tgt.reshape(B * T))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                n_tok = (tgt != -100).sum().item()
                train_loss += loss.item() * n_tok
                train_tokens += n_tok
            scheduler.step()
            avg_train = train_loss / max(train_tokens, 1)

            # ---- val ----
            avg_val = self._eval_loss(val_seqs, criterion)
            history["train_loss"].append(round(avg_train, 4))
            history["val_loss"].append(round(avg_val, 4))
            log.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                     epoch, cfg.max_epochs, avg_train, avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_left = cfg.patience
                if save_path:
                    self.model.save(save_path)
            else:
                patience_left -= 1
                if patience_left == 0:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        return history

    # ------------------------------------------------------------------
    def _eval_loss(self, seqs: list[Sequence], criterion: nn.Module) -> float:
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for inp, tgt, mask in _make_batches(
                    seqs, self.cfg.batch_size, self.cfg.pad_id, self.cfg.max_seq_len):
                inp, tgt, mask = inp.to(self.device), tgt.to(self.device), mask.to(self.device)
                logits = self.model(inp, key_padding_mask=mask)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), tgt.reshape(B * T))
                n_tok = (tgt != -100).sum().item()
                total_loss += loss.item() * n_tok
                total_tokens += n_tok
        return total_loss / max(total_tokens, 1)
