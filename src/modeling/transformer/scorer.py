# src/modeling/transformer/scorer.py

# Purpose: Define the AnomalyScorer class, which uses a trained NextTokenTransformerModel 
# to score sequences based on their negative log-likelihood (NLL).

# Input: The AnomalyScorer takes a list of Sequence objects as input to the score method. 
# Each Sequence contains a list of token IDs. 
# The scorer computes the NLL for each token in the sequence based on the predictions from the transformer model.

# Output: A numpy array of anomaly scores, one per sequence. Higher scores indicate more anomalous sequences.

# Used by: The AnomalyScorer class is used after training a NextTokenTransformerModel
# to evaluate new sequences for anomaly detection. 
# It can be used in a pipeline where sequences are scored and then classified 
# as anomalous or normal based on a threshold. 
# The class also provides methods to set and save the threshold for classification,

"""Stage 4B — Transformer: NLL-based anomaly scorer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...sequencing.models import Sequence
from .config import TransformerConfig
from .model import NextTokenTransformerModel


class AnomalyScorer:
    """
    Scores sequences using mean / max per-token negative log-likelihood.

    Higher score = more anomalous (log-prob farther from training distribution).

    Parameters
    ----------
    model    : trained NextTokenTransformerModel
    cfg      : TransformerConfig (supplies pad_id, score_reduction, max_seq_len)
    device   : torch device string (default "cpu")
    """

    def __init__(
        self,
        model: NextTokenTransformerModel,
        cfg: TransformerConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.cfg = cfg
        self.device = torch.device(device)
        self.threshold_: Optional[float] = None

    # ------------------------------------------------------------------
    def score(self, sequences: list[Sequence]) -> np.ndarray:
        """
        Return a float32 array of anomaly scores, one per sequence.
        """
        scores = []
        bs = self.cfg.batch_size
        pad = self.cfg.pad_id
        max_len = self.cfg.max_seq_len

        for start in range(0, len(sequences), bs):
            batch = sequences[start: start + bs]
            # build padded input/target tensors
            raw = [s.tokens[: max_len + 1] for s in batch]
            max_t = max(len(r) for r in raw)
            if max_t < 2:
                scores.extend([0.0] * len(batch))
                continue

            inp_list, tgt_list, mask_list, lengths = [], [], [], []
            for r in raw:
                inp = r[:-1]
                tgt = r[1:]
                pad_len = (max_t - 1) - len(inp)
                inp_list.append(inp + [pad] * pad_len)
                tgt_list.append(tgt + [pad] * pad_len)
                mask_list.append([False] * len(inp) + [True] * pad_len)
                lengths.append(len(inp))

            inp_t  = torch.tensor(inp_list, dtype=torch.long, device=self.device)
            tgt_t  = torch.tensor(tgt_list, dtype=torch.long, device=self.device)
            mask_t = torch.tensor(mask_list, dtype=torch.bool,  device=self.device)

            with torch.no_grad():
                logits = self.model(inp_t, key_padding_mask=mask_t)

            log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
            # gather the log-prob of the actual next token
            tgt_lp = log_probs.gather(2, tgt_t.unsqueeze(-1)).squeeze(-1)  # (B, T)
            nll = -tgt_lp   # negative log-likelihood

            for i, (nll_row, length) in enumerate(zip(nll, lengths)):
                if length == 0:
                    scores.append(0.0)
                    continue
                valid = nll_row[:length]
                if self.cfg.score_reduction == "max":
                    s = valid.max().item()
                else:
                    s = valid.mean().item()
                scores.append(s)

        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------
    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = threshold

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call set_threshold() before predict().")
        return (scores >= self.threshold_).astype(np.int8)

    # ------------------------------------------------------------------
    def save_threshold(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({"threshold": self.threshold_}, indent=2))

    @classmethod
    def load_threshold(cls, path: Path | str) -> float:
        return json.loads(Path(path).read_text())["threshold"]
