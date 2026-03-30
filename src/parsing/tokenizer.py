# src/parsing/tokenizer.py

# Purpose: Define the EventTokenizer class, which encodes template IDs
# into compact integer token IDs and provides methods for encoding and decoding.

# Input: - EventTokenizer: A class that can load template mappings 
# from CSV files and encode template IDs into token IDs, 
# as well as decode token IDs back to template text.

# Output: - EventTokenizer: Provides methods to transform template IDs 
# into token IDs for use in modeling, 
# and to decode token IDs back into human-readable template text.

# Used by: Other stages of the pipeline that need to convert template IDs into token 
# IDs for input into machine learning models, 
# and to interpret model outputs back into template text.

"""Stage 2 — Parsing: EventTokenizer using template vocabulary."""
from pathlib import Path
from typing import Optional

import pandas as pd


class EventTokenizer:
    """
    Encodes template_ids → compact integer token ids and back.

    Special tokens:
        PAD = 0   (also used as BOS in the transformer)
        UNK = 1
    Template IDs from templates.csv are shifted: token_id = template_id + 1
    (to leave 0 for PAD, 1 for UNK).

    Usage:
        tok = EventTokenizer()
        tok.load_from_csv("data/intermediate/templates.csv")
        tokens = tok.encode([6675, 6684, 6692])  # template_ids
        text   = tok.decode(tokens)              # back to template strs
    """

    PAD_ID: int = 0
    UNK_ID: int = 1
    _OFFSET: int = 2   # template_id -> token_id = template_id + OFFSET

    def __init__(self):
        self._tid_to_text: dict[int, str] = {}
        self._text_to_tid: dict[str, int] = {}
        self._sorted_tids: list[int] = []

    # ------------------------------------------------------------------
    def load_from_csv(self, templates_csv_path: Path | str) -> "EventTokenizer":
        df = pd.read_csv(templates_csv_path)
        self._tid_to_text = dict(zip(df["template_id"].astype(int),
                                     df["template_text"]))
        self._text_to_tid = {v: k for k, v in self._tid_to_text.items()}
        self._sorted_tids = sorted(self._tid_to_text.keys())
        return self

    # ------------------------------------------------------------------
    def encode(self, template_ids: list[int]) -> list[int]:
        """Map a list of template_ids to token_ids (int list)."""
        return [
            tid + self._OFFSET if tid in self._tid_to_text else self.UNK_ID
            for tid in template_ids
        ]

    def decode(self, token_ids: list[int]) -> list[str]:
        """Map token_ids back to template text strings."""
        result = []
        for tok in token_ids:
            if tok == self.PAD_ID:
                result.append("<PAD>")
            elif tok == self.UNK_ID:
                result.append("<UNK>")
            else:
                tid = tok - self._OFFSET
                result.append(self._tid_to_text.get(tid, "<UNK>"))
        return result

    def template_id_to_token(self, tid: int) -> int:
        return tid + self._OFFSET if tid in self._tid_to_text else self.UNK_ID

    def token_to_template_id(self, tok: int) -> Optional[int]:
        if tok < self._OFFSET:
            return None
        tid = tok - self._OFFSET
        return tid if tid in self._tid_to_text else None

    @property
    def vocab_size(self) -> int:
        """Total token vocabulary size (PAD + UNK + all templates)."""
        return len(self._tid_to_text) + self._OFFSET

    def to_vocab_dict(self) -> dict:
        """Serialisable {token_id: template_text} mapping."""
        vocab = {str(self.PAD_ID): "<PAD>", str(self.UNK_ID): "<UNK>"}
        for tid, text in self._tid_to_text.items():
            vocab[str(tid + self._OFFSET)] = text
        return vocab
