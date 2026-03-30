# src/sequencing/builders.py

# Purpose: Defines concrete implementations of the SequenceBuilder interface for building sequences from event data.
# It includes a SlidingWindowSequenceBuilder for creating fixed-length overlapping sequences and 
# a SessionSequenceBuilder for creating sequences based on session boundaries.

# Input: - SequenceBuilder: an abstract base class that defines the interface for building sequences.
#        - SlidingWindowSequenceBuilder: a class that implements SequenceBuilder to create sequences by sliding a fixed-size window over event data.
#        - SessionSequenceBuilder: a class that implements SequenceBuilder to create sequences based on pre-computed session data, 
#          parsing comma-separated template ID sequences.

# Output: - The concrete implementations of SequenceBuilder can be used to build sequences from event data, 
#           which can then be used for model training and evaluation.

# Used by: - The main application to build sequences from event data, which are then split into training, 
#            validation, and test sets for model training and evaluation.

"""Stage 3 — Sequencing: sequence builder implementations."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .models import Sequence


class SequenceBuilder(ABC):
    """Interface: build a list of Sequence objects from events."""

    @abstractmethod
    def build(self, events: pd.DataFrame) -> list[Sequence]:
        ...


# ---------------------------------------------------------------------------
class SlidingWindowSequenceBuilder(SequenceBuilder):
    """
    Builds fixed-length overlapping sequences by sliding over event rows.

    Suitable for datasets without natural session boundaries (e.g. BGL).
    Requires a 'token_id' or 'template_id' column in events.
    """

    def __init__(self, window: int = 50, stride: int = 10,
                 token_col: str = "template_id",
                 timestamp_col: Optional[str] = "timestamp",
                 label_col: Optional[str] = "label"):
        self.window = window
        self.stride = stride
        self.token_col = token_col
        self.timestamp_col = timestamp_col
        self.label_col = label_col

    def build(self, events: pd.DataFrame) -> list[Sequence]:
        tokens = events[self.token_col].tolist()
        labels = events[self.label_col].tolist() if self.label_col in events else None
        timestamps = (events[self.timestamp_col].tolist()
                      if self.timestamp_col in events else [])

        sequences = []
        for start in range(0, max(1, len(tokens) - self.window + 1), self.stride):
            end = start + self.window
            window_tokens = tokens[start:end]
            if not window_tokens:
                break
            window_ts = timestamps[start:end] if timestamps else []
            window_label = (max(labels[start:end]) if labels else None)
            sequences.append(Sequence(
                sequence_id=f"window_{start}",
                tokens=[int(t) for t in window_tokens],
                timestamps=window_ts,
                label=window_label,
            ))
        return sequences

    def iter_build(self, events: pd.DataFrame) -> Iterator[Sequence]:
        """Memory-efficient lazy version."""
        yield from self.build(events)


# ---------------------------------------------------------------------------
class SessionSequenceBuilder(SequenceBuilder):
    """
    Builds Sequence objects from pre-computed session_sequences_v2.csv.

    Parses the 'ordered_template_sequence' column (comma-separated template_ids).
    Can optionally apply EventTokenizer to map template_ids -> token_ids.
    """

    def __init__(self, tokenizer=None,
                 seq_col: str = "ordered_template_sequence",
                 id_col: str = "session_id",
                 label_col: str = "label",
                 nrows: Optional[int] = None):
        self.tokenizer = tokenizer
        self.seq_col = seq_col
        self.id_col = id_col
        self.label_col = label_col
        self.nrows = nrows

    def load_csv(self, path: Path | str) -> list[Sequence]:
        """Load sequences directly from session_sequences_v2.csv."""
        df = pd.read_csv(path, nrows=self.nrows)
        return self.build(df)

    def build(self, events: pd.DataFrame) -> list[Sequence]:
        """
        Build sequences from a DataFrame that contains
        'ordered_template_sequence' (comma-separated int string).
        """
        sequences = []
        for row in events.itertuples(index=False):
            raw_seq = getattr(row, self.seq_col, "")
            if not isinstance(raw_seq, str) or not raw_seq.strip():
                tids = []
            else:
                tids = [int(x) for x in raw_seq.split(",") if x.strip()]

            tokens = (self.tokenizer.encode(tids)
                      if self.tokenizer else tids)
            label = (int(getattr(row, self.label_col))
                     if hasattr(row, self.label_col) else None)
            sequences.append(Sequence(
                sequence_id=str(getattr(row, self.id_col, "")),
                tokens=tokens,
                timestamps=[],
                label=label,
            ))
        return sequences
