# src/sequencing/models.py

# Purpose: Defines the core domain model for the sequencing stage, which is the Sequence data class.
# The Sequence class represents a single sequence of events, 
# including its ID, token IDs, timestamps, and an optional label.
# It also includes utility methods for getting the length of the sequence 
# and a string representation.

# Input: - sequence_id: a unique identifier for the sequence, typically encoding time order.
#        - tokens: a list of integer token IDs representing the events in the sequence.
#        - timestamps: an optional list of timestamps corresponding to each token.
#        - label: an optional integer label for the sequence.

# Output: - The Sequence class can be used to represent sequences of events in a structured way, 
#           which can then be processed by sequence builders 
#           and splitters for model training and evaluation.

# Used by: - The sequence builders to create Sequence objects from event data.
#          - The dataset splitter to split lists of Sequence objects into training, 
#            validation, and test sets.

"""Stage 3 — Sequencing: core domain model."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sequence:
    """A single labelled event sequence represented as token IDs."""
    sequence_id: str
    tokens: list[int]
    timestamps: list = field(default_factory=list)
    label: Optional[int] = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return (f"Sequence(id={self.sequence_id!r}, "
                f"len={len(self.tokens)}, label={self.label})")
