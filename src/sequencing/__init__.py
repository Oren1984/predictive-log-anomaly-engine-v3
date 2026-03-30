# src/sequencing/__init__.py

# Purpose: Defines the public API for the sequencing module, 
# which includes classes for building sequences from event data and splitting datasets. 
# It imports key classes from submodules and exposes them in the package namespace.

# Input: - Sequence: a data class representing a sequence of tokens with optional timestamps and labels.
#        - SequenceBuilder: an abstract base class defining the interface for building sequences.
#        - SlidingWindowSequenceBuilder: a concrete implementation of 
#          SequenceBuilder that creates fixed-length overlapping sequences.
#        - SessionSequenceBuilder: a concrete implementation of 
#          SequenceBuilder that creates sequences based on session boundaries.
#        - DatasetSplitter: a utility class for splitting datasets into training, 
#          validation, and test sets.

# Output: - The imported classes are made available for external use when the sequencing package is imported.


# Used by: - The main application to build sequences from event data and 
#            split datasets for model training and evaluation.

from .models import Sequence
from .builders import SequenceBuilder, SlidingWindowSequenceBuilder, SessionSequenceBuilder
from .splitter import DatasetSplitter

__all__ = [
    "Sequence",
    "SequenceBuilder",
    "SlidingWindowSequenceBuilder",
    "SessionSequenceBuilder",
    "DatasetSplitter",
]
