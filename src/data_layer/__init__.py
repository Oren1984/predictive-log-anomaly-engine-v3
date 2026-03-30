# src/data_layer/__init__.py

# Purpose: Initialize the data_layer package by importing key classes and functions.

# Input: None (this file serves as an initializer for the package).

# Output: - Exposes LogEvent and KaggleDatasetLoader for external use when importing data_layer.

# Used by: - Other modules in the project that need to work with log events and load datasets.

from .models import LogEvent
from .loader import KaggleDatasetLoader

__all__ = ["LogEvent", "KaggleDatasetLoader"]
