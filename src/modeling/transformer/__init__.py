# src/modeling/transformer/__init__.py

# Purpose: Initialize the transformer modeling package by importing key classes and functions.

# Input: This file imports the following components from the transformer submodules:
# - TransformerConfig: A dataclass for storing hyperparameters and configuration settings for the transformer model
# - NextTokenTransformerModel: The main transformer model class that implements the architecture and forward pass
# - Trainer: A class responsible for training the transformer model on a dataset of sequences
# - AnomalyScorer: A class for scoring sequences based on the trained transformer model, 
#   useful for anomaly detection tasks

# Output: By importing these components, this __init__.py file allows users to easily access
# the transformer modeling functionality by importing from the transformer package. For example, users can do:
# from modeling.transformer import TransformerConfig, NextTokenTransformerModel, Trainer, AnomalyScorer

# Used by: This file is used by any code that needs to utilize the transformer modeling capabilities, 
# such as training a model or scoring sequences for anomaly detection. 
# It serves as the entry point for the transformer subpackage, 
# making it easier to manage imports and maintain a clean codebase.

from .config import TransformerConfig
from .model import NextTokenTransformerModel
from .trainer import Trainer
from .scorer import AnomalyScorer

__all__ = [
    "TransformerConfig",
    "NextTokenTransformerModel",
    "Trainer",
    "AnomalyScorer",
]
