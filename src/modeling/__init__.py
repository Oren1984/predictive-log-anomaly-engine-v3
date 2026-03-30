# src/modeling/__init__.py

# Purpose: This file serves as the initializer for the 'modeling' package. 
# It imports the necessary classes and functions from the submodules (baseline and transformer) 
# and makes them available for use when the package is imported. 
# This allows users to access these components directly from the 'modeling' 
# namespace without needing to import each submodule separately.

# Input: None (this file is meant to be imported by other modules, not executed directly)

# Output: When this package is imported, it will provide access to the following classes and functions:
# - BaselineFeatureExtractor
# - BaselineAnomalyModel
# - ThresholdCalibrator
# - TransformerConfig
# - NextTokenTransformerModel
# - Trainer
# - AnomalyScorer

# Used by: Other modules in the project that require access to the modeling components, such as training scripts, 
# evaluation scripts, or any module that needs to utilize the baseline or transformer models for anomaly detection.

from .baseline import BaselineFeatureExtractor, BaselineAnomalyModel, ThresholdCalibrator
from .transformer import TransformerConfig, NextTokenTransformerModel, Trainer, AnomalyScorer

__all__ = [
    "BaselineFeatureExtractor", "BaselineAnomalyModel", "ThresholdCalibrator",
    "TransformerConfig", "NextTokenTransformerModel", "Trainer", "AnomalyScorer",
]
