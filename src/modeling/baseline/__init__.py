# src/modeling/baseline/__init__.py

# Purpose: This file serves as the initializer for the 'baseline' subpackage within the 'modeling' package.
# It imports the necessary classes from the submodules (extractor, model, calibrator) 
# and makes them available for use when the 'baseline' package is imported. 
# This allows users to access these components directly from the 'modeling.baseline' 
# namespace without needing to import each submodule separately.

# Input: None (this file is meant to be imported by other modules, not executed directly)

# Output: When this package is imported, it will provide access to the following classes:
# - BaselineFeatureExtractor
# - BaselineAnomalyModel
# - ThresholdCalibrator

# Used by: Other modules in the project that require access to the baseline components, such as training scripts,
# evaluation scripts, or any module that needs to utilize the baseline feature extractor, 
# anomaly model, or threshold calibrator for anomaly detection.

from .extractor import BaselineFeatureExtractor
from .model import BaselineAnomalyModel
from .calibrator import ThresholdCalibrator

__all__ = [
    "BaselineFeatureExtractor",
    "BaselineAnomalyModel",
    "ThresholdCalibrator",
]
