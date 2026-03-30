# src/data/__init__.py

# Purpose: Initialize the data package for the predictive log anomaly engine.
# This file imports all the relevant classes and functions from the data submodules, 
# making them available for import when the data package is imported. 
# It serves as the entry point for the data package and allows other modules to easily 
# access the data models and synthetic generation tools defined in the data submodules.

# Input: This file is currently empty, but it serves as the entry point for the data package. 
# It imports the LogEvent class, various anomaly patterns, the synthetic log generator, and the scenario builder.

# Output: This file allows the data package to be recognized as a Python package 
# and provides a convenient way to import the data models 
# and synthetic generation tools from the data submodules. 
# It makes it easier for other modules to access the 
# data-related functionality without needing to import each submodule individually.

# Used by: This file is used by any module that imports from the data package, 
# such as src.app.main.py or src.core.contracts.main.py, 
# to access the data models and synthetic generation tools defined in the data submodules.

"""src.data — Core data models for the pipeline."""
from .log_event import LogEvent

__all__ = [
    "LogEvent",
]
