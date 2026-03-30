# src/observability/__init__.py

# Purpose: Initialize the observability package by importing key classes and functions related to metrics and logging.

# Input: This file imports the following components from the observability submodules:
# - MetricsRegistry: A class for registering and managing custom metrics for monitoring model performance and behavior
# - MetricsMiddleware: A middleware class for integrating metrics collection into the training and evaluation loops
# - configure_logging: A function for setting up logging configuration, such as log levels and formats

# Output: By importing these components, this __init__.py file allows users 
# to easily access the observability functionality by importing from the observability package. 
# For example, users can do: 

# Used by:  This file is used by any code that needs to utilize the observability capabilities, 
# such as during training or evaluation of models. 
# It serves as the entry point for the observability subpackage, 
# making it easier to manage imports and maintain a clean codebase.
# By centralizing the imports, users can access all observability features from a single location.

"""Stage 7 — Observability package."""
from .metrics import MetricsRegistry, MetricsMiddleware
from .logging import configure_logging

__all__ = ["MetricsRegistry", "MetricsMiddleware", "configure_logging"]
