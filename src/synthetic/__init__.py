# src/synthetic/__init__.py

# Purpose: Defines the public API for the synthetic data generation package, 
# which includes the core classes and patterns for generating synthetic log data.
# It imports the SyntheticLogGenerator class, various FailurePattern subclasses, 
# and the ScenarioBuilder class, making them available for use in other parts of the application.

# Input: - FailurePattern: the base class for defining failure patterns in synthetic log data.
#        - MemoryLeakPattern, DiskFullPattern, AuthBruteForcePattern, NetworkFlapPattern: 
#          concrete implementations of FailurePattern representing specific failure scenarios.
#        - SyntheticLogGenerator: a class for generating synthetic log events based on defined failure patterns and scenarios.
#        - ScenarioBuilder: a class for building scenario definitions that specify which failure patterns to use and how to configure them.

# Output: - The __init__.py file exposes the key classes and patterns for synthetic log generation, 
#           allowing other parts of the application to import and use them for generating synthetic data.   

# Used by: - The main application to generate synthetic log data for training and evaluating models, 
#            as well as for testing and demonstration purposes.

"""Stage 1 — Synthetic data generation package."""
from .generator import SyntheticLogGenerator
from .patterns import (
    AuthBruteForcePattern,
    DiskFullPattern,
    FailurePattern,
    MemoryLeakPattern,
    NetworkFlapPattern,
)
from .scenario_builder import ScenarioBuilder

__all__ = [
    "FailurePattern",
    "MemoryLeakPattern",
    "DiskFullPattern",
    "AuthBruteForcePattern",
    "NetworkFlapPattern",
    "SyntheticLogGenerator",
    "ScenarioBuilder",
]
