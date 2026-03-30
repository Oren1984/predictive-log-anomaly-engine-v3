# src/health/__init__.py

# Purpose: Defines the health checking system for the pipeline, 
# including critical and optional checks to determine overall health status.

# Input: None (this is an __init__. file that imports the HealthChecker class) 

# Output: The HealthChecker class is made available for import when the health package is imported.

# Used by: Other parts of the pipeline that need to perform health checks, 
# such as API endpoints or monitoring tools.

"""Stage 7 — Health package."""
from .checks import HealthChecker

__all__ = ["HealthChecker"]
