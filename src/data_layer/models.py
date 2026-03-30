# src/data_layer/models.py

# Purpose: Define the LogEvent dataclass as the core domain model 
# for representing normalized log events.

# Input: - Fields include timestamp, service, level, message, 
# meta (optional dict), and label (optional int).

# Output: - A structured representation of a log event that can be used 
# throughout the project for analysis and modeling.

# Used by: - Other modules in the project that need to work with log events, 
# such as the dataset loader and any analysis/modeling components.

"""Stage 1 — Data Layer: core domain models."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LogEvent:
    """Normalised representation of a single log line."""
    timestamp: Optional[float]
    service: str          # maps to 'dataset' column (hdfs / bgl)
    level: str            # empty string if not available in source
    message: str
    meta: dict = field(default_factory=dict)
    label: Optional[int] = None
