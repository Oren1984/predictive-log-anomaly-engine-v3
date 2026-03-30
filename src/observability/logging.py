# src/observability/logging.py

# Purpose: Define a helper function to configure logging for the API service,
# ensuring that all log messages have a consistent format and are output to stdout.

# Input: The configure_logging function takes an optional log level as input (default is "INFO").

# Output: The function configures the root logger to use a specific format for log messages,
# which includes the timestamp, log level, logger name, and message.

# Used by: This function is used by the API service and any other components that require logging.
# By calling configure_logging at the start of the application, 
# all log messages will be formatted consistently and will be output to stdout, 
# making it easier to monitor and debug the application.

"""Stage 7 — Observability: structured logging helpers."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a consistent format for the API service."""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        root.addHandler(handler)
