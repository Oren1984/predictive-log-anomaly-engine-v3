# src/data/log_event.py

# Purpose: Define the LogEvent dataclass, 
# which represents a single log entry in the synthetic log generation pipeline.
# This class includes fields for the timestamp, service name, log level, 
# message, metadata, and anomaly label. It also provides helper methods 
# for serializing to and from dictionaries, which is useful

# Input: The LogEvent class takes the following fields as input:
# - timestamp: A Unix timestamp (float) or a datetime object representing when the log event occurred.
# - service: A string representing the service or dataset name (e.g., "auth", "api", "billing", "db").
# - level: A string representing the log level (e.g., "INFO", "WARNING", "ERROR").
# - message: A string containing the raw log message text.
# - meta: An optional dictionary containing arbitrary key/value metadata (e.g., host, component, phase).
# - label: An optional integer representing the anomaly label (

# Output: The LogEvent class provides a structured representation of a log entry,
# along with methods to convert to and from a dictionary format that is suitable for JSON serialization.
# The to_dict() method converts a LogEvent instance into a dictionary, 
# while the from_dict() class method reconstructs a LogEvent instance from a dictionary.
# This allows for easy storage and retrieval of log events in formats like parquet, 
# which may not support complex nested types directly.

# Used by: The LogEvent class is used by various components of the synthetic log generation pipeline,
# including the synthetic log generator and scenario builder, to create and manipulate log events.

"""
src.data.log_event — Core LogEvent dataclass with IO helpers.

This module provides the canonical LogEvent type used by the synthetic pipeline.
It is compatible with src.data_layer.models.LogEvent (same fields) and adds
parquet-friendly to_dict() / from_dict() serialisation helpers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Union


@dataclass
class LogEvent:
    """
    Normalised representation of a single log line.

    Fields
    ------
    timestamp : Unix timestamp (float) or datetime.  Stored as float internally.
    service   : Service / dataset name (e.g. "auth", "api", "billing", "db").
    level     : Log level string ("INFO", "WARNING", "ERROR", …).
    message   : Raw log message text.
    meta      : Arbitrary key/value metadata dict (host, component, phase, …).
    label     : Anomaly label.  0 = normal, 1 = anomalous.  None = unknown.
    """

    timestamp: Union[float, datetime]
    service:   str
    level:     str
    message:   str
    meta:      dict = field(default_factory=dict)
    label:     Optional[int] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dict suitable for writing to parquet.

        The ``meta`` field is JSON-encoded to a string so it survives
        round-trips through parquet without requiring nested-type support.
        """
        ts = self.timestamp
        if isinstance(ts, datetime):
            ts = ts.timestamp()
        return {
            "timestamp": float(ts) if ts is not None else None,
            "service":   self.service,
            "level":     self.level,
            "message":   self.message,
            "meta":      json.dumps(self.meta or {}),
            "label":     int(self.label) if self.label is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogEvent":
        """
        Reconstruct a LogEvent from a dict produced by to_dict().

        ``meta`` may be a JSON string (from parquet) or already a dict.
        """
        meta_raw = d.get("meta", "{}")
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        else:
            meta = {}

        lbl = d.get("label")
        return cls(
            timestamp=float(d["timestamp"]) if d.get("timestamp") is not None else None,
            service=str(d.get("service", "")),
            level=str(d.get("level", "")),
            message=str(d.get("message", "")),
            meta=meta,
            label=int(lbl) if lbl is not None else None,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def timestamp_as_datetime(self) -> Optional[datetime]:
        """Return the timestamp as a UTC-aware datetime, or None."""
        if self.timestamp is None:
            return None
        ts = self.timestamp
        if isinstance(ts, datetime):
            return ts.astimezone(timezone.utc)
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
