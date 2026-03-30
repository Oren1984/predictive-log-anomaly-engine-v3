# src/data_layer/loader.py

# Purpose: Define the KaggleDatasetLoader class for loading and normalizing the unified event dataset.

# Input: - Optional root path to the dataset (defaults to project root).
#        - Optional nrows to limit the number of rows loaded for testing.

# Output: - Methods to download (no-op), load raw DataFrame, normalize schema, and iterate over LogEvent objects.

# Used by: - Other modules in the project that need to load and work with the unified event dataset.

"""Stage 1 — Data Layer: dataset loader for existing artifacts."""
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .models import LogEvent

_ROOT = Path(__file__).resolve().parents[2]
_UNIFIED = _ROOT / "data/processed/events_unified.csv"


class KaggleDatasetLoader:
    """
    Loads the pre-existing unified event dataset.

    download() is a no-op: data is already present in data/raw/.
    load_raw()         → raw DataFrame from events_unified.csv.
    normalize_schema() → DataFrame with LogEvent-compatible columns.
    iter_events()      → lazy iterator of LogEvent objects.
    """

    def __init__(self, root: Optional[Path] = None, nrows: Optional[int] = None):
        self.root = root or _ROOT
        self.nrows = nrows
        self._unified = self.root / "data/processed/events_unified.csv"

    # ------------------------------------------------------------------
    def download(self) -> None:
        """No-op: raw files already present under data/raw/."""
        pass

    # ------------------------------------------------------------------
    def load_raw(self) -> pd.DataFrame:
        """Return the full unified CSV as a DataFrame (or first nrows)."""
        return pd.read_csv(
            self._unified,
            dtype={"label": "int8"},
            nrows=self.nrows,
        )

    # ------------------------------------------------------------------
    def normalize_schema(self) -> pd.DataFrame:
        """
        Return a DataFrame with LogEvent-compatible columns:
            timestamp, service, level, message, session_id, label
        'service' maps from the 'dataset' column.
        'level' is empty (not present in raw data).
        """
        df = self.load_raw()
        return pd.DataFrame({
            "timestamp":  df["timestamp"],
            "service":    df["dataset"],
            "level":      "",
            "message":    df["message"],
            "session_id": df["session_id"],
            "label":      df["label"],
        })

    # ------------------------------------------------------------------
    def iter_events(self) -> Iterator[LogEvent]:
        """Lazily yield LogEvent objects (chunked to avoid OOM)."""
        chunk_size = 100_000
        reader = pd.read_csv(
            self._unified,
            dtype={"label": "int8"},
            chunksize=chunk_size,
            nrows=self.nrows,
        )
        for chunk in reader:
            for row in chunk.itertuples(index=False):
                yield LogEvent(
                    timestamp=row.timestamp if pd.notna(row.timestamp) else None,
                    service=row.dataset,
                    level="",
                    message=row.message,
                    meta={"session_id": row.session_id},
                    label=int(row.label),
                )
