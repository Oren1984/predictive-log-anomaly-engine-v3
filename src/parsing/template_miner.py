# src/parsing/template_miner.py

# Purpose: Define the TemplateMiner class, which wraps pre-computed 
# template mining outputs and provides methods to load from CSV artifacts or fit from scratch.

# Input: - TemplateMiner: A class that can load template mappings 
# from CSV files or fit a simple template mining algorithm on raw log messages.

# Output: - TemplateMiner: Provides methods to transform log messages 
# into template IDs based on pre-computed mappings or fitted templates.

# Used by: Other stages of the pipeline that need to convert raw log messages 
# into template IDs for further processing, such as tokenization and modeling.

"""Stage 2 — Parsing: template miner wrapper over existing CSV artifacts."""
from pathlib import Path
from typing import Optional

import pandas as pd


class TemplateMiner:
    """
    Wraps the pre-computed template mining outputs.

    Preferred usage (existing artifacts):
        miner = TemplateMiner()
        miner.load_from_csv("data/intermediate/templates.csv")
        ids = miner.transform_from_existing("data/intermediate/events_with_templates.csv")

    Fallback usage (fit from scratch, slow):
        miner.fit(events_df)
        ids = miner.transform(events_df)
    """

    def __init__(self):
        self._template_to_id: dict[str, int] = {}
        self._id_to_template: dict[int, str] = {}
        self._templates_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Load from existing CSV artifacts
    # ------------------------------------------------------------------
    def load_from_csv(self, templates_csv_path: Path | str) -> "TemplateMiner":
        """Load template_id <-> template_text mapping from templates.csv."""
        df = pd.read_csv(templates_csv_path)
        self._templates_df = df
        self._template_to_id = dict(zip(df["template_text"], df["template_id"]))
        self._id_to_template = dict(zip(df["template_id"], df["template_text"]))
        return self

    def transform_from_existing(
        self,
        events_with_templates_csv: Path | str,
        usecols: list[str] = None,
    ) -> pd.Series:
        """
        Return a Series of template_ids from the pre-computed
        events_with_templates.csv (does not re-run template mining).
        """
        cols = usecols or ["session_id", "dataset", "label",
                           "template_id", "template_text"]
        df = pd.read_csv(events_with_templates_csv, usecols=cols)
        return df["template_id"]

    # ------------------------------------------------------------------
    # Fit/transform from scratch (minimal Drain-lite)
    # ------------------------------------------------------------------
    _SUBS = [
        (r"blk_-?\d+",                                    "<BLK>"),
        (r"\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+", "<TS>"),
        (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?", "<IP>"),
        (r"\d{4}\.\d{2}\.\d{2}",                          "<DATE>"),
        (r"R\d+(?:-[A-Z\d]+)+(?::[A-Z]\d+-[A-Z]\d+)?",   "<NODE>"),
        (r"/[a-zA-Z0-9_./-]+",                             "<PATH>"),
        (r"\b[0-9a-f]{8,}\b",                             "<HEX>"),
        (r"\b\d+\b",                                       "<NUM>"),
        (r"\s+",                                           " "),
    ]

    def _generalize(self, series: pd.Series) -> pd.Series:
        import re
        result = series.fillna("").astype(str)
        for pat, repl in self._SUBS:
            result = result.str.replace(pat, repl, regex=True)
        return result.str.strip()

    def fit(self, events: pd.DataFrame) -> "TemplateMiner":
        """Derive templates from a DataFrame with a 'message' column."""
        templates = sorted(self._generalize(events["message"]).unique())
        self._template_to_id = {t: i + 1 for i, t in enumerate(templates)}
        self._id_to_template = {v: k for k, v in self._template_to_id.items()}
        return self

    def transform(self, events: pd.DataFrame) -> pd.Series:
        """Map messages to template_ids using fitted mapping."""
        generalized = self._generalize(events["message"])
        return generalized.map(self._template_to_id).fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._template_to_id)

    def get_template(self, tid: int) -> str:
        return self._id_to_template.get(tid, "<UNK>")
