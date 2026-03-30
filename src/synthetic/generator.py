# src/synthetic/generator.py

# Purpose: Defines the SyntheticLogGenerator class, which generates synthetic 
# log events based on defined failure patterns and scenarios.
# The generator takes a list of FailurePattern instances and a random seed, 
# and produces a list of LogEvent objects by running the patterns through scenario definitions.
# It also includes a method for converting the generated LogEvent objects 
# into a pandas DataFrame format that matches the expected schema for downstream processing.

# Input: - patterns: a list of FailurePattern instances that define how to emit log events for different failure scenarios.
#        - seed: a master random seed for reproducibility, which is used to derive 
#                per-scenario seeds for independent reproducibility of scenarios.
#        - n_events: the number of log events to generate for a given scenario.

# Output: - The generate method returns a list of LogEvent objects in chronological order, 
#           which can then be converted to a DataFrame using the events_to_dataframe method for further processing,
#           such as building sequences and splitting datasets for model training and evaluation.

# Used by: - The main application to generate synthetic log data for training and evaluating models,
#            as well as for testing and demonstration purposes.

"""Stage 1 — Synthetic: log event generator."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_layer.models import LogEvent
from .patterns import FailurePattern


class SyntheticLogGenerator:
    """
    Generate lists of LogEvent objects by running FailurePattern instances
    through scenario definitions produced by ScenarioBuilder.

    Parameters
    ----------
    patterns : registered FailurePattern instances (keyed by .name)
    seed     : master random seed; each scenario derives its own sub-seed
               so scenarios are independently reproducible
    """

    def __init__(self, patterns: list[FailurePattern], seed: int = 42) -> None:
        if not patterns:
            raise ValueError("At least one FailurePattern must be registered.")
        self.patterns = patterns
        self.seed = seed
        self._pattern_map: dict[str, FailurePattern] = {
            p.name: p for p in patterns
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate(self, n_events: int, scenario: dict) -> list[LogEvent]:
        """
        Generate *n_events* LogEvent objects for the given scenario.

        Parameters
        ----------
        n_events : number of events to emit
        scenario : scenario dict (from ScenarioBuilder.build_scenario)

        Returns
        -------
        list of LogEvent in chronological order (index 0 = earliest)
        """
        if n_events < 1:
            raise ValueError("n_events must be >= 1")

        # Derive a deterministic per-scenario RNG
        scenario_seed = self._scenario_seed(scenario.get("scenario_id", "default"))
        rng = random.Random(self.seed ^ scenario_seed)

        # Resolve pattern list (hybrid uses multiple)
        names = scenario.get("pattern_names") or [scenario.get("pattern_name", "")]
        active = [self._find_pattern(n) for n in names if n]
        if not active:
            raise ValueError(
                "No valid patterns found. scenario must include "
                "'pattern_name' or 'pattern_names'."
            )

        ctx = {
            **scenario,
            "n_events": n_events,
            "rng":      rng,
        }

        events: list[LogEvent] = []
        for t in range(n_events):
            # Round-robin across patterns for hybrid; single element for single-pattern
            pattern = active[t % len(active)]
            event   = pattern.emit_event(t, ctx)
            events.append(event)

        return events

    def generate_all(self, scenarios: list[dict]) -> list[LogEvent]:
        """
        Generate events for every scenario in *scenarios* and concatenate.

        Each scenario dict must include 'n_events' (or use its stored value).
        """
        all_events: list[LogEvent] = []
        for sc in scenarios:
            n = sc.get("n_events", 1000)
            all_events.extend(self.generate(n, sc))
        return all_events

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    @staticmethod
    def events_to_dataframe(events: list[LogEvent]) -> pd.DataFrame:
        """
        Convert a list of LogEvent objects to a pandas DataFrame.

        Columns:
            timestamp, service, level, message, session_id, label,
            host, component, scenario_id, phase

        This matches the normalize_schema() output of KaggleDatasetLoader
        (with additional synthetic-specific columns appended).
        """
        rows = []
        for ev in events:
            meta = ev.meta or {}
            rows.append({
                "timestamp":   ev.timestamp,
                "service":     ev.service,
                "level":       ev.level,
                "message":     ev.message,
                "session_id":  meta.get("session_id", ""),
                "label":       int(ev.label) if ev.label is not None else 0,
                # Synthetic-specific columns
                "host":        meta.get("host",        ""),
                "component":   meta.get("component",   ""),
                "scenario_id": meta.get("scenario_id", ""),
                "phase":       meta.get("phase",       ""),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_pattern(self, name: str) -> FailurePattern:
        if name not in self._pattern_map:
            available = sorted(self._pattern_map.keys())
            raise ValueError(
                f"Pattern {name!r} not registered. "
                f"Available: {available}"
            )
        return self._pattern_map[name]

    @staticmethod
    def _scenario_seed(scenario_id: str) -> int:
        """Derive a deterministic integer seed from a scenario_id string."""
        h = 0
        for ch in scenario_id:
            h = (h * 31 + ord(ch)) & 0xFFFF_FFFF
        return h
