# src/synthetic/scenario_builder.py

# Purpose: Defines the ScenarioBuilder class, which provides methods for constructing 
# scenario definition dictionaries that specify how to generate synthetic log events using the SyntheticLogGenerator.
# The ScenarioBuilder allows users to define scenarios with specific services, 
# hosts, event counts, phase proportions, and associated failure patterns. 
# It includes validation to ensure that the defined scenarios are well-formed 
# and can be used directly by the SyntheticLogGenerator to produce synthetic log data.

# Input: - scenario_id: a unique string identifier for the scenario.
#        - service: the name of the service affected by the scenario (e.g., "app-server").
#        - host: the identifier for the host involved in the scenario (e.g., "host-01").
#        - start_ts: the starting Unix timestamp for the generated events.

# Output: - The build_scenario method returns a dictionary containing all the necessary 
#           information to generate synthetic log events for the defined scenario, 
#           including phase boundaries and anomaly rates.
#         - The build_hybrid_scenario method provides a convenient way to create 
#           scenarios that involve multiple failure patterns, 
#           distributing events across the specified patterns in a round-robin fashion.

# Used by: - The main application to define scenarios for synthetic log generation,
#            which are then passed to the SyntheticLogGenerator to 
#            produce synthetic log data for training and evaluating models, 
#            as well as for testing and demonstration purposes.

"""Stage 1 — Synthetic: scenario definition builder."""
from __future__ import annotations

from typing import Optional


class ScenarioBuilder:
    """
    Build scenario definition dicts used by SyntheticLogGenerator.

    A scenario encodes which service/host is affected, how many events
    to generate, which phase proportions to use, and which pattern(s) to run.
    """

    # Default phase split for a realistic "gradual failure" scenario
    DEFAULT_PHASES = {
        "normal":      0.60,
        "degradation": 0.30,
        "failure":     0.10,
    }

    # ------------------------------------------------------------------
    def build_scenario(
        self,
        scenario_id:  str,
        service:      str,
        host:         str,
        start_ts:     float,
        n_events:     int,
        phases:       Optional[dict] = None,
        pattern_name: Optional[str] = None,
        pattern_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Return a scenario context dict ready for SyntheticLogGenerator.generate().

        Parameters
        ----------
        scenario_id   : unique string identifier
        service       : service name ("app-server", "storage", …)
        host          : host identifier ("host-01", …)
        start_ts      : starting Unix timestamp for generated events
        n_events      : total events to generate for this scenario
        phases        : proportion dict (keys: normal, degradation, failure).
                        Defaults to {normal:0.60, degradation:0.30, failure:0.10}.
                        Values must sum to 1.0 (±0.02 tolerance).
        pattern_name  : single pattern name (mutually exclusive with pattern_names)
        pattern_names : list of pattern names for a hybrid scenario
                        (patterns are cycled round-robin across events)

        Returns
        -------
        dict with keys: scenario_id, service, host, start_ts, n_events,
                        phases, phase_boundaries, anomaly_rate,
                        pattern_name / pattern_names
        """
        if phases is None:
            phases = dict(self.DEFAULT_PHASES)

        # Validate phases
        total = sum(phases.values())
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"phases must sum to 1.0 (got {total:.4f}). "
                f"Received: {phases}"
            )

        if n_events < 1:
            raise ValueError("n_events must be >= 1")

        # Compute exact event counts per phase
        n_normal = int(n_events * phases.get("normal", 0.60))
        n_deg    = int(n_events * phases.get("degradation", 0.30))
        n_fail   = n_events - n_normal - n_deg   # absorbs rounding remainder

        # Validate pattern specification
        if pattern_name is not None and pattern_names is not None:
            raise ValueError("Specify either pattern_name or pattern_names, not both.")
        if pattern_name is None and pattern_names is None:
            raise ValueError("One of pattern_name or pattern_names must be provided.")

        scenario = {
            "scenario_id":   scenario_id,
            "service":       service,
            "host":          host,
            "start_ts":      float(start_ts),
            "n_events":      n_events,
            "phases":        phases,
            "phase_boundaries": {
                "normal_end":      n_normal,
                "degradation_end": n_normal + n_deg,
                "failure_end":     n_events,
                "n_normal":        n_normal,
                "n_degradation":   n_deg,
                "n_failure":       n_fail,
            },
            "anomaly_rate": round(
                phases.get("degradation", 0.0) + phases.get("failure", 0.0), 4
            ),
        }

        if pattern_name is not None:
            scenario["pattern_name"]  = pattern_name
            scenario["pattern_names"] = [pattern_name]
        else:
            scenario["pattern_names"] = pattern_names
            # For compatibility keep a single "primary" name
            scenario["pattern_name"]  = pattern_names[0] if pattern_names else ""

        return scenario

    # ------------------------------------------------------------------
    def build_hybrid_scenario(
        self,
        scenario_id:   str,
        service:       str,
        host:          str,
        start_ts:      float,
        n_events:      int,
        pattern_names: list[str],
        phases:        Optional[dict] = None,
    ) -> dict:
        """
        Convenience wrapper to build a multi-pattern (hybrid) scenario.

        Events are distributed round-robin across *pattern_names*.
        """
        if not pattern_names:
            raise ValueError("pattern_names must be non-empty for a hybrid scenario.")
        return self.build_scenario(
            scenario_id=scenario_id,
            service=service,
            host=host,
            start_ts=start_ts,
            n_events=n_events,
            phases=phases,
            pattern_names=pattern_names,
        )
