# src/alerts/manager.py

# Purpose: Implement the AlertManager class, which manages the lifecycle of 
# alerts including deduplication and cooldown logic.

# Input: The AlertManager class takes an AlertPolicy and an optional clock function. 
# It provides an emit method that evaluates a RiskResult against the policy 
# and cooldown state to determine whether to fire an alert or suppress it.

# Output: The emit method returns a list of Alert objects when an alert is fired, 
# or an empty list when the alert is suppressed due to policy rules or cooldown. 
# The AlertManager also tracks statistics on the number of alerts fired and suppressed.

# Used by: The AlertManager is used in the main API implementation (src.api.app.py) 
# to manage alerts based on the results from the inference engine. 
# It is also tested in various test files (e.g., test_stage_06_alert_policy.py, test_stage_06_dedup_cooldown.py) 
# to ensure that the alerting logic works as expected under different conditions.

"""Stage 6 — Alerts: AlertManager with deduplication and cooldown."""
from __future__ import annotations

import time
from typing import Callable, Optional, TYPE_CHECKING

from .models import Alert, AlertPolicy

if TYPE_CHECKING:
    from ..runtime.types import RiskResult


class AlertManager:
    """
    Manage alert lifecycle: deduplication, cooldown, and emission.

    emit(risk_result) -> list[Alert]
      Returns [Alert] when the result should fire; [] when suppressed.

    Suppression rules (in order):
      1. risk_result.is_anomaly is False  -> no alert (policy.should_alert)
      2. score < policy.threshold (if > 0) -> filtered by policy
      3. Same stream_key alerted within cooldown_seconds -> cooldown suppression

    Parameters
    ----------
    policy      : AlertPolicy controlling thresholds, cooldown, severity.
    clock_fn    : callable returning current time in seconds (injectable for tests).
    """

    def __init__(
        self,
        policy: AlertPolicy,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.policy = policy
        self._clock: Callable[[], float] = clock_fn or time.time

        # stream_key -> timestamp of last fired alert
        self._last_alert: dict[str, float] = {}
        self._alert_count: int = 0
        self._suppressed_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def emit(self, risk_result: "RiskResult") -> list[Alert]:
        """
        Evaluate *risk_result* against policy and cooldown state.

        Returns
        -------
        list[Alert]  — [alert] if fired, [] if suppressed.
        """
        if not self.policy.should_alert(risk_result):
            return []

        stream_key = risk_result.stream_key
        now = self._clock()

        last = self._last_alert.get(stream_key)
        if last is not None and (now - last) < self.policy.cooldown_seconds:
            self._suppressed_count += 1
            return []

        alert = self.policy.risk_to_alert(risk_result)
        self._last_alert[stream_key] = now
        self._alert_count += 1
        return [alert]

    # ------------------------------------------------------------------
    # Stats / control
    # ------------------------------------------------------------------

    @property
    def alert_count(self) -> int:
        """Total alerts fired (not suppressed)."""
        return self._alert_count

    @property
    def suppressed_count(self) -> int:
        """Total alerts suppressed by cooldown."""
        return self._suppressed_count

    @property
    def active_stream_keys(self) -> list[str]:
        """Stream keys that have at least one fired alert recorded."""
        return list(self._last_alert.keys())

    def reset(self) -> None:
        """Clear all cooldown state and counters."""
        self._last_alert.clear()
        self._alert_count = 0
        self._suppressed_count = 0
