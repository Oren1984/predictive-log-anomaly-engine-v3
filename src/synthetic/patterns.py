# src/synthetic/patterns.py

# Purpose: Defines the FailurePattern abstract base class and concrete subclasses 
# representing specific failure scenarios (MemoryLeakPattern, 
# DiskFullPattern, AuthBruteForcePattern, NetworkFlapPattern).
# Each pattern simulates a specific type of failure across three phases (normal, degradation, failure) 
# and generates LogEvent objects with appropriate messages, levels, and labels.

# Input: - FailurePattern: an abstract base class that defines the interface for emitting log events based on a scenario context, 
#          including methods for determining the current phase and labeling events.
#        - MemoryLeakPattern: a concrete implementation of FailurePattern that simulates a memory leak scenario.
#        - DiskFullPattern: a concrete implementation of FailurePattern that simulates a disk filling up scenario.
#        - AuthBruteForcePattern: a concrete implementation of FailurePattern that simulates a brute-force authentication attack scenario.
#        - NetworkFlapPattern: a concrete implementation of FailurePattern that simulates a network interface becoming unstable scenario.

# Output: - Each concrete FailurePattern subclass implements the emit_event method 
#           to generate LogEvent objects with realistic messages and metadata based on the current phase of the scenario.

# Used by: - The SyntheticLogGenerator class to emit log events based on defined failure patterns and scenarios.

"""Stage 1 — Synthetic: failure pattern definitions."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..data_layer.models import LogEvent

# ---------------------------------------------------------------------------
# Fixed safe fake values for auth/network patterns
# ---------------------------------------------------------------------------
_USERS = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry"]
_IPS   = [
    "10.0.1.10", "10.0.1.20", "192.168.0.100",
    "172.16.0.5", "10.10.5.33", "192.168.1.200",
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class FailurePattern(ABC):
    """
    Abstract base for all synthetic failure patterns.

    A pattern generates LogEvent objects across three phases:
        normal      → service operating normally  (label=0)
        degradation → service degrading           (label=1)
        failure     → service fully failed        (label=1)

    The phase boundaries are derived from ctx["phases"] and ctx["n_events"].
    """

    name: str = "base"
    severity_curve: Optional[Callable[[float], float]] = None  # progress -> scale

    # ------------------------------------------------------------------
    @abstractmethod
    def emit_event(self, t: int, ctx: dict) -> LogEvent:
        """Return the LogEvent for event index *t* in the given scenario context.

        Parameters
        ----------
        t   : zero-based event index within the scenario
        ctx : scenario context dict (see SyntheticLogGenerator.generate)
        """

    # ------------------------------------------------------------------
    def is_failure_phase(self, t: int, ctx: dict) -> bool:
        """True when *t* falls in the failure phase."""
        return self._get_phase(t, ctx) == "failure"

    def label_for_event(self, t: int, ctx: dict) -> int:
        """
        0 for the normal phase; 1 for degradation or failure.

        Degradation phase is labeled anomalous from the start so that
        downstream anomaly detectors have clearly labeled positive examples.
        """
        return 0 if self._get_phase(t, ctx) == "normal" else 1

    # ------------------------------------------------------------------
    # Shared helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def _get_phase(self, t: int, ctx: dict) -> str:
        """Return "normal", "degradation", or "failure" for event index t."""
        n      = ctx["n_events"]
        phases = ctx["phases"]
        n_normal = int(n * phases.get("normal", 0.60))
        n_deg    = int(n * phases.get("degradation", 0.30))
        if t < n_normal:
            return "normal"
        if t < n_normal + n_deg:
            return "degradation"
        return "failure"

    def _degradation_progress(self, t: int, ctx: dict) -> float:
        """Return a progress value in [0, 1] across the degradation window."""
        n      = ctx["n_events"]
        phases = ctx["phases"]
        n_normal = int(n * phases.get("normal", 0.60))
        n_deg    = max(int(n * phases.get("degradation", 0.30)), 1)
        return min(1.0, (t - n_normal) / n_deg)

    def _session_id(self, t: int, ctx: dict, window: int = 50) -> str:
        """Build a sliding-window session ID (50 events per session by default)."""
        base = ctx.get("scenario_id", "synth")
        return f"{base}_s{t // window}"


# ---------------------------------------------------------------------------
# 1. MemoryLeakPattern
# ---------------------------------------------------------------------------

class MemoryLeakPattern(FailurePattern):
    """
    Simulates a gradually worsening memory leak that eventually OOM-kills
    the process.

    normal      : heap stable at 64–256 MB
    degradation : heap grows linearly toward 2 GB, GC pressure rises
    failure     : process OOM-killed at 3–4 GB heap
    """

    name = "memory_leak"

    def emit_event(self, t: int, ctx: dict) -> LogEvent:
        phase = self._get_phase(t, ctx)
        rng   = ctx["rng"]
        ts    = float(ctx.get("start_ts", 0.0)) + t
        svc   = ctx.get("service", "app-server")
        host  = ctx.get("host",    "host-01")
        sid   = self._session_id(t, ctx)

        if phase == "normal":
            heap = rng.randint(64, 256)
            rss  = heap + rng.randint(10, 40)
            gc   = rng.randint(0, 3)
            msg  = (f"memory_check heap={heap}MB rss={rss}MB "
                    f"gc_runs={gc} status=ok")
            level = "INFO"

        elif phase == "degradation":
            progress = self._degradation_progress(t, ctx)
            scale    = self.severity_curve(progress) if self.severity_curve else progress
            heap = int(256 + scale * 1792) + rng.randint(-32, 32)
            heap = max(257, heap)
            rss  = heap + rng.randint(50, 150)
            gc   = rng.randint(15, 80)
            msg  = (f"memory_check heap={heap}MB rss={rss}MB "
                    f"gc_runs={gc} status=degraded pressure=high")
            level = "WARNING"

        else:  # failure
            heap = rng.randint(2560, 4096)
            rss  = heap + rng.randint(200, 500)
            msg  = (f"memory_check heap={heap}MB rss={rss}MB "
                    f"gc_runs=0 status=oom_killed signal=SIGKILL")
            level = "ERROR"

        return LogEvent(
            timestamp=ts,
            service=svc,
            level=level,
            message=msg,
            meta={
                "host":        host,
                "component":   "memory_monitor",
                "scenario_id": ctx.get("scenario_id", ""),
                "phase":       phase,
                "session_id":  sid,
            },
            label=self.label_for_event(t, ctx),
        )


# ---------------------------------------------------------------------------
# 2. DiskFullPattern
# ---------------------------------------------------------------------------

class DiskFullPattern(FailurePattern):
    """
    Simulates a disk filling up over time until writes are blocked.

    normal      : /var/data usage 20–60%, plenty of free space
    degradation : usage climbs 65–95%, low free space warnings
    failure     : disk 100% full, writes blocked
    """

    name = "disk_full"

    def emit_event(self, t: int, ctx: dict) -> LogEvent:
        phase = self._get_phase(t, ctx)
        rng   = ctx["rng"]
        ts    = float(ctx.get("start_ts", 0.0)) + t
        svc   = ctx.get("service", "storage")
        host  = ctx.get("host",    "host-02")
        sid   = self._session_id(t, ctx)

        if phase == "normal":
            pct  = rng.randint(20, 60)
            free = round((100 - pct) * 0.5, 1)   # GB
            inod = rng.randint(10, 40)
            msg  = (f"disk_check path=/var/data used={pct}% "
                    f"free={free}GB inodes_used={inod}% status=ok")
            level = "INFO"

        elif phase == "degradation":
            progress = self._degradation_progress(t, ctx)
            scale    = self.severity_curve(progress) if self.severity_curve else progress
            pct  = int(65 + scale * 30) + rng.randint(-2, 2)
            pct  = min(98, max(65, pct))
            free = round((100 - pct) * 5.0, 0)   # MB (dwindling)
            msg  = (f"disk_check path=/var/data used={pct}% "
                    f"free={free:.0f}MB status=warning cleanup_needed")
            level = "WARNING"

        else:  # failure
            pct = rng.randint(99, 100)
            msg = (f"disk_check path=/var/data used={pct}% "
                   f"free=0MB status=full writes_blocked io_error=ENOSPC")
            level = "ERROR"

        return LogEvent(
            timestamp=ts,
            service=svc,
            level=level,
            message=msg,
            meta={
                "host":        host,
                "component":   "disk_monitor",
                "scenario_id": ctx.get("scenario_id", ""),
                "phase":       phase,
                "session_id":  sid,
            },
            label=self.label_for_event(t, ctx),
        )


# ---------------------------------------------------------------------------
# 3. AuthBruteForcePattern
# ---------------------------------------------------------------------------

class AuthBruteForcePattern(FailurePattern):
    """
    Simulates a brute-force login attack escalating to account lockouts.

    normal      : successful logins from legitimate users
    degradation : increasing failed login attempts from suspicious IPs
    failure     : accounts locked, auth service overwhelmed
    """

    name = "auth_brute_force"

    def emit_event(self, t: int, ctx: dict) -> LogEvent:
        phase = self._get_phase(t, ctx)
        rng   = ctx["rng"]
        ts    = float(ctx.get("start_ts", 0.0)) + t
        svc   = ctx.get("service", "auth-service")
        host  = ctx.get("host",    "host-03")
        sid   = self._session_id(t, ctx)

        user = rng.choice(_USERS)
        ip   = rng.choice(_IPS)

        if phase == "normal":
            session_tok = rng.randint(100000, 999999)
            msg  = (f"auth user={user} action=login "
                    f"src={ip} status=success session={session_tok}")
            level = "INFO"

        elif phase == "degradation":
            progress = self._degradation_progress(t, ctx)
            attempt  = int(1 + progress * 9) + rng.randint(0, 3)
            max_att  = 10
            msg  = (f"auth user={user} action=login "
                    f"src={ip} status=failed attempt={attempt}/{max_att}")
            level = "WARNING"

        else:  # failure
            locked_dur = rng.choice([300, 600, 900, 1800])
            total_fail = rng.randint(10, 50)
            msg  = (f"auth user={user} action=lockout "
                    f"src={ip} failed_attempts={total_fail} "
                    f"lockout_duration={locked_dur}s")
            level = "ERROR"

        return LogEvent(
            timestamp=ts,
            service=svc,
            level=level,
            message=msg,
            meta={
                "host":        host,
                "component":   "auth_daemon",
                "scenario_id": ctx.get("scenario_id", ""),
                "phase":       phase,
                "session_id":  sid,
            },
            label=self.label_for_event(t, ctx),
        )


# ---------------------------------------------------------------------------
# 4. NetworkFlapPattern
# ---------------------------------------------------------------------------

class NetworkFlapPattern(FailurePattern):
    """
    Simulates a network interface becoming unstable and eventually going down.

    normal      : eth0 stable, low latency, zero packet loss
    degradation : increasing latency and packet loss, retransmits rising
    failure     : interface down, repeated flap events
    """

    name = "network_flap"

    def emit_event(self, t: int, ctx: dict) -> LogEvent:
        phase = self._get_phase(t, ctx)
        rng   = ctx["rng"]
        ts    = float(ctx.get("start_ts", 0.0)) + t
        svc   = ctx.get("service", "network")
        host  = ctx.get("host",    "host-04")
        sid   = self._session_id(t, ctx)

        if phase == "normal":
            latency  = rng.randint(5, 50)
            bw       = rng.randint(900, 1000)
            msg  = (f"net_check iface=eth0 state=up "
                    f"latency={latency}ms loss=0.00% bw={bw}Mbps")
            level = "INFO"

        elif phase == "degradation":
            progress = self._degradation_progress(t, ctx)
            scale    = self.severity_curve(progress) if self.severity_curve else progress
            latency  = int(50 + scale * 950) + rng.randint(-20, 20)
            latency  = max(51, latency)
            loss     = round(scale * 25.0 + rng.uniform(-1.0, 1.0), 2)
            loss     = max(0.1, min(25.0, loss))
            retx     = int(scale * 500) + rng.randint(0, 50)
            msg  = (f"net_check iface=eth0 state=degraded "
                    f"latency={latency}ms loss={loss}% retx={retx}")
            level = "WARNING"

        else:  # failure
            flap_cnt = rng.randint(5, 30)
            msg  = (f"net_check iface=eth0 state=down "
                    f"flap_count={flap_cnt} uptime=0s link_failure=yes")
            level = "ERROR"

        return LogEvent(
            timestamp=ts,
            service=svc,
            level=level,
            message=msg,
            meta={
                "host":        host,
                "component":   "net_monitor",
                "scenario_id": ctx.get("scenario_id", ""),
                "phase":       phase,
                "session_id":  sid,
            },
            label=self.label_for_event(t, ctx),
        )
