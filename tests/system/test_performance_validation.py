# tests/system/test_performance_validation.py
"""
Phase 7.5 — Performance Validation.

Measures throughput and per-event latency of the InferenceEngine under
the fallback (demo_mode=True) configuration.  Realistic model files are
NOT required; the purpose is to validate that the streaming path is
capable of handling production-scale event rates.

Targets (conservative for CI safety):
  - 10 000 events in < 30 seconds (> 333 events/sec)
  - Average per-event latency < 3 ms

These bounds are deliberately loose so the test remains stable across
machines.  Actual throughput with the fallback scorer is typically
10 000-50 000 events/sec.

Marked @pytest.mark.slow so it is deselected in the fast CI suite
(-m "not slow") but runs in extended validation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.inference_engine import InferenceEngine
from src.runtime.types import RiskResult

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _events(n: int, service: str = "bgl", session: str = "perf") -> list[dict]:
    return [
        {
            "timestamp": float(i),
            "service": service,
            "session_id": session,
            "token_id": (i % 100) + 2,
            "label": 0,
        }
        for i in range(n)
    ]


def _make_engine(window_size: int = 50, stride: int = 10) -> InferenceEngine:
    eng = InferenceEngine(
        mode="baseline",
        window_size=window_size,
        stride=stride,
    )
    eng.demo_mode = True
    eng.fallback_score = 0.0  # score=0 -> no anomalies, focus on throughput
    return eng


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

class TestPerformanceThroughput:

    N = 10_000
    MAX_TOTAL_SECONDS = 30.0
    MIN_EVENTS_PER_SEC = 333

    def test_10k_events_within_time_budget(self):
        eng = _make_engine()
        evts = _events(self.N)

        start = time.perf_counter()
        for ev in evts:
            eng.ingest(ev)
        elapsed = time.perf_counter() - start

        eps = self.N / elapsed
        avg_ms = (elapsed / self.N) * 1000

        print(
            f"\n[Performance] N={self.N} | elapsed={elapsed:.3f}s | "
            f"eps={eps:.0f} | avg_latency={avg_ms:.4f}ms"
        )

        assert elapsed < self.MAX_TOTAL_SECONDS, (
            f"10k events took {elapsed:.2f}s (limit {self.MAX_TOTAL_SECONDS}s)"
        )
        assert eps >= self.MIN_EVENTS_PER_SEC, (
            f"Throughput {eps:.0f} eps below minimum {self.MIN_EVENTS_PER_SEC} eps"
        )

    def test_window_emission_rate(self):
        """Count emitted windows to confirm stride is working at scale."""
        W, S, N = 50, 10, 10_000
        eng = _make_engine(window_size=W, stride=S)
        evts = _events(N)
        emitted = 0

        start = time.perf_counter()
        for ev in evts:
            r = eng.ingest(ev)
            if r is not None:
                emitted += 1
        elapsed = time.perf_counter() - start

        expected = (N - W) // S + 1
        wps = emitted / elapsed

        print(
            f"\n[Window rate] emitted={emitted} (expected={expected}) | "
            f"windows/sec={wps:.0f} | elapsed={elapsed:.3f}s"
        )

        assert emitted == expected

    def test_average_per_event_latency_ms(self):
        N = 10_000
        MAX_AVG_LATENCY_MS = 3.0
        eng = _make_engine()
        evts = _events(N)

        start = time.perf_counter()
        for ev in evts:
            eng.ingest(ev)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / N) * 1000
        print(f"\n[Latency] avg_per_event={avg_ms:.4f}ms (limit={MAX_AVG_LATENCY_MS}ms)")

        assert avg_ms < MAX_AVG_LATENCY_MS, (
            f"Avg per-event latency {avg_ms:.3f}ms exceeds {MAX_AVG_LATENCY_MS}ms"
        )

    def test_multi_service_throughput(self):
        """Interleaved services should not significantly reduce throughput."""
        N = 10_000
        eng = _make_engine(window_size=50, stride=10)

        events = []
        for i in range(N):
            svc = f"svc_{i % 5}"
            events.append({
                "timestamp": float(i),
                "service": svc,
                "session_id": "ms",
                "token_id": (i % 50) + 2,
                "label": 0,
            })

        start = time.perf_counter()
        for ev in events:
            eng.ingest(ev)
        elapsed = time.perf_counter() - start

        eps = N / elapsed
        print(f"\n[Multi-service] N={N} | eps={eps:.0f} | elapsed={elapsed:.3f}s")

        assert elapsed < 30.0, f"Multi-service 10k events took {elapsed:.2f}s"
