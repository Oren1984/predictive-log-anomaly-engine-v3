#!/usr/bin/env python
# scripts/validation/run_performance_validation.py
"""
Phase 7.5 — Performance Validation Script.

Measures InferenceEngine throughput at scale (10 000, 50 000 events).
Uses demo_mode=True (fallback scorer) so no trained models are required.

Usage:
    python scripts/validation/run_performance_validation.py

Output:
    Console summary with events/sec, latency per event, window emission rate.
    Results printed only (no model files written).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.inference_engine import InferenceEngine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIOS = [
    {"n": 1_000,  "window_size": 50, "stride": 10, "label": "1k events"},
    {"n": 10_000, "window_size": 50, "stride": 10, "label": "10k events"},
    {"n": 50_000, "window_size": 50, "stride": 10, "label": "50k events"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _events(n: int, service: str = "bgl", session: str = "perf") -> list[dict]:
    return [
        {
            "timestamp": float(i),
            "service": service,
            "session_id": session,
            "token_id": (i % 200) + 2,
            "label": 0,
        }
        for i in range(n)
    ]


def _make_engine(window_size: int, stride: int) -> InferenceEngine:
    eng = InferenceEngine(mode="baseline", window_size=window_size, stride=stride)
    eng.demo_mode = True
    eng.fallback_score = 0.0
    return eng


def _run_scenario(n: int, window_size: int, stride: int, label: str) -> dict:
    eng = _make_engine(window_size=window_size, stride=stride)
    evts = _events(n)

    emitted = 0
    start = time.perf_counter()
    for ev in evts:
        r = eng.ingest(ev)
        if r is not None:
            emitted += 1
    elapsed = time.perf_counter() - start

    eps = n / elapsed
    avg_ms = (elapsed / n) * 1000
    expected_windows = (n - window_size) // stride + 1

    return {
        "label": label,
        "n": n,
        "elapsed_s": round(elapsed, 4),
        "eps": round(eps, 1),
        "avg_ms": round(avg_ms, 4),
        "emitted": emitted,
        "expected_windows": expected_windows,
        "windows_correct": emitted == expected_windows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  Phase 7.5 - Performance Validation")
    print("  Mode: baseline (demo_mode=True, fallback scorer)")
    print("=" * 65)

    results = []
    for scenario in SCENARIOS:
        print(f"\nRunning: {scenario['label']} ...", flush=True)
        res = _run_scenario(**scenario)
        results.append(res)

        status = "OK" if res["windows_correct"] else "MISMATCH"
        print(f"  Events      : {res['n']:,}")
        print(f"  Elapsed     : {res['elapsed_s']:.4f}s")
        print(f"  Throughput  : {res['eps']:,.0f} events/sec")
        print(f"  Avg latency : {res['avg_ms']:.4f} ms/event")
        print(f"  Windows     : {res['emitted']} (expected {res['expected_windows']}) [{status}]")

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  {'Scenario':<18} {'EPS':>10} {'Avg ms':>10} {'Windows':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(
            f"  {r['label']:<18} {r['eps']:>10,.0f} "
            f"{r['avg_ms']:>10.4f} {r['emitted']:>10,}"
        )

    print("\n  All scenarios completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
