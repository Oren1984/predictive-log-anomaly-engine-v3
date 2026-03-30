#!/usr/bin/env python
# scripts/validation/run_memory_validation.py
"""
Phase 7.5 — Memory Stability Validation Script.

Validates that memory usage does not grow unboundedly during long streaming
runs (100 000 events).  Uses demo_mode=True (fallback scorer) so no trained
models are required.

Reports:
  - RSS memory before, during (every 10k events), and after
  - Peak memory observed
  - Memory growth (before -> after)
  - Conclusion: PASS / WARN based on observed growth

Requires: psutil  (already in requirements.txt)

Usage:
    python scripts/validation/run_memory_validation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from src.runtime.inference_engine import InferenceEngine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EVENTS      = 100_000
WINDOW_SIZE   = 50
STRIDE        = 10
SNAPSHOT_FREQ = 10_000   # take memory snapshot every N events
MAX_GROWTH_MB = 200       # warn if RSS grows by more than this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    if not _PSUTIL_AVAILABLE:
        return 0.0
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def _events_gen(n: int):
    for i in range(n):
        yield {
            "timestamp": float(i),
            "service": "bgl",
            "session_id": f"mem_{i % 100}",   # 100 distinct sessions
            "token_id": (i % 200) + 2,
            "label": 0,
        }


def _make_engine() -> InferenceEngine:
    eng = InferenceEngine(mode="baseline", window_size=WINDOW_SIZE, stride=STRIDE)
    eng.demo_mode = True
    eng.fallback_score = 0.0
    return eng


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  Phase 7.5 - Memory Stability Validation")
    if not _PSUTIL_AVAILABLE:
        print("  WARNING: psutil not available - memory metrics unavailable")
    print(f"  N={N_EVENTS:,} | window={WINDOW_SIZE} | stride={STRIDE}")
    print(f"  Sessions: 100 rotating (tests LRU eviction)")
    print("=" * 65)

    eng = _make_engine()

    mem_before = _rss_mb()
    print(f"\n  RSS before  : {mem_before:.1f} MB")

    snapshots: list[tuple[int, float]] = [(0, mem_before)]
    emitted = 0
    t_start = time.perf_counter()

    for i, ev in enumerate(_events_gen(N_EVENTS), start=1):
        r = eng.ingest(ev)
        if r is not None:
            emitted += 1

        if i % SNAPSHOT_FREQ == 0:
            rss = _rss_mb()
            snapshots.append((i, rss))
            elapsed = time.perf_counter() - t_start
            eps = i / elapsed
            print(
                f"  [{i:>7,}] RSS={rss:.1f} MB  "
                f"| emitted={emitted:,}  | eps={eps:.0f}"
            )

    elapsed_total = time.perf_counter() - t_start
    mem_after = _rss_mb()
    snapshots.append((N_EVENTS, mem_after))

    peak_mb = max(m for _, m in snapshots)
    growth_mb = mem_after - mem_before
    eps = N_EVENTS / elapsed_total

    print("\n" + "=" * 65)
    print("  Results")
    print("=" * 65)
    print(f"  Events processed : {N_EVENTS:,}")
    print(f"  Windows emitted  : {emitted:,}")
    print(f"  Elapsed          : {elapsed_total:.2f}s")
    print(f"  Throughput       : {eps:.0f} events/sec")
    print(f"  RSS before       : {mem_before:.1f} MB")
    print(f"  RSS after        : {mem_after:.1f} MB")
    print(f"  RSS peak         : {peak_mb:.1f} MB")
    print(f"  RSS growth       : {growth_mb:+.1f} MB")
    print(f"  Active keys now  : {len(eng.buffer.active_keys())}")

    if not _PSUTIL_AVAILABLE:
        verdict = "SKIP (psutil unavailable)"
    elif growth_mb > MAX_GROWTH_MB:
        verdict = f"WARN - growth {growth_mb:.1f} MB exceeds {MAX_GROWTH_MB} MB threshold"
    else:
        verdict = "PASS"

    print(f"\n  Verdict: {verdict}")
    print("=" * 65)


if __name__ == "__main__":
    main()
