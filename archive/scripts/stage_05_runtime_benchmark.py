# scripts/stage_05_runtime_benchmark.py

# Purpose: Measure the runtime performance (throughput, latency, and memory) 
# of the anomaly detection models in a streaming inference scenario.

# Input: Reads data/processed/events_tokenized.parquet, which contains tokenized events with timestamps, 
# service names, session IDs, token IDs, template IDs, and labels.

# Output: Writes reports/stage_05_runtime_benchmark.md, a markdown report summarizing the benchmark results, 
# and logs detailed information to ai_workspace/logs/stage_05_runtime_benchmark.log.

# Used by: This script is used by the main pipeline to evaluate the runtime performance 
# of the anomaly detection models under realistic streaming conditions. 
# It can be run independently to benchmark different models (baseline, transformer, ensemble) 
# and configurations (demo vs full mode, keying by service vs session). 
# The results help inform trade-offs between model complexity and operational efficiency.

"""
Stage 05 — Runtime Benchmark: measure throughput, latency, and memory.

Reads   : data/processed/events_tokenized.parquet
Writes  : reports/stage_05_runtime_benchmark.md
Logs to : ai_workspace/logs/stage_05_runtime_benchmark.log

Usage:
    python scripts/stage_05_runtime_benchmark.py --mode demo --model ensemble
    python scripts/stage_05_runtime_benchmark.py --mode full  --model baseline
    python scripts/stage_05_runtime_benchmark.py --mode demo --n-events 50000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime import InferenceEngine

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "ai_workspace" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_log_path = LOG_DIR / "stage_05_runtime_benchmark.log"
_handler_file = logging.FileHandler(_log_path, encoding="utf-8")
_handler_file.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
_handler_stream = logging.StreamHandler(sys.stdout)
_handler_stream.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_handler_file, _handler_stream])
log = logging.getLogger(__name__)

EVENTS_PARQUET   = ROOT / "data" / "processed" / "events_tokenized.parquet"
BENCHMARK_REPORT = REPORTS_DIR / "stage_05_runtime_benchmark.md"

DEMO_MAX_EVENTS = 20_000
FULL_MAX_EVENTS = 200_000
WINDOW_SIZE     = 50
STRIDE          = 10


def _memory_mb() -> float:
    """Return current RSS memory in MB (best-effort; requires psutil)."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def run_benchmark(mode: str, model: str, n_events: int | None, key_by: str = "service") -> dict:
    log.info("=== Stage 05 Runtime Benchmark ===")
    log.info("mode=%s  model=%s  window=%d  stride=%d  key_by=%s",
             mode, model, WINDOW_SIZE, STRIDE, key_by)

    if n_events is None:
        n_events = DEMO_MAX_EVENTS if mode == "demo" else FULL_MAX_EVENTS

    # ------------------------------------------------------------------
    # Load events
    # ------------------------------------------------------------------
    log.info("Loading %d events from %s ...", n_events, EVENTS_PARQUET)
    df = pd.read_parquet(EVENTS_PARQUET).head(n_events)
    n_total = len(df)
    log.info("Loaded %d events", n_total)

    # ------------------------------------------------------------------
    # Warm up engine (excluded from timing)
    # ------------------------------------------------------------------
    mem_before = _memory_mb()
    t_load_start = time.perf_counter()

    engine = InferenceEngine(mode=model, window_size=WINDOW_SIZE, stride=STRIDE)
    engine.load_artifacts()

    load_s = time.perf_counter() - t_load_start
    mem_after_load = _memory_mb()
    log.info("Artifact load time: %.2fs  |  RSS after load: %.1f MB", load_s, mem_after_load)

    # ------------------------------------------------------------------
    # Benchmark loop
    # ------------------------------------------------------------------
    window_latencies: list[float] = []   # wall-clock time to score one window
    n_anomalies = 0
    mem_peak = mem_after_load

    t_stream_start = time.perf_counter()

    for _, row in df.iterrows():
        ts_event = row.get("timestamp")
        try:
            ts_float = float(ts_event) if ts_event is not None and ts_event == ts_event else 0.0
        except (TypeError, ValueError):
            ts_float = 0.0

        sid = "" if key_by == "service" else str(row.get("session_id", ""))
        event = {
            "timestamp":   ts_float,
            "service":     str(row.get("service", "")),
            "session_id":  sid,
            "token_id":    int(row.get("token_id", 1)),
            "template_id": int(row.get("template_id", 0)),
            "label":       row.get("label"),
        }

        t0 = time.perf_counter()
        risk = engine.ingest(event)
        t1 = time.perf_counter()

        if risk is not None:
            window_latencies.append(t1 - t0)
            if risk.is_anomaly:
                n_anomalies += 1

        # Track peak memory every 2000 events
        if len(window_latencies) % 20 == 0:
            m = _memory_mb()
            if m > mem_peak:
                mem_peak = m

    total_stream_s = time.perf_counter() - t_stream_start
    mem_final = _memory_mb()
    if mem_final > mem_peak:
        mem_peak = mem_final

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    n_emitted    = len(window_latencies)
    events_sec   = n_total / total_stream_s if total_stream_s > 0 else 0.0
    avg_lat_ms   = (sum(window_latencies) / n_emitted * 1000) if n_emitted else 0.0
    p95_lat_ms   = 0.0
    if n_emitted:
        sorted_lats = sorted(window_latencies)
        idx = max(0, int(len(sorted_lats) * 0.95) - 1)
        p95_lat_ms = sorted_lats[idx] * 1000
    anom_rate    = (n_anomalies / n_emitted * 100) if n_emitted else 0.0

    metrics = {
        "mode":              mode,
        "model":             model,
        "key_by":            key_by,
        "n_events":          n_total,
        "n_windows_emitted": n_emitted,
        "n_anomalies":       n_anomalies,
        "anomaly_rate_pct":  round(anom_rate, 2),
        "events_per_sec":    round(events_sec, 1),
        "avg_window_latency_ms": round(avg_lat_ms, 3),
        "p95_window_latency_ms": round(p95_lat_ms, 3),
        "total_stream_s":    round(total_stream_s, 2),
        "artifact_load_s":   round(load_s, 2),
        "mem_before_mb":     round(mem_before, 1),
        "mem_after_load_mb": round(mem_after_load, 1),
        "mem_peak_mb":       round(mem_peak, 1),
    }

    log.info("Benchmark results:")
    for k, v in metrics.items():
        log.info("  %s: %s", k, v)

    # ------------------------------------------------------------------
    # Write markdown report
    # ------------------------------------------------------------------
    _write_report(metrics)
    return metrics


def _write_report(m: dict) -> None:
    lines = [
        "# Stage 05 Runtime Benchmark",
        "",
        f"**Mode:** {m['mode']}  |  **Model:** {m['model']}",
        "",
        "## Throughput",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Events processed | {m['n_events']:,} |",
        f"| Events/sec | {m['events_per_sec']:,.1f} |",
        f"| Total stream time | {m['total_stream_s']} s |",
        f"| Artifact load time | {m['artifact_load_s']} s |",
        "",
        "## Latency (per emitted window)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Windows emitted | {m['n_windows_emitted']:,} |",
        f"| Avg latency | {m['avg_window_latency_ms']} ms |",
        f"| P95 latency | {m['p95_window_latency_ms']} ms |",
        "",
        "## Anomaly Detection",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Anomalies flagged | {m['n_anomalies']:,} |",
        f"| Anomaly rate | {m['anomaly_rate_pct']} % |",
        "",
        "## Memory",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| RSS before load | {m['mem_before_mb']} MB |",
        f"| RSS after artifact load | {m['mem_after_load_mb']} MB |",
        f"| Peak RSS during stream | {m['mem_peak_mb']} MB |",
        "",
        "---",
        "_Generated by stage_05_runtime_benchmark.py_",
    ]
    report_text = "\n".join(lines)
    BENCHMARK_REPORT.write_text(report_text, encoding="utf-8")
    log.info("Benchmark report written: %s", BENCHMARK_REPORT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 Runtime Benchmark")
    parser.add_argument("--mode",     default="demo", choices=["demo", "full"])
    parser.add_argument("--model",    default="ensemble",
                        choices=["baseline", "transformer", "ensemble"])
    parser.add_argument("--n-events", type=int, default=None,
                        help="Override default event count")
    parser.add_argument("--key-by",   default="service",
                        choices=["service", "session"], dest="key_by")
    args = parser.parse_args()
    run_benchmark(args.mode, args.model, args.n_events, args.key_by)


if __name__ == "__main__":
    main()
