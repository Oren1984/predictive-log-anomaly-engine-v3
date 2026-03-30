# scripts/stage_05_run.py
# This script is a one-command wrapper to run the demo and benchmark for the predictive log anomaly engine. 
# It allows you to specify the mode (demo or full) and which model to use (baseline, transformer, or ensemble). 
# You can also choose to skip the demo or benchmark if you only want to run one of them.

"""
Stage 05 — One-command wrapper: runs demo + benchmark.

Usage:
    python scripts/stage_05_run.py --mode demo --model ensemble
    python scripts/stage_05_run.py --mode full  --model baseline
    python scripts/stage_05_run.py --skip-benchmark
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _run_demo(mode: str, model: str, key_by: str = "service",
              use_runtime_thresholds: bool = False) -> None:
    log.info("--- Stage 05 Demo (mode=%s, model=%s, key_by=%s, runtime_thr=%s) ---",
             mode, model, key_by, use_runtime_thresholds)
    demo_mod = importlib.import_module("scripts.stage_05_runtime_demo")
    summary = demo_mod.run_demo(mode, model, key_by,
                                use_runtime_thresholds=use_runtime_thresholds)
    log.info("Demo finished: %s windows emitted, %.1f%% anomaly rate",
             summary.get("windows_emitted", 0),
             summary.get("anomaly_rate_pct", 0.0))


def _run_benchmark(mode: str, model: str) -> None:
    log.info("--- Stage 05 Benchmark (mode=%s, model=%s) ---", mode, model)
    bench_mod = importlib.import_module("scripts.stage_05_runtime_benchmark")
    metrics = bench_mod.run_benchmark(mode, model, n_events=None)
    log.info(
        "Benchmark finished: %.1f events/sec | avg latency %.3f ms | peak %.1f MB",
        metrics.get("events_per_sec", 0.0),
        metrics.get("avg_window_latency_ms", 0.0),
        metrics.get("mem_peak_mb", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 Run Wrapper")
    parser.add_argument("--mode",  default="demo", choices=["demo", "full"])
    parser.add_argument("--model", default="ensemble",
                        choices=["baseline", "transformer", "ensemble"])
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Run demo only (skip benchmark)")
    parser.add_argument("--skip-demo", action="store_true",
                        help="Run benchmark only (skip demo)")
    parser.add_argument("--use-runtime-thresholds", action="store_true",
                        dest="use_runtime_thresholds",
                        help="Load thresholds from artifacts/threshold_runtime.json")
    args = parser.parse_args()

    if not args.skip_demo:
        _run_demo(args.mode, args.model, key_by="service",
                  use_runtime_thresholds=args.use_runtime_thresholds)

    if not args.skip_benchmark:
        _run_benchmark(args.mode, args.model)

    log.info("=== Stage 05 complete ===")
    log.info("")
    log.info("To run individually:")
    log.info("  python scripts/stage_05_runtime_demo.py --mode demo --model baseline")
    log.info("  python scripts/stage_05_runtime_demo.py --mode demo --model transformer")
    log.info("  python scripts/stage_05_runtime_demo.py --mode demo --model ensemble")
    log.info("  python scripts/stage_05_runtime_benchmark.py --mode demo --model ensemble")


if __name__ == "__main__":
    main()
