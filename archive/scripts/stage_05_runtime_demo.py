# scripts/stage_05_runtime_demo.py

# Purpose: Simulate a live log stream and score windows using the InferenceEngine.

# Input: Reads data/processed/events_tokenized.parquet, which contains tokenized events with timestamps, 
# service/session IDs, token/template IDs, and labels.

# Output: Writes reports/runtime_demo_results.csv (per-window scores and labels) and 
# reports/runtime_demo_evidence.jsonl (detailed evidence for each window).

# Used by: This script is a standalone runtime demo and is not directly used by other scripts, 
# but it relies on the InferenceEngine class defined in src/runtime.py, which is the core of the scoring logic. 
# The results and evidence it produces can be used for analysis and reporting in later stages.

"""
Stage 05 — Runtime Demo: simulate a live log stream and score windows.

Reads   : data/processed/events_tokenized.parquet
Writes  : reports/runtime_demo_results.csv
          reports/runtime_demo_evidence.jsonl
Logs to : ai_workspace/logs/stage_05_runtime_demo.log

Options:
    --mode   demo (20k events) | full (all events)
    --model  baseline | transformer | ensemble
    --key-by service  (default: all events per service in one stream)
             session  (per session_id, fine-grained — many short HDFS sessions)

Usage:
    python scripts/stage_05_runtime_demo.py --mode demo --model baseline
    python scripts/stage_05_runtime_demo.py --mode demo --model transformer
    python scripts/stage_05_runtime_demo.py --mode demo --model ensemble
    python scripts/stage_05_runtime_demo.py --mode full --model ensemble
"""
from __future__ import annotations

import argparse
import json
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

_log_path = LOG_DIR / "stage_05_runtime_demo.log"
_handler_file = logging.FileHandler(_log_path, encoding="utf-8")
_handler_file.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
_handler_stream = logging.StreamHandler(sys.stdout)
_handler_stream.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_handler_file, _handler_stream])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVENTS_PARQUET = ROOT / "data" / "processed" / "events_tokenized.parquet"
RESULTS_CSV    = REPORTS_DIR / "runtime_demo_results.csv"
EVIDENCE_JSONL = REPORTS_DIR / "runtime_demo_evidence.jsonl"

DEMO_MAX_EVENTS = 20_000
WINDOW_SIZE     = 50
STRIDE          = 10
PRINT_INTERVAL  = 5_000   # console progress every N events


def _row_to_event(row: dict, key_by: str = "service") -> dict:
    """Convert a parquet row (as dict) to the event format InferenceEngine expects.

    key_by="service"  -> session_id is blanked so all events from a service
                         share one rolling window (realistic streaming demo).
    key_by="session"  -> original session_id kept (fine-grained, but HDFS
                         sessions are very short and rarely fill the window).
    """
    ts = row.get("timestamp")
    try:
        ts = float(ts) if ts is not None and ts == ts else 0.0   # nan -> 0.0
    except (TypeError, ValueError):
        ts = 0.0
    sid = "" if key_by == "service" else str(row.get("session_id", ""))
    return {
        "timestamp": ts,
        "service":    str(row.get("service", "")),
        "session_id": sid,
        "token_id":   int(row.get("token_id", 1)),
        "template_id": int(row.get("template_id", 0)),
        "label":      row.get("label"),
    }


def run_demo(mode: str, model: str, key_by: str = "service",
             use_runtime_thresholds: bool = False) -> dict:
    """
    Stream events through InferenceEngine and collect RiskResults.

    Returns a summary dict with counts.
    """
    log.info("=== Stage 05 Runtime Demo ===")
    log.info("mode=%s  model=%s  window=%d  stride=%d  key_by=%s  runtime_thresholds=%s",
             mode, model, WINDOW_SIZE, STRIDE, key_by, use_runtime_thresholds)
    log.info("Parquet: %s", EVENTS_PARQUET)

    max_events = DEMO_MAX_EVENTS if mode == "demo" else None

    # ------------------------------------------------------------------
    # Load events
    # ------------------------------------------------------------------
    log.info("Loading events parquet ...")
    df = pd.read_parquet(EVENTS_PARQUET)
    if max_events is not None:
        df = df.head(max_events)
    n_total = len(df)
    log.info("Loaded %d events", n_total)

    # ------------------------------------------------------------------
    # Set up engine
    # ------------------------------------------------------------------
    engine = InferenceEngine(
        mode=model,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        use_runtime_thresholds=use_runtime_thresholds,
    )
    engine.load_artifacts()

    # ------------------------------------------------------------------
    # Stream events
    # ------------------------------------------------------------------
    results = []
    evidence_rows = []
    t_start = time.perf_counter()

    log.info("Starting stream ingestion (key_by=%s) ...", key_by)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        event = _row_to_event(row.to_dict(), key_by=key_by)
        risk = engine.ingest(event)

        if risk is not None:
            results.append({
                "ts":           risk.timestamp,
                "stream_key":   risk.stream_key,
                "model":        risk.model,
                "risk_score":   risk.risk_score,
                "threshold":    risk.threshold,
                "is_anomaly":   risk.is_anomaly,
                "label":        risk.meta.get("label"),
                "window_size":  risk.meta.get("window_size"),
                "top_template": (
                    risk.evidence_window.get("templates_preview", [""])[0]
                    if risk.evidence_window.get("templates_preview") else ""
                ),
            })
            evidence_rows.append(json.dumps(risk.to_dict(), default=str))

            # Console summary line
            top_tpl = (
                risk.evidence_window.get("templates_preview", [""])[0][:50]
                if risk.evidence_window.get("templates_preview") else ""
            )
            flag = "ANOM" if risk.is_anomaly else "    "
            print(
                f"[{flag}] {risk.stream_key[:30]:30s} | "
                f"{risk.model:10s} | "
                f"score={risk.risk_score:.4f} thr={risk.threshold:.4f} | "
                f"{top_tpl}"
            )

        if i % PRINT_INTERVAL == 0:
            elapsed = time.perf_counter() - t_start
            log.info("  %d / %d events processed | %d windows emitted | %.1fs elapsed",
                     i, n_total, len(results), elapsed)

    elapsed = time.perf_counter() - t_start
    log.info("Stream complete: %d events in %.2fs", n_total, elapsed)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if results:
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
        log.info("Results saved: %s (%d rows)", RESULTS_CSV, len(results))

        with open(EVIDENCE_JSONL, "w", encoding="utf-8") as fh:
            fh.write("\n".join(evidence_rows) + "\n")
        log.info("Evidence saved: %s", EVIDENCE_JSONL)
    else:
        log.warning("No windows emitted (too few events for window_size=%d?)", WINDOW_SIZE)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_emitted = len(results)
    n_anom    = sum(1 for r in results if r["is_anomaly"])
    anom_rate = (n_anom / n_emitted * 100) if n_emitted else 0.0
    events_per_sec = n_total / elapsed if elapsed > 0 else 0.0

    summary = {
        "events_processed": n_total,
        "windows_emitted":  n_emitted,
        "anomalies_flagged": n_anom,
        "anomaly_rate_pct": round(anom_rate, 2),
        "elapsed_s":        round(elapsed, 2),
        "events_per_sec":   round(events_per_sec, 1),
        "mode":             mode,
        "model":            model,
        "key_by":           key_by,
    }

    log.info("--- Summary ---")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)
    log.info("--- End Stage 05 Demo ---")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 Runtime Demo")
    parser.add_argument("--mode",   default="demo", choices=["demo", "full"])
    parser.add_argument("--model",  default="ensemble",
                        choices=["baseline", "transformer", "ensemble"])
    parser.add_argument("--key-by", default="service",
                        choices=["service", "session"],
                        dest="key_by",
                        help="Stream key granularity: service (one stream per service) "
                             "or session (one stream per session_id)")
    parser.add_argument("--use-runtime-thresholds", action="store_true",
                        dest="use_runtime_thresholds",
                        help="Load thresholds from artifacts/threshold_runtime.json "
                             "instead of the default threshold.json files")
    args = parser.parse_args()
    run_demo(args.mode, args.model, args.key_by,
             use_runtime_thresholds=args.use_runtime_thresholds)


if __name__ == "__main__":
    main()
