# scripts/stage_05_runtime_calibrate.py

# Purpose: Perform stream-based threshold calibration for the runtime demo by streaming events through the InferenceEngine,
# collecting risk scores, and choosing thresholds based on either F1 maximisation (if valid labels are present) 
# or a percentile method targeting a specified alert rate.

# Input: Reads data/processed/events_tokenized.parquet, which contains tokenized events with timestamps, 
# service/session IDs, token/template IDs, and labels.

# Output: Writes artifacts/threshold_runtime.json (calibrated thresholds and metadata), 
# reports/runtime_calibration_scores.csv (per-window scores and labels), 
# reports/stage_31_runtime_calibration_report.md (a markdown report summarizing the calibration process and results), 
# and logs to ai_workspace/logs/stage_05_runtime_calibrate.log.

# Used by: This script is used by the main pipeline to perform runtime threshold calibration for the anomaly detection models. 
# The generated thresholds are used in the runtime demo (stage_05_runtime_demo.py) 
# to flag anomalies based on the collected risk scores. 
# The report provides insights into the calibration process, chosen thresholds, and achieved alert rates.

"""
Stage 31 — Runtime Calibration: stream-based threshold calibration (NO retraining).

Streams N events through InferenceEngine per model (baseline, transformer, ensemble),
collects risk scores, and computes calibrated thresholds via F1 or percentile method.

Outputs:
    artifacts/threshold_runtime.json
    reports/runtime_calibration_scores.csv
    reports/stage_31_runtime_calibration_report.md
Logs to:
    ai_workspace/logs/stage_05_runtime_calibrate.log

Usage:
    python scripts/stage_05_runtime_calibrate.py --mode demo --model ensemble --n-events 50000 --target-alert-rate 0.005
    python scripts/stage_05_runtime_calibrate.py --mode demo --model baseline --n-events 20000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime import InferenceEngine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVENTS_PARQUET  = ROOT / "data" / "processed" / "events_tokenized.parquet"
ARTIFACTS_DIR   = ROOT / "artifacts"
REPORTS_DIR     = ROOT / "reports"
LOG_DIR         = ROOT / "ai_workspace" / "logs"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_RUNTIME_PATH = ARTIFACTS_DIR / "threshold_runtime.json"
SCORES_CSV_PATH        = REPORTS_DIR   / "runtime_calibration_scores.csv"
REPORT_MD_PATH         = REPORTS_DIR   / "stage_31_runtime_calibration_report.md"
LOG_PATH               = LOG_DIR       / "stage_05_runtime_calibrate.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_file_handler   = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(_fmt)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

# Force-add to root logger so messages reach the file regardless of whether
# basicConfig has already been called (e.g. under pytest).
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_PATH)
    for h in _root_logger.handlers
):
    _root_logger.addHandler(_file_handler)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
           for h in _root_logger.handlers):
    _root_logger.addHandler(_stream_handler)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_event(row: dict, key_by: str = "service") -> dict:
    """Convert a parquet row to the event format InferenceEngine expects."""
    ts = row.get("timestamp")
    try:
        ts = float(ts) if ts is not None and ts == ts else 0.0
    except (TypeError, ValueError):
        ts = 0.0
    sid = "" if key_by == "service" else str(row.get("session_id", ""))
    return {
        "timestamp":   ts,
        "service":     str(row.get("service", "")),
        "session_id":  sid,
        "token_id":    int(row.get("token_id", 1)),
        "template_id": int(row.get("template_id", 0)),
        "label":       row.get("label"),
    }


def _mem_mb() -> Optional[float]:
    """Return current RSS memory usage in MB if psutil available."""
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _collect_scores(
    model_mode: str,
    df: pd.DataFrame,
    key_by: str,
    window_size: int,
    stride: int,
) -> list[dict]:
    """
    Stream events through InferenceEngine in *model_mode*, collect one row per
    emitted window: {timestamp, stream_key, model, risk_score, label}.
    """
    log.info("Collecting scores for model=%s ...", model_mode)
    t0 = time.perf_counter()
    mem_before = _mem_mb()

    engine = InferenceEngine(
        mode=model_mode,
        window_size=window_size,
        stride=stride,
        root=ROOT,
    )
    engine.load_artifacts()

    mem_after_load = _mem_mb()
    if mem_before is not None and mem_after_load is not None:
        log.info("Memory after load: %.1f MB (delta +%.1f MB)",
                 mem_after_load, mem_after_load - mem_before)

    rows = []
    for _, row in df.iterrows():
        event = _row_to_event(row.to_dict(), key_by=key_by)
        result = engine.ingest(event)
        if result is not None:
            lbl = result.meta.get("label")
            try:
                lbl = int(lbl) if lbl is not None and lbl == lbl else None
            except (TypeError, ValueError):
                lbl = None
            rows.append({
                "timestamp":  result.timestamp,
                "stream_key": result.stream_key,
                "model":      result.model,
                "risk_score": result.risk_score,
                "label":      lbl,
            })

    elapsed = time.perf_counter() - t0
    n_events = len(df)
    throughput = n_events / elapsed if elapsed > 0 else 0.0
    log.info(
        "model=%s: %d events -> %d windows in %.2fs (%.0f events/s)",
        model_mode, n_events, len(rows), elapsed, throughput,
    )
    return rows


def _score_stats(scores: np.ndarray) -> dict:
    """Compute min, p50, p95, p99, max for a score array."""
    if len(scores) == 0:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "min": float(np.min(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(np.max(scores)),
    }


def _calibrate_threshold(
    scores: np.ndarray,
    labels: Optional[np.ndarray],
    target_alert_rate: float,
) -> tuple[float, str]:
    """
    Choose calibration threshold.

    Method A (f1): if valid binary labels exist, maximise F1.
    Method B (percentile): otherwise use quantile(1 - target_alert_rate).

    Returns (threshold, method_name).
    """
    scores = np.asarray(scores, dtype=float)

    # Determine whether F1 calibration is feasible
    use_f1 = False
    if labels is not None and len(labels) == len(scores):
        valid_mask = ~np.isnan(labels.astype(float))
        valid_labels = labels[valid_mask].astype(int)
        if len(valid_labels) >= 20 and len(np.unique(valid_labels)) == 2:
            use_f1 = True
            log.info("Labels found with %d valid windows; using F1 calibration.",
                     int(valid_mask.sum()))

    if use_f1:
        valid_mask = ~np.isnan(labels.astype(float))
        s = scores[valid_mask]
        y = labels[valid_mask].astype(int)
        # Try 200 candidate thresholds across score range
        candidates = np.linspace(s.min(), s.max(), 200)
        best_f1, best_thr = -1.0, float(candidates[0])
        for thr in candidates:
            preds = (s >= thr).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

        # Sanity check: F1 threshold must not flag the vast majority of windows.
        # If >50% alert rate, the calibration slice is skewed (predominantly
        # anomalous) — fall back to percentile so the demo stays realistic.
        f1_alert_rate = float((scores >= best_thr).mean())
        if f1_alert_rate <= 0.50:
            log.info("F1 calibration: best_f1=%.4f threshold=%.6f alert_rate=%.4f",
                     best_f1, best_thr, f1_alert_rate)
            return best_thr, "f1"

        log.warning(
            "F1-optimal threshold gives alert_rate=%.2f%% (>50%%); "
            "calibration slice is skewed — falling back to percentile method.",
            f1_alert_rate * 100,
        )

    # Percentile method
    q = float(np.quantile(scores, 1.0 - target_alert_rate))
    achieved = float((scores >= q).mean())
    log.info(
        "Percentile calibration: target_alert_rate=%.4f threshold=%.6f achieved=%.4f",
        target_alert_rate, q, achieved,
    )
    return q, "percentile"


def _achieved_alert_rate(scores: np.ndarray, threshold: float) -> float:
    if len(scores) == 0:
        return 0.0
    return float((scores >= threshold).mean())


# ---------------------------------------------------------------------------
# Main calibration run
# ---------------------------------------------------------------------------

def run_calibration(
    mode: str = "demo",
    model: str = "ensemble",
    n_events: int = 50_000,
    key_by: str = "service",
    window_size: int = 50,
    stride: int = 10,
    target_alert_rate: float = 0.005,
) -> dict:
    """
    Run stream-based calibration for all three models (baseline, transformer, ensemble).

    Parameters
    ----------
    mode              : "demo" labelling for artifact metadata
    model             : kept for CLI compatibility; always calibrates all three internally
    n_events          : number of events to stream from parquet
    key_by            : stream key granularity ("service" | "session")
    window_size       : rolling window size (tokens)
    stride            : stride between emitted windows
    target_alert_rate : target fraction of windows flagged as anomalies (for percentile method)

    Returns calibration summary dict.
    """
    t_start = time.perf_counter()
    log.info("=" * 60)
    log.info("Stage 31 — Runtime Calibration START")
    log.info(
        "mode=%s  n_events=%d  key_by=%s  window=%d  stride=%d  target_rate=%.4f",
        mode, n_events, key_by, window_size, stride, target_alert_rate,
    )
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Load and slice parquet (sorted by timestamp for realistic streaming)
    # ------------------------------------------------------------------
    log.info("Loading parquet: %s", EVENTS_PARQUET)
    df_full = pd.read_parquet(EVENTS_PARQUET)
    if "timestamp" in df_full.columns:
        df_full = df_full.sort_values("timestamp").reset_index(drop=True)
    df = df_full.head(n_events)
    actual_n_events = len(df)
    log.info("Using %d events for calibration", actual_n_events)

    # ------------------------------------------------------------------
    # Collect scores for each model
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    score_data: dict[str, np.ndarray]  = {}
    label_data: dict[str, Optional[np.ndarray]] = {}

    for m in ("baseline", "transformer", "ensemble"):
        rows = _collect_scores(m, df, key_by, window_size, stride)
        all_rows.extend(rows)
        scores = np.array([r["risk_score"] for r in rows], dtype=float)
        score_data[m] = scores

        # Labels: use max label per window (already aggregated by SequenceBuffer)
        raw_labels = [r["label"] for r in rows]
        if any(l is not None for l in raw_labels):
            lbl_arr = np.array(
                [float(l) if l is not None else float("nan") for l in raw_labels]
            )
            label_data[m] = lbl_arr
        else:
            label_data[m] = None

    n_windows = len([r for r in all_rows if r["model"] == "baseline"])
    log.info("Windows emitted per model: %d", n_windows)

    # ------------------------------------------------------------------
    # Calibrate thresholds per model
    # ------------------------------------------------------------------
    thresholds: dict[str, float] = {}
    methods: dict[str, str]      = {}
    stats: dict[str, dict]       = {}
    alert_rates: dict[str, float] = {}

    for m in ("baseline", "transformer", "ensemble"):
        scores = score_data[m]
        labels = label_data[m]
        thr, meth = _calibrate_threshold(scores, labels, target_alert_rate)
        thresholds[m] = round(thr, 8)
        methods[m]    = meth
        stats[m]      = _score_stats(scores)
        alert_rates[m] = _achieved_alert_rate(scores, thr)
        log.info(
            "Model %-12s | method=%-10s | threshold=%.6f | alert_rate=%.4f",
            m, meth, thr, alert_rates[m],
        )

    # Unified method and mode (all three use the same logic, pick from ensemble)
    chosen_method = methods.get("ensemble", "percentile")

    # ------------------------------------------------------------------
    # Save artifacts/threshold_runtime.json
    # ------------------------------------------------------------------
    generated_at = datetime.now(timezone.utc).isoformat()
    runtime_artifact = {
        "generated_at":      generated_at,
        "mode":              mode,
        "key_by":            key_by,
        "window_size":       window_size,
        "stride":            stride,
        "n_events":          actual_n_events,
        "n_windows":         n_windows,
        "method":            chosen_method,
        "target_alert_rate": target_alert_rate,
        "thresholds": {
            "baseline":    thresholds["baseline"],
            "transformer": thresholds["transformer"],
            "ensemble":    thresholds["ensemble"],
        },
        "score_stats": {
            "baseline":    stats["baseline"],
            "transformer": stats["transformer"],
            "ensemble":    stats["ensemble"],
        },
    }

    with open(THRESHOLD_RUNTIME_PATH, "w", encoding="utf-8") as fh:
        json.dump(runtime_artifact, fh, indent=2)
    log.info("Saved: %s", THRESHOLD_RUNTIME_PATH)

    # ------------------------------------------------------------------
    # Save reports/runtime_calibration_scores.csv
    # ------------------------------------------------------------------
    df_scores = pd.DataFrame(all_rows)
    df_scores.to_csv(SCORES_CSV_PATH, index=False)
    log.info("Saved: %s (%d rows)", SCORES_CSV_PATH, len(df_scores))

    # ------------------------------------------------------------------
    # Save reports/stage_31_runtime_calibration_report.md
    # ------------------------------------------------------------------
    _write_report(
        generated_at=generated_at,
        mode=mode,
        n_events=actual_n_events,
        n_windows=n_windows,
        key_by=key_by,
        window_size=window_size,
        stride=stride,
        target_alert_rate=target_alert_rate,
        method=chosen_method,
        methods=methods,
        thresholds=thresholds,
        alert_rates=alert_rates,
        stats=stats,
        elapsed=time.perf_counter() - t_start,
    )

    elapsed_total = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info("Stage 31 — Runtime Calibration COMPLETE in %.2fs", elapsed_total)
    log.info("Outputs:")
    log.info("  %s", THRESHOLD_RUNTIME_PATH)
    log.info("  %s", SCORES_CSV_PATH)
    log.info("  %s", REPORT_MD_PATH)
    log.info("  %s", LOG_PATH)
    log.info("=" * 60)

    return runtime_artifact


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    generated_at: str,
    mode: str,
    n_events: int,
    n_windows: int,
    key_by: str,
    window_size: int,
    stride: int,
    target_alert_rate: float,
    method: str,
    methods: dict,
    thresholds: dict,
    alert_rates: dict,
    stats: dict,
    elapsed: float,
) -> None:
    cmd = (
        f"python scripts/stage_05_runtime_calibrate.py "
        f"--mode {mode} --model ensemble "
        f"--n-events {n_events} "
        f"--target-alert-rate {target_alert_rate}"
    )

    method_rationale = (
        "Labels exist in the calibration windows and contain both normal/anomaly "
        "classes, so **F1 maximisation** was used to choose the threshold."
        if method == "f1"
        else
        "Labels were absent or contained only one class in the calibration windows, "
        "so **percentile calibration** was used: "
        f"`threshold = quantile(scores, 1 - {target_alert_rate})`, "
        f"targeting {target_alert_rate * 100:.2f}% of windows flagged."
    )

    rows_thr = "\n".join(
        f"| {m:12s} | {thresholds[m]:.6f} | {methods[m]:12s} | {alert_rates[m]*100:.3f}% |"
        for m in ("baseline", "transformer", "ensemble")
    )

    stats_lines = []
    for m in ("baseline", "transformer", "ensemble"):
        s = stats[m]
        stats_lines.append(
            f"| {m:12s} | {s['min']:.4f} | {s['p50']:.4f} | "
            f"{s['p95']:.4f} | {s['p99']:.4f} | {s['max']:.4f} |"
        )
    rows_stats = "\n".join(stats_lines)

    report = f"""# Stage 31 — Runtime Calibration Report

_Generated: {generated_at}_

## Command Used

```powershell
{cmd}
```

## Run Parameters

| Parameter | Value |
|-----------|-------|
| mode | {mode} |
| key_by | {key_by} |
| window_size | {window_size} |
| stride | {stride} |
| n_events | {n_events:,} |
| n_windows (per model) | {n_windows:,} |
| target_alert_rate | {target_alert_rate} |
| total elapsed (s) | {elapsed:.2f} |

## Calibration Method

**Chosen method: `{method}`**

{method_rationale}

## Calibrated Thresholds

| Model | Threshold | Method | Achieved Alert Rate |
|-------|-----------|--------|---------------------|
{rows_thr}

## Score Statistics

| Model | min | p50 | p95 | p99 | max |
|-------|-----|-----|-----|-----|-----|
{rows_stats}

## Notes

- These are **demo-calibrated thresholds**, not production calibration.
  Production calibration requires a representative held-out labeled dataset.
- Thresholds target `{target_alert_rate * 100:.2f}%` of streaming windows flagged as anomalies.
- To use calibrated thresholds in the runtime demo, pass `--use-runtime-thresholds`:

```powershell
python scripts/stage_05_runtime_demo.py --mode demo --model ensemble --use-runtime-thresholds
```

## Output Files

| File | Description |
|------|-------------|
| `artifacts/threshold_runtime.json` | Calibrated thresholds and score statistics |
| `reports/runtime_calibration_scores.csv` | Per-window scores (all models) |
| `reports/stage_31_runtime_calibration_report.md` | This report |
| `ai_workspace/logs/stage_05_runtime_calibrate.log` | Full execution log |
"""

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    log.info("Saved: %s", REPORT_MD_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 31 — Stream-based runtime threshold calibration"
    )
    parser.add_argument("--mode",   default="demo", choices=["demo", "full"],
                        help="Metadata label for the calibration artifact")
    parser.add_argument("--model",  default="ensemble",
                        choices=["baseline", "transformer", "ensemble"],
                        help="Primary model (calibration always runs all three)")
    parser.add_argument("--n-events", type=int, default=50_000,
                        help="Number of events to stream from parquet (default 50000)")
    parser.add_argument("--key-by", default="service",
                        choices=["service", "session"], dest="key_by",
                        help="Stream key granularity")
    parser.add_argument("--window-size", type=int, default=50, dest="window_size")
    parser.add_argument("--stride",      type=int, default=10)
    parser.add_argument("--target-alert-rate", type=float, default=0.005,
                        dest="target_alert_rate",
                        help="Target fraction of windows flagged (default 0.005 = 0.5%%)")
    args = parser.parse_args()

    run_calibration(
        mode=args.mode,
        model=args.model,
        n_events=args.n_events,
        key_by=args.key_by,
        window_size=args.window_size,
        stride=args.stride,
        target_alert_rate=args.target_alert_rate,
    )


if __name__ == "__main__":
    main()
