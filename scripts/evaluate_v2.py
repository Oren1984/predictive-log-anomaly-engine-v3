#!/usr/bin/env python
# scripts/evaluate_v2.py
#
# Phase 8 — V1 vs V2 Pipeline Evaluation
#
# Replays labeled HDFS events through both inference pipelines and computes
# precision, recall, F1, false-positive rate, and average latency per call.
#
# Ground truth source:
#   data/raw/HDFS_1/anomaly_label.csv   (BlockId → Normal | Anomaly)
#
# Event source:
#   data/processed/events_tokenized.parquet
#
# Template lookup (v2 proxy-log construction):
#   data/intermediate/templates.csv
#
# Strategy
# --------
# Both pipelines work on per-session rolling windows.  Since the raw
# HDFS.log is ~1.5 GB, we avoid re-parsing it by using events that were
# already tokenized during preprocessing:
#
#   V1  — feed each event as a dict {service, session_id, token_id, timestamp}
#          through InferenceEngine.ingest().
#
#   V2  — reconstruct a "proxy" raw-log string from the template_text for each
#          event's token_id, then feed through V2Pipeline.process_log().
#          Because template_text is already the output of TemplateMiner's
#          _generalize() step, passing it back through _V2LogTokenizer's
#          identical substitution pipeline is idempotent — the lookup
#          produces the same token_id as during original preprocessing.
#
# Session-level prediction: a session is flagged as anomalous if at least
# one emitted window is flagged is_anomaly=True (logical-OR aggregation).
#
# Usage:
#   python scripts/evaluate_v2.py
#   python scripts/evaluate_v2.py --max-sessions 2000 --window-size 10
#   python scripts/evaluate_v2.py --output results/my_eval.json
#
# Environment overrides:
#   EVAL_MAX_SESSIONS   integer (default 2000)
#   EVAL_WINDOW_SIZE    integer (default 5)
#   EVAL_V1_MODE        baseline | transformer | ensemble (default baseline)
#   EVAL_OUTPUT         path for JSON report (default evaluation_report.json)
#
# Note on window size
# -------------------
# The HDFS dataset in events_tokenized.parquet has very short sessions;
# most blocks have 1-5 events.  The default window_size of 5 balances
# evaluation coverage (≈5 000 eligible sessions) against model fidelity.
# Use --window-size 10 to match the v2 production config (≈42 eligible
# sessions — acceptable for a quick smoke-level comparison).

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running as `python scripts/evaluate_v2.py`
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,          # suppress model-load chatter
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Keep our own logger at INFO for progress updates
logger = logging.getLogger("evaluate_v2")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_LABEL_CSV = _PROJECT_ROOT / "data" / "raw" / "HDFS_1" / "anomaly_label.csv"
_EVENTS_PARQUET = _PROJECT_ROOT / "data" / "processed" / "events_tokenized.parquet"
_TEMPLATES_CSV = _PROJECT_ROOT / "data" / "intermediate" / "templates.csv"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "evaluation_report.json"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ground_truth(label_csv: Path) -> Dict[str, int]:
    """Return {block_id: 0 | 1} from anomaly_label.csv."""
    import pandas as pd

    df = pd.read_csv(label_csv)
    # Columns: BlockId, Label  (values: "Normal" | "Anomaly")
    gt: Dict[str, int] = {}
    for _, row in df.iterrows():
        gt[str(row["BlockId"])] = 0 if str(row["Label"]).strip() == "Normal" else 1
    logger.info(
        "Ground truth loaded: %d sessions (%d anomalous)",
        len(gt), sum(gt.values()),
    )
    return gt


def load_events(events_parquet: Path) -> "pd.DataFrame":
    """Load events_tokenized.parquet and return the full DataFrame."""
    import pandas as pd

    df = pd.read_parquet(events_parquet)
    logger.info(
        "Events loaded: %d rows, %d unique sessions",
        len(df), df["session_id"].nunique(),
    )
    return df


def load_template_map(templates_csv: Path) -> Dict[int, str]:
    """Return {token_id: template_text} using token_id = template_id + 2."""
    import pandas as pd

    df = pd.read_csv(templates_csv, usecols=["template_id", "template_text"])
    return {
        int(row["template_id"]) + 2: str(row["template_text"])
        for _, row in df.iterrows()
    }


def select_sessions(
    df: "pd.DataFrame",
    gt: Dict[str, int],
    window_size: int,
    max_sessions: int,
) -> List[str]:
    """
    Return up to *max_sessions* session_ids that:
      - have ≥ window_size events (so at least one window will be emitted)
      - appear in the ground-truth label map
    Attempts a balanced mix of normal and anomalous sessions.
    """
    # Only HDFS events have block-level ground truth
    hdfs = df[df["service"] == "hdfs"] if "service" in df.columns else df

    counts = hdfs.groupby("session_id").size()
    eligible = counts[counts >= window_size].index.intersection(list(gt.keys()))

    normal_ids = [s for s in eligible if gt[s] == 0]
    anom_ids   = [s for s in eligible if gt[s] == 1]

    half = max_sessions // 2
    selected = anom_ids[:half] + normal_ids[:half]
    if len(selected) < max_sessions:
        # Top-up with whatever is available
        extra = [s for s in eligible if s not in set(selected)]
        selected += extra[: max_sessions - len(selected)]

    logger.info(
        "Selected %d sessions for evaluation "
        "(%d anomalous, %d normal, window_size≥%d)",
        len(selected),
        sum(gt[s] == 1 for s in selected),
        sum(gt[s] == 0 for s in selected),
        window_size,
    )
    return selected


# ---------------------------------------------------------------------------
# V1 evaluation
# ---------------------------------------------------------------------------

def evaluate_v1(
    session_ids: List[str],
    df: "pd.DataFrame",
    window_size: int,
    mode: str,
) -> Tuple[Dict[str, Optional[bool]], float]:
    """
    Feed events through the v1 InferenceEngine.

    Returns
    -------
    predictions : {session_id: True|False|None}
        None when the session had no windows emitted (< window_size events
        survived after filtering).
    avg_latency_ms : average time per ingest() call in milliseconds.
    """
    from src.runtime.inference_engine import InferenceEngine

    engine = InferenceEngine(
        mode=mode,
        window_size=window_size,
        stride=1,           # emit on every event once window is full
    )
    engine.load_artifacts()

    hdfs = df[df["service"] == "hdfs"] if "service" in df.columns else df
    session_set = set(session_ids)
    subset = hdfs[hdfs["session_id"].isin(session_set)].copy()

    # Sort by session then (approximate) order for deterministic replay
    if "timestamp" in subset.columns:
        subset = subset.sort_values(["session_id", "timestamp"])
    else:
        subset = subset.sort_values("session_id")

    predictions: Dict[str, Optional[bool]] = {s: None for s in session_ids}
    total_time = 0.0
    n_calls = 0

    for row in subset.itertuples(index=False):
        sid = str(row.session_id)
        event = {
            "service": "hdfs",
            "session_id": sid,
            "token_id": int(row.token_id),
            "timestamp": float(row.timestamp) if row.timestamp == row.timestamp else 0.0,
        }

        t0 = time.perf_counter()
        result = engine.ingest(event)
        total_time += time.perf_counter() - t0
        n_calls += 1

        if result is not None and result.is_anomaly:
            predictions[sid] = True
        elif result is not None and predictions[sid] is None:
            predictions[sid] = False

    # Sessions with at least one window but never flagged anomalous → False
    for sid in session_ids:
        if predictions[sid] is None:
            # Check if any window was emitted at all
            pass  # leave as None; caller handles this

    avg_latency_ms = (total_time / n_calls * 1000) if n_calls > 0 else 0.0
    return predictions, avg_latency_ms


# ---------------------------------------------------------------------------
# V2 evaluation
# ---------------------------------------------------------------------------

def evaluate_v2(
    session_ids: List[str],
    df: "pd.DataFrame",
    template_map: Dict[int, str],
    window_size: int,
) -> Tuple[Dict[str, Optional[bool]], float]:
    """
    Feed events through the v2 V2Pipeline using template_text as proxy raw-log.

    Returns
    -------
    predictions : {session_id: True|False|None}
    avg_latency_ms : average time per process_log() call in milliseconds.
    """
    from src.runtime.pipeline_v2 import V2Pipeline, V2PipelineConfig

    cfg = V2PipelineConfig(window_size=window_size)
    pipeline = V2Pipeline(cfg)
    pipeline.load_models()

    hdfs = df[df["service"] == "hdfs"] if "service" in df.columns else df
    session_set = set(session_ids)
    subset = hdfs[hdfs["session_id"].isin(session_set)].copy()

    if "timestamp" in subset.columns:
        subset = subset.sort_values(["session_id", "timestamp"])
    else:
        subset = subset.sort_values("session_id")

    _UNK_LOG = "UNKNOWN"   # fallback proxy log for token_ids absent from template_map

    predictions: Dict[str, Optional[bool]] = {s: None for s in session_ids}
    total_time = 0.0
    n_calls = 0

    for row in subset.itertuples(index=False):
        sid = str(row.session_id)
        token_id = int(row.token_id)
        raw_log = template_map.get(token_id, _UNK_LOG)
        ts = float(row.timestamp) if row.timestamp == row.timestamp else 0.0

        t0 = time.perf_counter()
        result = pipeline.process_log(
            raw_log=raw_log,
            service="hdfs",
            session_id=sid,
            timestamp=ts,
        )
        total_time += time.perf_counter() - t0
        n_calls += 1

        if result.window_emitted:
            if result.is_anomaly:
                predictions[sid] = True
            elif predictions[sid] is None:
                predictions[sid] = False

    avg_latency_ms = (total_time / n_calls * 1000) if n_calls > 0 else 0.0
    return predictions, avg_latency_ms


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: Dict[str, Optional[bool]],
    gt: Dict[str, int],
) -> dict:
    """
    Compute binary classification metrics against ground truth.

    Sessions where no window was emitted (predictions[sid] is None) are
    counted as predicted-normal (False) since the model produced no evidence
    of anomaly — this is the conservative default.
    """
    tp = fp = tn = fn = 0
    skipped = 0

    for sid, pred in predictions.items():
        true_label = gt.get(sid)
        if true_label is None:
            skipped += 1
            continue

        predicted_anomaly = bool(pred)   # None → False

        if predicted_anomaly and true_label == 1:
            tp += 1
        elif predicted_anomaly and true_label == 0:
            fp += 1
        elif not predicted_anomaly and true_label == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false positive rate

    return {
        "total_sessions": total,
        "skipped": skipped,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       4),
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(
    v1_metrics: dict,
    v2_metrics: dict,
    v1_latency_ms: float,
    v2_latency_ms: float,
    n_sessions: int,
    window_size: int,
    v1_mode: str,
) -> None:
    sep = "=" * 62

    print()
    print(sep)
    print("  Phase 8 - V1 vs V2 Pipeline Evaluation")
    print(sep)
    print(f"  Sessions evaluated : {n_sessions}")
    print(f"  Window size        : {window_size}")
    print(f"  V1 mode            : {v1_mode}")
    print(sep)
    print(f"  {'Metric':<22}  {'V1':>10}  {'V2':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")

    def row(label, key):
        v1_val = v1_metrics.get(key, "—")
        v2_val = v2_metrics.get(key, "—")
        if isinstance(v1_val, float):
            v1_str = f"{v1_val:.4f}"
        else:
            v1_str = str(v1_val)
        if isinstance(v2_val, float):
            v2_str = f"{v2_val:.4f}"
        else:
            v2_str = str(v2_val)
        print(f"  {label:<22}  {v1_str:>10}  {v2_str:>10}")

    row("Precision",         "precision")
    row("Recall",            "recall")
    row("F1 Score",          "f1")
    row("False Positive Rate", "fpr")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    row("TP", "TP")
    row("FP", "FP")
    row("TN", "TN")
    row("FN", "FN")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    print(f"  {'Avg latency/call':<22}  {v1_latency_ms:>9.3f}ms  {v2_latency_ms:>9.3f}ms")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def write_report(
    output_path: Path,
    v1_metrics: dict,
    v2_metrics: dict,
    v1_latency_ms: float,
    v2_latency_ms: float,
    n_sessions: int,
    window_size: int,
    v1_mode: str,
) -> None:
    report = {
        "evaluation_config": {
            "sessions_evaluated": n_sessions,
            "window_size": window_size,
            "v1_mode": v1_mode,
            "ground_truth_source": str(_LABEL_CSV),
            "events_source": str(_EVENTS_PARQUET),
        },
        "v1": {
            **v1_metrics,
            "avg_latency_ms": round(v1_latency_ms, 4),
        },
        "v2": {
            **v2_metrics,
            "avg_latency_ms": round(v2_latency_ms, 4),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Report written to: {output_path}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare v1 and v2 inference pipelines on labeled HDFS data."
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=int(os.environ.get("EVAL_MAX_SESSIONS", "2000")),
        help="Maximum number of sessions to evaluate (default: 2000)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.environ.get("EVAL_WINDOW_SIZE", "5")),
        help=(
            "Rolling window size used by both pipelines (default: 5). "
            "The HDFS dataset has very short sessions; window=5 gives ~5 000 "
            "eligible sessions while window=10 gives only ~42."
        ),
    )
    parser.add_argument(
        "--v1-mode",
        choices=("baseline", "transformer", "ensemble"),
        default=os.environ.get("EVAL_V1_MODE", "baseline"),
        help="V1 InferenceEngine scoring mode (default: baseline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("EVAL_OUTPUT", str(_DEFAULT_OUTPUT))),
        help=f"Path for JSON evaluation report (default: {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for path, desc in [
        (_LABEL_CSV,     "anomaly_label.csv"),
        (_EVENTS_PARQUET,"events_tokenized.parquet"),
        (_TEMPLATES_CSV, "templates.csv"),
    ]:
        if not path.exists():
            logger.error("Required file not found: %s (%s)", path, desc)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading ground truth labels ...")
    gt = load_ground_truth(_LABEL_CSV)

    logger.info("Loading tokenized events ...")
    df = load_events(_EVENTS_PARQUET)

    logger.info("Loading template map for v2 proxy logs ...")
    template_map = load_template_map(_TEMPLATES_CSV)
    logger.info("Template map: %d entries", len(template_map))

    # ------------------------------------------------------------------
    # Session selection
    # ------------------------------------------------------------------
    session_ids = select_sessions(df, gt, args.window_size, args.max_sessions)
    if not session_ids:
        logger.error(
            "No eligible sessions found. "
            "Try reducing --window-size or check that events_tokenized.parquet "
            "contains HDFS events with session_ids matching anomaly_label.csv."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # V1 evaluation
    # ------------------------------------------------------------------
    print(f"\n[1/2] Running V1 pipeline (mode={args.v1_mode}) on {len(session_ids)} sessions ...")
    v1_preds, v1_latency = evaluate_v1(session_ids, df, args.window_size, args.v1_mode)
    v1_metrics = compute_metrics(v1_preds, gt)
    print(
        f"      Done. TP={v1_metrics['TP']} FP={v1_metrics['FP']} "
        f"TN={v1_metrics['TN']} FN={v1_metrics['FN']}  "
        f"latency={v1_latency:.3f}ms/call"
    )

    # ------------------------------------------------------------------
    # V2 evaluation
    # ------------------------------------------------------------------
    print(f"[2/2] Running V2 pipeline on {len(session_ids)} sessions ...")
    v2_preds, v2_latency = evaluate_v2(session_ids, df, template_map, args.window_size)
    v2_metrics = compute_metrics(v2_preds, gt)
    print(
        f"      Done. TP={v2_metrics['TP']} FP={v2_metrics['FP']} "
        f"TN={v2_metrics['TN']} FN={v2_metrics['FN']}  "
        f"latency={v2_latency:.3f}ms/call"
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print_summary(
        v1_metrics, v2_metrics,
        v1_latency, v2_latency,
        len(session_ids), args.window_size, args.v1_mode,
    )
    write_report(
        args.output,
        v1_metrics, v2_metrics,
        v1_latency, v2_latency,
        len(session_ids), args.window_size, args.v1_mode,
    )


if __name__ == "__main__":
    main()
