# scripts/stage_06_demo_alerts.py

# Purpose: Simulate live stream ingestion and fire alerts using the InferenceEngine and AlertManager.

# Input: Reads data/processed/events_tokenized.parquet (n_events rows) produced by stage_02_templates.py. 
# Each row is converted to an event dict and fed to the InferenceEngine. 
# RiskResults are processed by AlertManager and dispatched via N8nWebhookClient 
# in DRY_RUN mode (writes to outbox instead of making network calls).

# Output: Writes artifacts/n8n_outbox/<alert_id>.json (one file per fired alert) 
# and logs to ai_workspace/logs/stage_06_demo_alerts.log

# Used by: This is a standalone demo script that can be run directly to simulate the alerting pipeline. 
# It does not have downstream dependencies but relies on the output of 
# stage_02_templates.py to provide the input parquet file.

"""
Stage 06 -- Alerts Demo: simulate live stream ingestion and fire alerts.

Reads   : data/processed/events_tokenized.parquet (n_events rows)
Writes  : artifacts/n8n_outbox/<alert_id>.json  (one file per fired alert)
Logs to : ai_workspace/logs/stage_06_demo_alerts.log

Behaviour
---------
- Feeds tokenised events through InferenceEngine (ensemble by default).
- RiskResults are converted to Alerts via AlertPolicy + AlertManager.
- Alerts are dispatched via N8nWebhookClient in DRY_RUN mode (writes to outbox).
- No network calls are made unless N8N_DRY_RUN=false and N8N_WEBHOOK_URL is set.

Usage:
    python scripts/stage_06_demo_alerts.py
    python scripts/stage_06_demo_alerts.py --n-events 2000
    python scripts/stage_06_demo_alerts.py --n-events 2000 --model ensemble
    python scripts/stage_06_demo_alerts.py --n-events 2000 --cooldown 0
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.alerts import Alert, AlertManager, AlertPolicy, N8nWebhookClient
from src.runtime import InferenceEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "ai_workspace" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_path = LOG_DIR / "stage_06_demo_alerts.log"
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_h_file = logging.FileHandler(_log_path, encoding="utf-8")
_h_file.setFormatter(_fmt)
_h_stream = logging.StreamHandler(sys.stdout)
_h_stream.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_h_file, _h_stream])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVENTS_PARQUET = ROOT / "data" / "processed" / "events_tokenized.parquet"
OUTBOX_DIR     = ROOT / "artifacts" / "n8n_outbox"

DEMO_MAX_EVENTS = 2_000
WINDOW_SIZE     = 50
STRIDE          = 10


def _row_to_event(row: dict, key_by: str = "service") -> dict:
    """Convert a parquet row to the dict format InferenceEngine expects."""
    ts = row.get("timestamp")
    try:
        ts = float(ts) if ts is not None and ts == ts else 0.0
    except (TypeError, ValueError):
        ts = 0.0

    sid = "" if key_by == "service" else (row.get("session_id") or "")
    tok = row.get("token_id") or row.get("template_id") or 1

    return {
        "timestamp":  ts,
        "service":    row.get("service") or row.get("dataset") or "unknown",
        "session_id": sid,
        "token_id":   int(tok),
        "label":      row.get("label"),
    }


def run_demo(
    n_events: int,
    model: str,
    cooldown: float,
    key_by: str,
) -> dict:
    log.info("=== Stage 06 Alerts Demo ===")
    log.info("n_events=%d  model=%s  cooldown=%.0fs  key_by=%s",
             n_events, model, cooldown, key_by)

    if not EVENTS_PARQUET.exists():
        log.error("Input parquet not found: %s", EVENTS_PARQUET)
        log.error("Run stage_02_templates.py (--mode demo) first.")
        sys.exit(1)

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Load events
    # ------------------------------------------------------------------
    log.info("Loading %d events from %s ...", n_events, EVENTS_PARQUET)
    df = pd.read_parquet(EVENTS_PARQUET).head(n_events)
    log.info("Loaded %d rows", len(df))

    # ------------------------------------------------------------------
    # Initialise engine
    # ------------------------------------------------------------------
    log.info("Initialising InferenceEngine (mode=%s) ...", model)
    engine = InferenceEngine(
        mode=model,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    engine.load_artifacts()

    # ------------------------------------------------------------------
    # Initialise alert pipeline
    # ------------------------------------------------------------------
    policy  = AlertPolicy(cooldown_seconds=cooldown)
    manager = AlertManager(policy=policy)
    client  = N8nWebhookClient(outbox_dir=OUTBOX_DIR)

    log.info("Alert pipeline: cooldown=%.0fs  dry_run=%s  outbox=%s",
             cooldown, client.dry_run, OUTBOX_DIR)

    # ------------------------------------------------------------------
    # Stream events
    # ------------------------------------------------------------------
    events_processed = 0
    windows_emitted  = 0
    alerts_fired     = 0
    outbox_paths: list[str] = []

    for _, row in df.iterrows():
        event = _row_to_event(row.to_dict(), key_by=key_by)
        result = engine.ingest(event)
        events_processed += 1

        if result is not None:
            windows_emitted += 1

            for alert in manager.emit(result):
                alerts_fired += 1
                send_result = client.send(alert)
                if "path" in send_result:
                    outbox_paths.append(send_result["path"])
                log.info(
                    "[ALERT] %s | svc=%s | sev=%s | score=%.2f",
                    alert.alert_id[:8],
                    alert.service,
                    alert.severity,
                    alert.score,
                )

        if events_processed % 500 == 0:
            log.info("  %d / %d events | %d windows | %d alerts",
                     events_processed, n_events, windows_emitted, alerts_fired)

    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "events_processed":  events_processed,
        "windows_emitted":   windows_emitted,
        "alerts_fired":      alerts_fired,
        "alerts_suppressed": manager.suppressed_count,
        "outbox_files":      len(outbox_paths),
        "outbox_dir":        str(OUTBOX_DIR),
        "elapsed_s":         round(elapsed, 2),
        "events_per_sec":    round(events_processed / max(elapsed, 1e-3), 1),
    }

    log.info("=== Summary ===")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)
    log.info("=== Stage 06 Alerts Demo complete in %.2fs ===", elapsed)

    if outbox_paths:
        log.info("Sample outbox files:")
        for p in outbox_paths[:3]:
            log.info("  %s", p)
        if len(outbox_paths) > 3:
            log.info("  ... and %d more", len(outbox_paths) - 3)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 06 Alerts Demo")
    parser.add_argument("--n-events", dest="n_events", type=int,
                        default=DEMO_MAX_EVENTS,
                        help=f"Events to stream (default {DEMO_MAX_EVENTS})")
    parser.add_argument("--model", default="ensemble",
                        choices=["baseline", "transformer", "ensemble"],
                        help="Scoring model (default: ensemble)")
    parser.add_argument("--cooldown", type=float, default=60.0,
                        help="Alert cooldown in seconds per stream key (default 60)")
    parser.add_argument("--key-by", dest="key_by", default="service",
                        choices=["service", "session"],
                        help="Stream key mode (default: service)")
    args = parser.parse_args()

    run_demo(
        n_events=args.n_events,
        model=args.model,
        cooldown=args.cooldown,
        key_by=args.key_by,
    )


if __name__ == "__main__":
    main()
