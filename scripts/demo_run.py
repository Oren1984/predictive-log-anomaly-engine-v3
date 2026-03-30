"""
scripts/demo_run.py -- Optional local end-to-end demo runner (Stage 7.2).

Usage:
    python scripts/demo_run.py
    python scripts/demo_run.py --events 100

What it does (all in-process, no running server required):
    1. Generates synthetic log events using SyntheticLogGenerator
    2. Builds an in-process FastAPI app with MockPipeline (no model files needed)
    3. POSTs all events to POST /ingest
    4. GETs alerts from GET /alerts
    5. POSTs 3 questions to POST /query (RAG stub)
    6. Prints a readable console summary

Requirements: only packages already in requirements.txt + requirements-dev.txt.
Does NOT download any data or models.  Runs in < 30 seconds.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import Settings
from src.synthetic import (
    AuthBruteForcePattern,
    MemoryLeakPattern,
    NetworkFlapPattern,
    ScenarioBuilder,
    SyntheticLogGenerator,
)
from tests.helpers_stage_07 import MockPipeline, _stub_risk_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TS = 1_704_067_200.0
_SEP = "-" * 60

_DEMO_QUESTIONS = [
    "How does the alert threshold work?",
    "What model is used for anomaly detection?",
    "How do I run the stack with Docker?",
]


# ---------------------------------------------------------------------------
# Build in-process client
# ---------------------------------------------------------------------------

def _build_client() -> tuple[TestClient, MockPipeline]:
    """
    Create an in-process FastAPI app with auth disabled and a MockPipeline
    pre-configured for demo mode (small window, zero cooldown, anomaly result).
    """
    cfg = Settings()
    cfg.disable_auth = True
    cfg.metrics_enabled = False
    cfg.window_size = 5
    cfg.stride = 1
    cfg.alert_cooldown_seconds = 0.0

    pipeline = MockPipeline(settings=cfg)
    # Prime the engine so every window fires a critical alert
    pipeline.engine.next_result = _stub_risk_result(
        score=3.0, is_anomaly=True, threshold=1.0
    )

    app = create_app(settings=cfg, pipeline=pipeline)
    return TestClient(app, raise_server_exceptions=True), pipeline


# ---------------------------------------------------------------------------
# Synthetic event generation
# ---------------------------------------------------------------------------

def _generate_events(n: int) -> list[dict]:
    """
    Generate n synthetic log events from three patterns across three services.
    No external data files required.
    """
    patterns = [MemoryLeakPattern(), NetworkFlapPattern(), AuthBruteForcePattern()]
    services = ["bgl", "hdfs", "auth"]
    pattern_names = ["memory_leak", "network_flap", "auth_brute_force"]

    gen = SyntheticLogGenerator(patterns, seed=42)
    builder = ScenarioBuilder()

    # Distribute n events across 3 scenarios
    counts = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    counts = [max(c, 1) for c in counts]  # each scenario needs >= 1 event

    scenarios = [
        builder.build_scenario(
            scenario_id=f"demo-{i}",
            service=services[i],
            host=f"host-{i+1:02d}",
            start_ts=_BASE_TS + i * 3600,
            n_events=counts[i],
            pattern_name=pattern_names[i],
        )
        for i in range(3)
    ]

    raw = gen.generate_all(scenarios)[:n]
    return [
        {
            "service": ev.service,
            "token_id": abs(hash(ev.message)) % 7833 + 2,
            "session_id": f"{ev.service}-demo",
            "timestamp": float(ev.timestamp or 0),
            "label": int(ev.label or 0),
        }
        for ev in raw
    ]


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------

def _step_ingest(client: TestClient, events: list[dict]) -> dict:
    """POST all events to /ingest; return summary stats."""
    print(f"\n[1/3] INGEST  -- {len(events)} synthetic events")

    t0 = time.perf_counter()
    windows = alerts_fired = errors = 0

    for ev in events:
        try:
            resp = client.post("/ingest", json=ev)
            if resp.status_code == 200:
                body = resp.json()
                if body.get("window_emitted"):
                    windows += 1
                if body.get("alert"):
                    alerts_fired += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    elapsed = time.perf_counter() - t0
    rate = len(events) / elapsed if elapsed > 0 else 0

    print(f"  Events ingested : {len(events)}")
    print(f"  Windows emitted : {windows}")
    print(f"  Alerts fired    : {alerts_fired}")
    print(f"  Errors          : {errors}")
    print(f"  Throughput      : {rate:.0f} events/s  ({elapsed*1000:.0f} ms total)")

    return {"windows": windows, "alerts_fired": alerts_fired, "errors": errors}


def _step_alerts(client: TestClient) -> list[dict]:
    """GET /alerts and render a summary table."""
    print("\n[2/3] ALERTS  -- GET /alerts")

    resp = client.get("/alerts")
    if resp.status_code != 200:
        print(f"  ERROR: HTTP {resp.status_code}")
        return []

    data = resp.json()
    alerts = data.get("alerts", [])
    print(f"  Ring buffer count: {data.get('count', 0)}")

    if not alerts:
        print("  (no alerts in buffer)")
    else:
        print(f"  {'Severity':<10} {'Service':<8} {'Score':>7}  Timestamp")
        print(f"  {'-'*10} {'-'*8} {'-'*7}  {'-'*20}")
        for a in alerts[:8]:
            sev = a.get("severity", "?").upper()
            svc = a.get("service", "?")
            score = f"{a.get('score', 0):.3f}"
            ts = a.get("timestamp", 0)
            try:
                from datetime import datetime, timezone
                ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_str = str(ts)
            print(f"  {sev:<10} {svc:<8} {score:>7}  {ts_str}")
        if len(alerts) > 8:
            print(f"  ... and {len(alerts) - 8} more")

    return alerts


def _step_rag(client: TestClient, questions: list[str]) -> None:
    """POST each question to /query and print answers."""
    print("\n[3/3] RAG ASK -- POST /query")

    for i, q in enumerate(questions, 1):
        resp = client.post("/query", json={"question": q})
        if resp.status_code != 200:
            print(f"\n  Q{i}: {q}")
            print(f"  ERROR: HTTP {resp.status_code}")
            continue
        data = resp.json()
        answer = data.get("answer", "")
        sources = [s["id"] for s in data.get("sources", [])[:3]]

        print(f"\n  Q{i}: {q}")
        # Wrap answer at 70 chars
        words = answer.split()
        line, lines = [], []
        for w in words:
            if len(" ".join(line + [w])) > 70:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(f"  A:  {l}")
        if sources:
            print(f"  Sources: {', '.join(sources)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(n_events: int = 75) -> None:
    print(_SEP)
    print("  Predictive Log Anomaly Engine - Demo Runner  (Stage 7.2)")
    print(_SEP)
    print(f"  Generating {n_events} synthetic events (no data files needed)...")

    t_total = time.perf_counter()

    events = _generate_events(n_events)
    client, _ = _build_client()

    ingest = _step_ingest(client, events)
    alerts = _step_alerts(client)
    _step_rag(client, _DEMO_QUESTIONS)

    elapsed = time.perf_counter() - t_total

    print(f"\n{_SEP}")
    print(f"  Demo complete in {elapsed:.1f}s")
    print(
        f"  events={n_events}  "
        f"windows={ingest['windows']}  "
        f"alerts={ingest['alerts_fired']}  "
        f"errors={ingest['errors']}"
    )
    print(_SEP)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predictive Log Anomaly Engine — in-process demo runner"
    )
    parser.add_argument(
        "--events", type=int, default=75,
        help="Number of synthetic events to ingest (default: 75)",
    )
    args = parser.parse_args()
    run_demo(n_events=args.events)


if __name__ == "__main__":
    main()
