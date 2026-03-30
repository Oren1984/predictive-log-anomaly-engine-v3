# scripts/stage_01_synth_generate.py

# Purpose: This script generates synthetic log events for testing and demonstration purposes.
# It creates multiple scenarios with different patterns of normal and anomalous behavior,
# and outputs the generated events in both CSV and Parquet formats,
# along with a schema description and a summary report.

# Input: Command-line arguments specifying the mode, number of events, services, output paths, and random seed.

# Output: Generated synthetic log events in CSV and Parquet formats, schema description, scenario definitions, and a summary report.

# Used by: This script can be run independently to generate synthetic data for testing the anomaly detection pipeline.
# It is not directly used by other scripts but provides a synthetic dataset
# that can be used in place of real data for stages 02 and beyond.

"""
Stage 01 — Synthetic: generate synthetic log events.

Builds 4 single-pattern scenarios + 1 hybrid scenario and writes:
  data/synth/events_synth.csv
  data/synth/events_synth.parquet   (canonical schema: timestamp,service,level,message,meta,label)
  data/synth/schema.md
  data/synth/scenarios.json
  reports/stage_01_synth_report.md

Logs to: ai_workspace/logs/stage_01_generate_synth.log

Legacy usage (unchanged):
    python scripts/stage_01_synth_generate.py --mode demo
    python scripts/stage_01_synth_generate.py --mode full --events 200000
    python scripts/stage_01_synth_generate.py --mode demo --seed 123

New usage (adds --n-events, --services, --out, --schema-out):
    python scripts/stage_01_synth_generate.py --n-events 50000
    python scripts/stage_01_synth_generate.py --n-events 50000 --services auth,api,billing,db
    python scripts/stage_01_synth_generate.py --n-events 50000 --out data/synth/events_synth.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.synthetic import (
    AuthBruteForcePattern,
    DiskFullPattern,
    MemoryLeakPattern,
    NetworkFlapPattern,
    ScenarioBuilder,
    SyntheticLogGenerator,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR  = ROOT / "ai_workspace" / "logs"
OUT_DIR  = ROOT / "data" / "synth"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

_log_path = LOG_DIR / "stage_01_generate_synth.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Fixed epoch for reproducible timestamps (2024-01-01 00:00:00 UTC)
_BASE_TS = 1_704_067_200.0

# Phase split used by all scenarios
_DEFAULT_PHASES = {"normal": 0.60, "degradation": 0.30, "failure": 0.10}

# Default service names (one per single-pattern scenario)
_DEFAULT_SERVICES = ["app-server", "storage", "auth-service", "network"]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _build_scenarios(n_total: int, seed: int, services: list[str] | None = None) -> list[dict]:
    """Return a list of 5 scenario dicts (4 single-pattern + 1 hybrid).

    Parameters
    ----------
    n_total  : total events across all scenarios
    seed     : random seed (unused here; kept for API symmetry)
    services : list of service names for the 4 single-pattern scenarios.
               Cycled if fewer than 4 are given.  Defaults to legacy names.
    """
    builder = ScenarioBuilder()

    # Distribute events: 4 equal single-pattern + hybrid gets remainder
    n_each   = n_total // 5
    n_hybrid = n_total - 4 * n_each

    STEP = 3600  # seconds between scenario start times (1 hour apart)

    # Resolve service names
    svc_list = services if services else _DEFAULT_SERVICES
    svc = [svc_list[i % len(svc_list)] for i in range(4)]

    scenarios = [
        builder.build_scenario(
            scenario_id  = "mem_leak_001",
            service      = svc[0],
            host         = "host-01",
            start_ts     = _BASE_TS,
            n_events     = n_each,
            phases       = dict(_DEFAULT_PHASES),
            pattern_name = "memory_leak",
        ),
        builder.build_scenario(
            scenario_id  = "disk_full_001",
            service      = svc[1],
            host         = "host-02",
            start_ts     = _BASE_TS + STEP,
            n_events     = n_each,
            phases       = dict(_DEFAULT_PHASES),
            pattern_name = "disk_full",
        ),
        builder.build_scenario(
            scenario_id  = "auth_brute_001",
            service      = svc[2],
            host         = "host-03",
            start_ts     = _BASE_TS + 2 * STEP,
            n_events     = n_each,
            phases       = dict(_DEFAULT_PHASES),
            pattern_name = "auth_brute_force",
        ),
        builder.build_scenario(
            scenario_id  = "net_flap_001",
            service      = svc[3],
            host         = "host-04",
            start_ts     = _BASE_TS + 3 * STEP,
            n_events     = n_each,
            phases       = dict(_DEFAULT_PHASES),
            pattern_name = "network_flap",
        ),
        builder.build_hybrid_scenario(
            scenario_id   = "hybrid_001",
            service       = "hybrid-svc",
            host          = "host-05",
            start_ts      = _BASE_TS + 4 * STEP,
            n_events      = n_hybrid,
            pattern_names = [
                "memory_leak",
                "disk_full",
                "auth_brute_force",
                "network_flap",
            ],
            phases        = {"normal": 0.50, "degradation": 0.35, "failure": 0.15},
        ),
    ]
    return scenarios


# ---------------------------------------------------------------------------
# Canonical schema helpers
# ---------------------------------------------------------------------------

def _events_to_canonical_df(events: list) -> pd.DataFrame:
    """Convert events to canonical parquet schema.

    Columns: timestamp, service, level, message, meta (JSON str), label.
    Sorted by timestamp (strictly increasing).
    """
    rows = []
    for ev in events:
        meta = ev.meta or {}
        rows.append({
            "timestamp": float(ev.timestamp) if ev.timestamp is not None else 0.0,
            "service":   ev.service,
            "level":     ev.level,
            "message":   ev.message,
            "meta":      json.dumps(meta),
            "label":     int(ev.label) if ev.label is not None else 0,
        })
    df = pd.DataFrame(rows, columns=["timestamp", "service", "level", "message", "meta", "label"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _write_schema_md(schema_out: Path) -> None:
    """Write schema.md describing the canonical parquet schema."""
    schema_out.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Synthetic Log Events Schema\n\n"
        "## File: events_synth.parquet\n\n"
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| timestamp | float64 | Unix timestamp (seconds since epoch, UTC). Strictly increasing. |\n"
        "| service | str | Service name generating the event (e.g. auth, api, billing, db). |\n"
        "| level | str | Log severity level: `INFO`, `WARNING`, or `ERROR`. |\n"
        "| message | str | Human-readable log message text. |\n"
        "| meta | str | JSON-encoded metadata dict (host, component, phase, scenario_id, session_id). |\n"
        "| label | int64 | Anomaly label: `0` = normal, `1` = anomalous (degradation or failure). |\n\n"
        "## Label Meanings\n\n"
        "| label | Meaning | Phase |\n"
        "|-------|---------|-------|\n"
        "| 0 | Normal -- service operating as expected | normal |\n"
        "| 1 | Anomalous -- service degrading or failed | degradation, failure |\n\n"
        "## Meta Keys (inside JSON string)\n\n"
        "| Key | Description |\n"
        "|-----|-------------|\n"
        "| host | Hostname generating the event |\n"
        "| component | Sub-component within the service |\n"
        "| scenario_id | Scenario identifier |\n"
        "| phase | One of `normal`, `degradation`, `failure` |\n"
        "| session_id | Sliding-window session ID (50 events per window) |\n\n"
        "## Pattern Types\n\n"
        "| Pattern | Failure Signature |\n"
        "|---------|-------------------|\n"
        "| memory_leak | heap grows until OOM-kill |\n"
        "| disk_full | /var/data fills until ENOSPC |\n"
        "| auth_brute_force | login failures escalate to lockouts |\n"
        "| network_flap | eth0 latency/loss until interface down |\n"
        "| hybrid | all four patterns interleaved |\n"
    )
    schema_out.write_text(content, encoding="utf-8")


def _write_synth_report(
    df_canonical: pd.DataFrame,
    n_events: int,
    services: list[str],
    seed: int,
    out_path: Path,
    schema_out: Path,
    elapsed: float,
    report_path: Path,
) -> None:
    """Write 1-page report to reports/stage_01_synth_report.md."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts = df_canonical["label"].value_counts().sort_index().to_dict()
    n_normal  = label_counts.get(0, 0)
    n_anomaly = label_counts.get(1, 0)
    n_total   = len(df_canonical)

    cmd = (
        f"python scripts/stage_01_synth_generate.py "
        f"--n-events {n_events} --services {','.join(services)} --seed {seed}"
    )

    # Sample 5 rows
    sample = df_canonical.head(5)[["timestamp", "service", "level", "message", "label"]]
    sample_lines = [
        "| timestamp | service | level | message | label |",
        "|-----------|---------|-------|---------|-------|",
    ]
    for _, row in sample.iterrows():
        msg = str(row["message"])[:60].replace("|", "\\|")
        sample_lines.append(
            f"| {row['timestamp']:.0f} | {row['service']} | {row['level']} | {msg} | {row['label']} |"
        )

    lines = [
        "# Stage 01 -- Synthetic Data Generation Report",
        "",
        "> **synthetic demo generator for requirement coverage**",
        "",
        f"**Command:** `{cmd}`  ",
        f"**Events generated:** {n_total:,}  ",
        f"**Services:** {', '.join(services)}  ",
        f"**Seed:** {seed}  ",
        f"**Elapsed:** {elapsed:.2f}s  ",
        f"**Output:** `{out_path}`  ",
        f"**Schema:** `{schema_out}`  ",
        "",
        "## Label Distribution",
        "",
        "| label | meaning | count | pct |",
        "|-------|---------|------:|----:|",
        f"| 0 | normal | {n_normal:,} | {n_normal / max(n_total, 1) * 100:.1f}% |",
        f"| 1 | anomaly | {n_anomaly:,} | {n_anomaly / max(n_total, 1) * 100:.1f}% |",
        f"| **total** | | {n_total:,} | 100.0% |",
        "",
        "## Sample Rows (first 5)",
        "",
    ] + sample_lines + [
        "",
        "## Schema",
        "",
        "Columns: `timestamp`, `service`, `level`, `message`, `meta` (JSON string), `label`  ",
        f"Full schema: `{schema_out}`  ",
        "",
        "---",
        "",
        "_Generated by stage_01_synth_generate.py -- synthetic demo generator for requirement coverage_",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def run_generate(
    mode: str,
    seed: int,
    n_events: int,
    outdir: Path,
    services: list[str] | None = None,
    out_path: Path | None = None,
    schema_out: Path | None = None,
) -> dict:
    log.info("=== Stage 01 Synthetic Generate ===")
    log.info("mode=%s  seed=%d  n_events=%d  outdir=%s", mode, seed, n_events, outdir)
    if services:
        log.info("services=%s", services)

    t_start = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve output paths
    if out_path is None:
        out_path = outdir / "events_synth.parquet"
    if schema_out is None:
        schema_out = outdir / "schema.md"
    report_path = ROOT / "reports" / "stage_01_synth_report.md"

    # ------------------------------------------------------------------
    # Build scenarios and generator
    # ------------------------------------------------------------------
    scenarios = _build_scenarios(n_events, seed, services=services)
    log.info("Built %d scenarios", len(scenarios))
    for sc in scenarios:
        log.info("  %s: pattern=%s  n=%d  anomaly_rate=%.1f%%",
                 sc["scenario_id"],
                 sc.get("pattern_name", sc.get("pattern_names")),
                 sc["n_events"],
                 sc["anomaly_rate"] * 100)

    generator = SyntheticLogGenerator(
        patterns=[
            MemoryLeakPattern(),
            DiskFullPattern(),
            AuthBruteForcePattern(),
            NetworkFlapPattern(),
        ],
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Generate events
    # ------------------------------------------------------------------
    log.info("Generating events ...")
    events = generator.generate_all(scenarios)
    log.info("Generated %d total events", len(events))

    df = SyntheticLogGenerator.events_to_dataframe(events)
    log.info("DataFrame shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Label / phase statistics
    # ------------------------------------------------------------------
    label_counts = df["label"].value_counts().sort_index().to_dict()
    phase_counts = df["phase"].value_counts().to_dict()
    anomaly_rate = float(df["label"].mean() * 100)

    log.info("Label distribution: %s", label_counts)
    log.info("Phase distribution:  %s", phase_counts)
    log.info("Anomaly rate: %.2f%%", anomaly_rate)

    # ------------------------------------------------------------------
    # Write legacy outputs (CSV + full-schema parquet)
    # ------------------------------------------------------------------
    csv_path = outdir / "events_synth.csv"
    df.to_csv(csv_path, index=False)
    log.info("CSV written: %s (%d rows)", csv_path, len(df))

    try:
        df.to_parquet(outdir / "events_synth_full.parquet", index=False)
    except Exception as exc:
        log.warning("Full-schema parquet write failed (non-fatal): %s", exc)

    # Scenarios JSON (strip non-serialisable fields for clean output)
    scen_path = outdir / "scenarios.json"
    sc_records = []
    for sc in scenarios:
        rec = {k: v for k, v in sc.items() if k != "rng"}
        sc_records.append(rec)
    with open(scen_path, "w", encoding="utf-8") as fh:
        json.dump(sc_records, fh, indent=2)
    log.info("Scenarios JSON written: %s", scen_path)

    # ------------------------------------------------------------------
    # Write canonical parquet (timestamp, service, level, message, meta, label)
    # ------------------------------------------------------------------
    df_canonical = _events_to_canonical_df(events)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_canonical.to_parquet(out_path, index=False)
    log.info("Canonical parquet written: %s (%d rows)", out_path, len(df_canonical))

    # Throughput
    elapsed_gen = time.perf_counter() - t_start
    throughput = len(events) / max(elapsed_gen, 1e-6)
    log.info("Throughput: %.0f events/sec", throughput)

    # Memory (optional)
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
        log.info("RSS memory: %.1f MB", rss_mb)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Write schema.md and report
    # ------------------------------------------------------------------
    _write_schema_md(schema_out)
    log.info("Schema written: %s", schema_out)

    effective_services = services if services else _DEFAULT_SERVICES
    _write_synth_report(
        df_canonical=df_canonical,
        n_events=n_events,
        services=effective_services,
        seed=seed,
        out_path=out_path,
        schema_out=schema_out,
        elapsed=elapsed_gen,
        report_path=report_path,
    )
    log.info("Report written: %s", report_path)

    elapsed = time.perf_counter() - t_start
    parquet_ok = out_path.exists()
    summary = {
        "n_events":       len(df_canonical),
        "n_scenarios":    len(scenarios),
        "label_counts":   label_counts,
        "phase_counts":   phase_counts,
        "anomaly_rate":   round(anomaly_rate, 2),
        "csv_path":       str(csv_path),
        "parquet_path":   str(out_path),
        "parquet_ok":     parquet_ok,
        "elapsed_s":      round(elapsed, 2),
        "seed":           seed,
        "mode":           mode,
    }

    log.info("--- Summary ---")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)
    log.info("=== Stage 01 Synthetic Generate complete in %.2fs ===", elapsed)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 01 Synthetic Generate")
    # Legacy args (unchanged)
    parser.add_argument("--mode",    default="demo", choices=["demo", "full"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--events",  type=int, default=None,
                        help="Override default event count (legacy)")
    parser.add_argument("--outdir",  type=Path, default=ROOT / "data" / "synth")
    # New args
    parser.add_argument("--n-events", dest="n_events", type=int, default=None,
                        help="Number of events to generate (default 50000; overrides --events)")
    parser.add_argument("--services", type=str, default="auth,api,billing,db",
                        help="Comma-separated service names (default: auth,api,billing,db)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "synth" / "events_synth.parquet",
                        help="Output path for canonical parquet")
    parser.add_argument("--schema-out", dest="schema_out", type=Path,
                        default=ROOT / "data" / "synth" / "schema.md",
                        help="Output path for schema.md")
    args = parser.parse_args()

    # Resolve n_events: --n-events > --events > mode default
    n_events = args.n_events or args.events
    if n_events is None:
        n_events = 50_000 if args.mode == "demo" else 200_000

    services = [s.strip() for s in args.services.split(",") if s.strip()]

    run_generate(
        mode=args.mode,
        seed=args.seed,
        n_events=n_events,
        outdir=args.outdir,
        services=services,
        out_path=args.out,
        schema_out=args.schema_out,
    )


if __name__ == "__main__":
    main()
