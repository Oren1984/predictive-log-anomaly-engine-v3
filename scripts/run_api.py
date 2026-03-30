# scripts/run_api.py

# Run API — start the FastAPI service with uvicorn.

# Purpose: This script starts the FastAPI server that serves the anomaly detection model.
# It reads configuration from environment variables or command-line arguments and runs the API server using uvicorn.

# Input: None (the server will read configuration from environment variables or CLI args)

# Output: The API server will be running and ready to accept requests on the specified host and port.

# Used by: main.py delegates to this script; can also be run directly.

"""
run_api -- start the FastAPI service with uvicorn.

Usage
-----
    python scripts/run_api.py
    python scripts/run_api.py --host 127.0.0.1 --port 8000
    python scripts/run_api.py --disable-auth
    python scripts/run_api.py --model baseline

Quick test (in a second terminal after server is running)
---------------------------------------------------------
    # health check (no key needed)
    curl http://localhost:8000/health

    # ingest an event
    curl -X POST http://localhost:8000/ingest \\
         -H "X-API-Key: changeme" \\
         -H "Content-Type: application/json" \\
         -d '{"service":"hdfs","token_id":10,"timestamp":1704067200}'

    # list recent alerts
    curl http://localhost:8000/alerts -H "X-API-Key: changeme"

    # prometheus metrics
    curl http://localhost:8000/metrics
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API server")
    parser.add_argument("--host",         default=None,
                        help="Bind host (default: API_HOST env or 0.0.0.0)")
    parser.add_argument("--port",         type=int, default=None,
                        help="Bind port (default: API_PORT env or 8000)")
    parser.add_argument("--api-key",      dest="api_key", default=None,
                        help="API key (default: API_KEY env var)")
    parser.add_argument("--disable-auth", dest="disable_auth",
                        action="store_true", default=False,
                        help="Disable API key authentication")
    parser.add_argument("--model",        default=None,
                        choices=["baseline", "transformer", "ensemble"],
                        help="Scoring model (default: MODEL_MODE env or ensemble)")
    parser.add_argument("--reload",       action="store_true", default=False,
                        help="Enable uvicorn hot reload (dev mode only)")
    args = parser.parse_args()

    # Push CLI overrides into env so Settings picks them up
    if args.host:
        os.environ["API_HOST"] = args.host
    if args.port:
        os.environ["API_PORT"] = str(args.port)
    if args.api_key:
        os.environ["API_KEY"] = args.api_key
    if args.disable_auth:
        os.environ["DISABLE_AUTH"] = "true"
    if args.model:
        os.environ["MODEL_MODE"] = args.model

    from src.api.settings import Settings

    cfg = Settings()

    import uvicorn
    uvicorn.run(
        "src.api.app:create_app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=args.reload,
        factory=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
