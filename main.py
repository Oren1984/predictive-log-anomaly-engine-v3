# main.py
# Project entrypoint for the Predictive Log Anomaly Engine.

# This file starts the FastAPI service using the existing create_app factory.
# It delegates to the same uvicorn configuration used by
# scripts/run_api.py.

# Usage:
#   python main.py
#   python main.py --host 127.0.0.1 --port 8000
#   python main.py --model baseline

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path regardless of where this is called from
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Delegate to the existing entrypoint which handles arg-parsing and uvicorn startup.
# This keeps main.py thin and avoids duplicating configuration logic.
if __name__ == "__main__":
    from scripts.run_api import main
    main()
