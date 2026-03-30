# src/alerts/n8n_client.py

# Purpose: Implement the N8nWebhookClient class, which is responsible for sending alert payloads to an n8n webhook.

# Input: The N8nWebhookClient class takes configuration parameters for the webhook URL, dry-run mode, timeout, and outbox directory. 
# It provides a send method that accepts an Alert object and dispatches it 
# according to the configured mode (either by POSTing to the webhook or writing to a local outbox).

# Output: The send method returns a result dictionary indicating the status of the operation, 
# including whether it was a dry run, the path to the outbox file, or the HTTP status code.

# Used by: The N8nWebhookClient is used in the main API implementation (src.api.app.py) to send alerts to n8n when they are fired. 
# It is also tested in the test file test_stage_06_n8n_outbox.py

"""Stage 6 — Alerts: n8n webhook client (safe DRY_RUN by default)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Alert

logger = logging.getLogger(__name__)

_DEFAULT_OUTBOX = Path("artifacts") / "n8n_outbox"


class N8nWebhookClient:
    """
    Forward Alert payloads to an n8n webhook or write them to a local outbox.

    Configuration — resolved in priority order: constructor args > environment variables.

    Environment variables
    ---------------------
    N8N_WEBHOOK_URL        Optional webhook URL. If empty or absent, always dry-runs.
    N8N_DRY_RUN            "true" (default) to skip the HTTP call and write to outbox.
                           Set to "false" to enable live POSTing (requires N8N_WEBHOOK_URL).
    N8N_TIMEOUT_SECONDS    HTTP request timeout in seconds (default 5).

    DRY_RUN behaviour (safe default)
    ---------------------------------
    Writes the JSON payload to:
        <outbox_dir>/<alert_id>.json
    Returns: {"status": "dry_run", "path": "<absolute-path>"}

    LIVE behaviour (N8N_DRY_RUN=false + URL present)
    -------------------------------------------------
    POSTs the JSON payload to N8N_WEBHOOK_URL.
    Returns: {"status": "ok", "status_code": <int>}
    On error: falls back to outbox and returns {"status": "error", "reason": "<msg>"}
    Never raises.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        dry_run: Optional[bool] = None,
        timeout: Optional[float] = None,
        outbox_dir: Optional[Path] = None,
    ) -> None:
        self.webhook_url: str = webhook_url or os.environ.get("N8N_WEBHOOK_URL", "")

        if dry_run is None:
            env_val = os.environ.get("N8N_DRY_RUN", "true").strip().lower()
            dry_run = env_val != "false"
        self.dry_run: bool = dry_run

        if timeout is None:
            try:
                timeout = float(os.environ.get("N8N_TIMEOUT_SECONDS", "5"))
            except (ValueError, TypeError):
                timeout = 5.0
        self.timeout: float = timeout

        self.outbox_dir: Path = Path(outbox_dir) if outbox_dir else _DEFAULT_OUTBOX

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, alert: "Alert") -> dict:
        """
        Dispatch *alert* according to the current mode.

        Returns a result dict (never raises).
        """
        if self.dry_run or not self.webhook_url:
            return self._write_outbox(alert.alert_id, alert.to_dict())
        return self._post(alert.to_dict())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_outbox(self, alert_id: str, payload: dict) -> dict:
        """Write payload JSON to outbox_dir/<alert_id>.json."""
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outbox_dir / f"{alert_id}.json"
        out_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("DRY_RUN: alert payload written -> %s", out_path)
        return {"status": "dry_run", "path": str(out_path)}

    def _post(self, payload: dict) -> dict:
        """POST payload to the configured webhook URL."""
        try:
            import requests  # optional dependency
        except ImportError:
            logger.error(
                "requests library not installed; falling back to outbox. "
                "Install with: pip install requests"
            )
            return self._write_outbox(payload.get("alert_id", "unknown"), payload)

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            logger.info("Alert POSTed to n8n: status=%d url=%s",
                        resp.status_code, self.webhook_url)
            return {"status": "ok", "status_code": resp.status_code}

        except Exception as exc:
            logger.warning(
                "n8n POST failed (%s) -- falling back to outbox", exc
            )
            result = self._write_outbox(payload.get("alert_id", "unknown"), payload)
            result["status"] = "error"
            result["reason"] = str(exc)
            return result
