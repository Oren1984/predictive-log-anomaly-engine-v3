# n8n Flow Stub — Predictive Log Anomaly Alerts

## Overview

This stub describes how to connect the Stage 6 alert outbox to n8n for
downstream notification (Slack, email, GitHub Issues, PagerDuty, etc.).

The alert pipeline writes JSON payloads to `artifacts/n8n_outbox/` in DRY_RUN
mode. To send live, set `N8N_WEBHOOK_URL` and `N8N_DRY_RUN=false`.

---

## Payload Schema

See `examples/n8n/sample_alert_payload.json` for a concrete example.

| Field | Type | Description |
|-------|------|-------------|
| `alert_id` | string (UUID) | Unique alert identifier |
| `severity` | string | `critical` / `high` / `medium` / `low` |
| `service` | string | Affected service name |
| `score` | float | Raw anomaly score |
| `timestamp` | float | Unix epoch of last event in window |
| `evidence_window` | object | Template previews and window metadata |
| `model_name` | string | `baseline` / `transformer` / `ensemble` |
| `threshold` | float | Decision threshold used |
| `meta` | object | Stream key, emit index, label, window size |

---

## Suggested n8n Flow

```
[Webhook Trigger]
      |
      v
[IF: severity == "critical"]
   YES --> [Slack: #alerts-critical] --> [GitHub: Create Issue]
   NO  --> [IF: severity in ["high","medium"]]
              YES --> [Slack: #alerts-general]
              NO  --> [Log to file / ignore]
```

---

## Node Configuration

### 1. Webhook Trigger

- **HTTP Method:** POST
- **Path:** `/webhook/anomaly-alert`
- **Authentication:** Header Auth (add `X-Alert-Token` for security)
- **Response Mode:** Immediately

The full URL that must be set in `.env`:
```
N8N_WEBHOOK_URL=https://<your-n8n-host>/webhook/anomaly-alert
```

### 2. IF Node — Critical Severity

- **Condition:** `{{ $json.severity }}` equals `critical`

### 3. Slack Node — Critical Channel

- **Channel:** `#alerts-critical`
- **Message:**
  ```
  :rotating_light: *CRITICAL ANOMALY* — {{ $json.service }}
  Score: {{ $json.score }} (threshold: {{ $json.threshold }})
  Model: {{ $json.model_name }}
  Alert ID: {{ $json.alert_id }}
  Evidence: {{ $json.evidence_window.templates_preview[0] }}
  ```

### 4. GitHub Issue Node (optional, critical only)

- **Repository:** your-org/your-repo
- **Title:** `[ANOMALY] {{ $json.service }} — score {{ $json.score }}`
- **Body:** Include `alert_id`, `service`, `score`, `evidence_window`.
- **Labels:** `anomaly`, `severity:critical`

### 5. Slack Node — General Channel

- **Channel:** `#alerts-general`
- **Message:**
  ```
  :warning: *{{ $json.severity | upper }} ANOMALY* — {{ $json.service }}
  Score: {{ $json.score }} | Model: {{ $json.model_name }}
  Alert ID: {{ $json.alert_id }}
  ```

### 6. Email (SMTP) Node — optional

- **To:** on-call@example.com (for critical)
- **Subject:** `[{{ $json.severity | upper }}] Log Anomaly: {{ $json.service }}`
- **Body:** Full JSON payload as formatted text.

---

## Environment Variables

Add these to `.env` (see `.env.example`):

```dotenv
N8N_WEBHOOK_URL=https://<your-n8n-host>/webhook/anomaly-alert
N8N_DRY_RUN=true
N8N_TIMEOUT_SECONDS=5
```

Set `N8N_DRY_RUN=false` only when your n8n instance is running and the
webhook path is configured.

---

## Testing the Webhook Locally

1. Start n8n locally: `npx n8n`
2. Create a Webhook node at `/webhook/anomaly-alert`
3. Set env vars and run the demo:

```powershell
$env:N8N_WEBHOOK_URL = "http://localhost:5678/webhook/anomaly-alert"
$env:N8N_DRY_RUN = "false"
python scripts/data_pipeline/stage_06_demo_alerts.py --n-events 2000 --cooldown 0
```

Or send a single test payload manually:

```powershell
$payload = Get-Content examples/n8n/sample_alert_payload.json | ConvertFrom-Json | ConvertTo-Json
Invoke-RestMethod -Uri $env:N8N_WEBHOOK_URL -Method Post -Body $payload -ContentType "application/json"
```
