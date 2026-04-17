"""
Auto-generated from v3_semantic_demo.ipynb.
Converted from notebook to runnable Python script.
This script exercises the local API demo endpoints.
"""


"""
# V3 Semantic Layer Demo

This notebook demonstrates the **Phase 8 V3 semantic API** of the Predictive Log Anomaly Engine.

It shows:
- Ingesting events via `POST /v3/ingest` and receiving semantic enrichment
- Fetching explanations via `GET /v3/alerts/{alert_id}/explanation`
- Inspecting model and semantic layer status via `GET /v3/models/info`
- Viewing V3 metrics from `GET /metrics`

**Prerequisites:**
```bash
# Start the API with semantic enabled:
SEMANTIC_ENABLED=true docker compose -f docker/docker-compose.yml up
# Or run locally:
SEMANTIC_ENABLED=true DEMO_MODE=true python -m uvicorn src.api.app:create_app --factory --port 8000
```
"""

import os
import json
import time

try:
    import requests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
    import requests

BASE_URL = os.getenv('PREDICTIVE_API_BASE_URL', 'http://localhost:8000')

def pretty(data):
    print(json.dumps(data, indent=2, default=str))

print('Base URL:', BASE_URL)


"""
## 1. Check V3 Models Info
"""

resp = requests.get(f'{BASE_URL}/v3/models/info')
resp.raise_for_status()
info = resp.json()
print('=== /v3/models/info ===')
pretty(info)

print()
print(f'Inference mode:        {info["inference_mode"]}')
print(f'Artifacts loaded:      {info["artifacts_loaded"]}')
print(f'Semantic enabled:      {info["semantic_enabled"]}')
print(f'Semantic model loaded: {info["semantic_model_loaded"]}')


"""
## 2. Check Health (V3 Semantic Component)
"""

resp = requests.get(f'{BASE_URL}/health')
health = resp.json()
print('=== /health ===')
pretty(health)

semantic_component = health.get('components', {}).get('semantic', {})
print()
print('Semantic component:', semantic_component)


"""
## 3. Ingest Events via POST /v3/ingest

With `DEMO_MODE=true` the engine uses a synthetic score so alerts fire without real trained models.
"""

# Ingest enough events to fill a window (stride=1 in demo mode)
fired_alert = None
for i in range(15):
    payload = {
        'service': 'hdfs',
        'token_id': (abs(hash(f'demo-{i}')) % 7833) + 2,
        'session_id': 'v3-demo-session',
        'timestamp': time.time(),
    }
    resp = requests.post(f'{BASE_URL}/v3/ingest', json=payload)
    body = resp.json()
    if body.get('alert'):
        fired_alert = body['alert']
        print(f'Alert fired on event {i+1}!')
        break

if fired_alert:
    print('\n=== Alert payload (V3 semantic fields) ===')
    for field in ('alert_id', 'severity', 'score', 'explanation',
                  'semantic_similarity', 'evidence_tokens', 'top_similar_events'):
        print(f'  {field}: {fired_alert.get(field)}')
else:
    print('No alert fired yet — try increasing DEMO_SCORE or reducing window size.')


"""
## 4. Fetch Explanation for a Specific Alert
"""

if fired_alert:
    alert_id = fired_alert['alert_id']
    resp = requests.get(f'{BASE_URL}/v3/alerts/{alert_id}/explanation')
    print(f'=== /v3/alerts/{alert_id}/explanation ===')
    pretty(resp.json())
else:
    print('No alert available — run cell 3 first.')


"""
## 5. List All Alerts (with V3 fields)
"""

resp = requests.get(f'{BASE_URL}/alerts')
data = resp.json()
print(f'Total alerts in buffer: {data["count"]}')

for alert in data['alerts']:
    print()
    print(f'  ID:                  {alert["alert_id"]}')
    print(f'  Severity:            {alert["severity"]}')
    print(f'  Score:               {alert["score"]:.3f}')
    print(f'  Explanation:         {alert.get("explanation")}')
    print(f'  Semantic similarity: {alert.get("semantic_similarity")}')


"""
## 6. V3 Prometheus Metrics
"""

resp = requests.get(f'{BASE_URL}/metrics')
lines = resp.text.splitlines()
v3_lines = [l for l in lines if 'semantic' in l]

print('=== V3 Semantic Metrics ===')
for line in v3_lines:
    print(line)
