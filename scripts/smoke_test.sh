#!/usr/bin/env bash
# smoke_test.sh — start the compose stack, verify /health, /metrics,
# ingest->alert pipeline, then tear down.
# Idempotent: always runs `docker compose down -v` first to clear old state.
set -euo pipefail

HEALTH_URL="http://localhost:8000/health"
METRICS_URL="http://localhost:8000/metrics"
INGEST_URL="http://localhost:8000/ingest"
ALERTS_URL="http://localhost:8000/alerts"
MAX_WAIT=60
INTERVAL=5

echo "=== Smoke Test: Predictive Log Anomaly Engine ==="

# Ensure clean state
echo "[1/6] Cleaning up any existing compose state..."
docker compose down -v --remove-orphans 2>/dev/null || true

# Ensure model directories exist (empty is fine; compose mounts them)
mkdir -p models artifacts

# Start stack
echo "[2/6] Starting compose stack (detached)..."
docker compose up -d --build

# Wait for /health
echo "[3/6] Waiting for API to respond (up to ${MAX_WAIT}s)..."
elapsed=0
while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
  echo "  ${elapsed}s: GET /health -> HTTP $STATUS"
  if [ "$STATUS" = "200" ]; then
    echo "  API is up!"
    break
  fi
  if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "ERROR: API did not respond within ${MAX_WAIT}s"
    docker compose logs api
    docker compose down -v
    exit 1
  fi
  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done

# Smoke /health
echo "[4/6] Verifying /health and /metrics..."
BODY=$(curl -sf "$HEALTH_URL")
echo "  GET /health: $BODY"

MET_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$METRICS_URL")
echo "  GET /metrics -> HTTP $MET_STATUS"
if [ "$MET_STATUS" != "200" ]; then
  echo "ERROR: /metrics returned HTTP $MET_STATUS"
  docker compose down -v
  exit 1
fi

# Ingest events and verify alert creation
echo "[5/6] Posting 10 events to /ingest (DEMO_MODE; WINDOW_SIZE=5, STRIDE=1)..."
INGEST_PAYLOAD='{"service":"smoke","token_id":42}'
for i in $(seq 1 10); do
  R=$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "$INGEST_URL" \
      -H "Content-Type: application/json" \
      -d "$INGEST_PAYLOAD")
  echo "  event $i -> HTTP $R"
  if [ "$R" != "200" ]; then
    echo "ERROR: /ingest returned HTTP $R on event $i"
    docker compose logs api
    docker compose down -v
    exit 1
  fi
done

# Verify at least 1 alert was created
echo "  Checking /alerts..."
ALERTS_BODY=$(curl -sf "$ALERTS_URL")
echo "  GET /alerts: $ALERTS_BODY"
ALERT_COUNT=$(echo "$ALERTS_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['count'])" 2>/dev/null || echo "0")
echo "  Alert count: $ALERT_COUNT"
if [ "$ALERT_COUNT" -lt 1 ]; then
  echo "ERROR: Expected at least 1 alert, got $ALERT_COUNT"
  docker compose logs api
  docker compose down -v
  exit 1
fi
echo "  ingest->alert path VERIFIED (count=$ALERT_COUNT)"

# Tear down
echo "[6/6] Tearing down compose stack..."
docker compose down -v

echo ""
echo "=== Smoke test PASSED ==="
