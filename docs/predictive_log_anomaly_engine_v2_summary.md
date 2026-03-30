# Predictive Log Anomaly Engine v2 — Repository Summary

> Generated: 2026-03-29
> Source: Full repository scan of `predictive-log-anomaly-engine-v2`

---

## Executive Summary

Predictive Log Anomaly Engine v2 is an end-to-end AIOps system designed to detect anomalous behavior in log streams using a rolling-window ML pipeline.

The system exposes a REST API for ingesting log events, processes them through a sequence-based anomaly detection engine (IsolationForest + Transformer ensemble), and generates alerts with severity classification.

The project demonstrates:
- Real-time anomaly detection
- ML + system integration
- Full observability (Prometheus + Grafana)
- Containerized deployment (Docker Compose)
- CI/CD validation pipeline

Two inference pipelines exist:
- v1 (active): ensemble-based anomaly detection
- v2 (optional): deep learning pipeline (LSTM + Autoencoder + MLP)

The system is designed for:
- Academic demonstration
- System design evaluation
- DevOps + ML integration showcase

---

## Quick Understanding (TL;DR)

- Input: tokenized log events (v1) or raw logs (v2)
- Core logic: rolling window anomaly scoring
- Models: IsolationForest + Transformer (v1)
- Output: anomaly alerts with severity
- Observability: Prometheus + Grafana
- Deployment: Docker Compose
- API: FastAPI

---

## Mental Model of the System

Think of the system as a pipeline of 4 layers:

1. Ingestion Layer
   → Receives log tokens via API

2. Sequence Layer
   → Builds rolling windows of events

3. Intelligence Layer
   → Applies anomaly detection models

4. Action Layer
   → Emits alerts + exposes metrics

---

## 1. Project Overview

**What the system does:**
Predictive Log Anomaly Engine v2 is a real-time log anomaly detection system. It ingests structured log events via a REST API, processes them through a rolling-window ML pipeline, and emits scored anomaly alerts with severity classification.

**Main purpose:**
Detect anomalous sequences of log template tokens in streaming log data from distributed systems (HDFS, BGL datasets). Provide a deployable, observable, API-first anomaly detection service.

**High-level workflow:**
1. Log events arrive as `POST /ingest` requests carrying a `token_id` (pre-mined template identifier).
2. Events are buffered per stream (service + session) in a rolling window.
3. When a full window is available, the inference engine scores it using one or more ML models.
4. Anomalous windows trigger alerts, which are deduplicated by cooldown policy and written to an outbox or forwarded to a webhook.
5. All activity is instrumented with Prometheus metrics, visualized in a Grafana dashboard.

**Core value:**
End-to-end demonstrable AIOps pipeline — from raw log events through ML inference, alerting, and observability — packaged in a containerized, CI-tested stack suitable for academic study, system engineering review, or portfolio demonstration.

---

## 2. System Architecture

**Main components:**

| Component | Role |
|-----------|------|
| FastAPI application (`src/api/`) | REST interface, middleware orchestration |
| Inference Engine v1 (`src/runtime/inference_engine.py`) | Active scoring engine: baseline + transformer + ensemble |
| Inference Engine v2 (`src/runtime/inference_engine_v2.py`) | Optional v2 pipeline: Word2Vec + LSTM + Autoencoder + MLP |
| Sequence Buffer (`src/runtime/sequence_buffer.py`) | Per-stream rolling window |
| Alert Manager (`src/alerts/manager.py`) | Dedup, cooldown, severity classification |
| n8n Webhook Client (`src/alerts/n8n_client.py`) | Alert forwarding or local outbox write |
| Prometheus + Grafana (`prometheus/`, `grafana/`) | Metrics scraping and dashboarding |
| Authentication Middleware (`src/security/auth.py`) | API key enforcement |
| Metrics Middleware (`src/observability/metrics.py`) | Prometheus instrumentation |

**How data moves through the system (v1 runtime):**

```
POST /ingest (token_id, service, session_id, timestamp)
  → AuthMiddleware (X-API-Key validation)
  → MetricsMiddleware (latency recording)
  → Route handler (routes.py)
  → Pipeline.process_event()
    → SequenceBuffer.ingest()       # append token to rolling deque
    → [if stride boundary reached]
    → InferenceEngine.score()
      → BaselineFeatureExtractor    # 204-dimensional feature vector
      → IsolationForest             # baseline anomaly score
      → NextTokenTransformer        # NLL-based transformer score
      → Ensemble normalization      # (b_norm + t_norm) / 2.0
    → AlertManager.emit()           # policy check + cooldown
    → N8nWebhookClient.send()       # outbox write or HTTP POST
  ← IngestResponse (window_emitted, risk_result, alert)
```

**Important architecture decisions:**
- v1 and v2 inference paths are completely isolated — no shared state.
- v2 engine is loaded only when `MODEL_MODE` env contains `"v2"` (opt-in).
- Demo mode (`DEMO_MODE=true`) bypasses model loading and returns a synthetic anomaly score, enabling CI smoke tests without trained artifacts.
- `ProactiveEngine` (`src/engine/proactive_engine.py`) is explicitly marked as legacy and is not wired to any API route.
- Alert webhook integration defaults to a dry-run local outbox (`artifacts/n8n_outbox/`), preventing accidental live webhook calls.

---

## 3. Repository Structure

```
predictive-log-anomaly-engine-v2/
├── main.py                         # Thin entrypoint → delegates to scripts/stage_07_run_api.py
├── src/                            # All runtime application code
│   ├── api/                        # FastAPI app, routes, schemas, settings, UI
│   ├── runtime/                    # Inference engines, sequence buffer, result types
│   ├── alerts/                     # Alert manager, models, n8n client
│   ├── modeling/                   # ML model classes (baseline, transformer, v2 models)
│   ├── parsing/                    # Template miner, tokenizer, parsers
│   ├── sequencing/                 # Sequence dataclass and builders
│   ├── observability/              # Prometheus metrics registry and middleware
│   ├── security/                   # API key authentication middleware
│   ├── health/                     # Health check logic
│   ├── data_layer/                 # LogEvent domain model and loader
│   ├── data/                       # Synthetic data generators (demo/test use)
│   ├── synthetic/                  # Alternate synthetic generator module
│   ├── preprocessing/              # Log preprocessor (used by v2/legacy paths)
│   ├── dataset/                    # LogDataset wrapper
│   └── engine/                     # ProactiveEngine (legacy, not wired)
├── scripts/                        # Training + API entrypoint scripts
│   └── stage_07_run_api.py         # Primary API startup script
├── tests/                          # Full test suite
│   ├── unit/                       # Unit tests for core components
│   ├── integration/                # API integration smoke tests
│   ├── system/                     # End-to-end and performance tests
│   └── helpers_stage_07.py         # Shared mock helpers
├── models/                         # Trained model artifacts
│   ├── baseline.pkl                # IsolationForest (v1)
│   ├── transformer.pt              # Causal transformer (v1)
│   ├── embeddings/word2vec.model   # Word2Vec (v2)
│   ├── behavior/behavior_model.pt  # LSTM behavior model (v2)
│   ├── anomaly/anomaly_detector.pt # Autoencoder (v2)
│   └── severity/severity_classifier.pt  # MLP severity (v2)
├── artifacts/                      # Vocabulary, thresholds, alert outbox
│   ├── vocab.json                  # token_id → template text
│   ├── templates.json              # template_id → template text
│   ├── threshold.json              # baseline decision threshold
│   ├── threshold_transformer.json  # transformer decision threshold
│   └── n8n_outbox/                 # Dry-run alert JSON payloads
├── docker/                         # Dockerfile and docker-compose.yml
├── prometheus/                     # prometheus.yml + alerts.yml
├── grafana/                        # Dashboard JSON + provisioning configs
├── requirements/                   # requirements.txt, requirements-dev.txt
├── templates/                      # Jinja2/HTML for demo UI
├── data/                           # Raw, processed, intermediate datasets
├── notebooks/                      # Jupyter notebooks (demo/exploratory)
├── docs/                           # Documentation (partially deleted in git status)
├── ai_workspace/                   # AI development stage reports (partially deleted)
└── .github/workflows/ci.yml        # GitHub Actions CI pipeline
```

**Central folders:** `src/`, `models/`, `artifacts/`, `tests/`, `docker/`, `prometheus/`, `grafana/`
**Supportive/optional:** `notebooks/`, `ai_workspace/`, `data/`

---

## 4. API Layer

**Framework:** FastAPI, served via Uvicorn (ASGI)

**Application factory:** `src/api/app.py` — `create_app()` function, used as a factory by Uvicorn (`--factory` flag).

**Middleware stack (outer → inner):**
1. `MetricsMiddleware` — records `ingest_latency_seconds` for `POST /ingest`
2. `AuthMiddleware` — validates `X-API-Key` header; bypasses public endpoints

**Routers included:**
- `router` (v1 routes) — `src/api/routes.py`
- `router_v2` (v2 routes) — `src/api/routes_v2.py`
- `ui_router` (demo UI) — `src/api/ui.py`

**V1 Endpoints** (`src/api/routes.py`):

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/ingest` | Required | Ingest a single log event token; returns scoring result and alert if fired |
| GET | `/alerts` | Public | Returns up to `ALERT_BUFFER_SIZE` recent alerts |
| GET | `/health` | Public | Service health check with component status |
| GET | `/metrics` | Public | Prometheus metrics in text exposition format |

**V2 Endpoints** (`src/api/routes_v2.py`):

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v2/ingest` | Required | Ingest a raw log string; tokenized internally by v2 pipeline |
| GET | `/v2/alerts` | Public | Returns v2 alert buffer (503 if v2 engine not loaded) |

**Demo UI Endpoints** (`src/api/ui.py`):

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | Public | Serves `templates/index.html` single-page demo UI |
| POST | `/query` | Public | Hardcoded RAG stub — 8-topic Q&A knowledge base |

**`POST /ingest` request/response:**
```json
// Request
{ "service": "hdfs", "token_id": 42, "session_id": "blk_123", "timestamp": 1704067200.0, "label": 0 }

// Response
{ "window_emitted": true, "risk_result": { "score": 1.8, "is_anomaly": true, ... }, "alert": { "severity": "high", ... } }
```

**`GET /health` response:**
```json
{ "status": "healthy", "uptime_seconds": 123.4, "components": { "inference_engine": "ok", "alert_manager": "ok" } }
```

---

## 5. Detection and Scoring Logic

### V1 Inference Engine (`src/runtime/inference_engine.py`) — Active Runtime

**Three selectable modes** (via `MODEL_MODE` env var):

#### Baseline Mode (`"baseline"`)
- Model: scikit-learn `IsolationForest` (300 estimators, random_state=42)
- Feature extraction: `BaselineFeatureExtractor` — 204 features per window
  - 4 aggregate features: length, unique count, unique ratio, Shannon entropy
  - 200 raw + normalized counts for top-100 token IDs
- Score convention: negated IsolationForest score (higher = more anomalous)
- Decision: `score >= threshold_baseline` (~0.33)

#### Transformer Mode (`"transformer"`)
- Model: GPT-style causal decoder transformer (`src/modeling/transformer/model.py`)
  - vocab_size=7835, d_model=256, n_heads=8, n_layers=4, d_ff=1024, max_seq_len=512
- Scoring: cumulative negative log-likelihood (NLL) of next tokens
- Decision: `score >= threshold_transformer` (~0.034)

#### Ensemble Mode (`"ensemble"`) — Default
- Combines both models with normalized scoring:
  - `b_norm = baseline_score / threshold_baseline`
  - `t_norm = transformer_score / threshold_transformer`
  - `ensemble_score = (b_norm + t_norm) / 2.0`
  - Decision threshold: `ensemble_score >= 1.0`
- A score above 1.0 means at least one model is above its individual threshold.

**Demo/Fallback behavior:**
- When `DEMO_MODE=true` and models are missing: returns `DEMO_SCORE` (default 2.0), guaranteed anomaly.
- When `DEMO_MODE=false` and models are missing: returns 0.0, no spurious alerts.
- This distinction is critical: demo mode is used in CI smoke tests; production mode fails safe.

### V2 Inference Engine (`src/runtime/inference_engine_v2.py`, `src/runtime/pipeline_v2.py`) — Optional

Activated when `MODEL_MODE` contains `"v2"`. Uses a four-stage pipeline:
1. `_V2LogTokenizer` — normalizes raw log string via regex substitutions → template_id → token_id
2. `Word2Vec` — maps `str(token_id)` → float32 embedding vector (`models/embeddings/word2vec.model`)
3. `SystemBehaviorModel` (LSTM) — produces a context vector from rolling window (`models/behavior/behavior_model.pt`)
4. `AnomalyDetector` (Autoencoder) — reconstruction error → anomaly score + flag (`models/anomaly/anomaly_detector.pt`)
5. `SeverityClassifier` (MLP) — classifies severity into Info/Warning/Critical (`models/severity/severity_classifier.pt`)

The V2 pipeline accepts raw log strings (not pre-tokenized token_ids), making it self-contained for log text input.

**All four V2 model artifacts are present in the repository** (`models/` subdirectories have `.pt` and `.model` files). V2 routes return HTTP 503 if the engine was not loaded at startup.

### ProactiveEngine (`src/engine/proactive_engine.py`) — Legacy, Not Wired

Explicitly self-documented as `STATUS: LEGACY`. Orchestrates `LogPreprocessor → LSTM → Autoencoder → SeverityClassifier` in a different runtime path. Not included in any API route. Retained for test coverage and reference.

---

## 6. Data Processing Pipeline

**Offline pipeline (training-time, produces artifacts):**

```
Raw HDFS / BGL logs
  → TemplateMiner (src/parsing/template_miner.py)
    → 9-step regex substitution pipeline
      (BLK IDs, timestamps, IPs, dates, node names, paths, hex, numbers, whitespace)
    → 7,833 unique templates from 15.9M events
    → artifacts/templates.json, artifacts/vocab.json
  → EventTokenizer (src/parsing/tokenizer.py)
    → token_id = template_id + 2  (0=PAD, 1=UNK reserved)
  → SequenceBuilder (src/sequencing/builders.py)
    → sliding windows, stored as data/processed/sequences_train.parquet
  → BaselineFeatureExtractor.fit() on training sequences
  → IsolationForest.fit() → models/baseline.pkl
  → NextTokenTransformerModel.train() → models/transformer.pt
  → Threshold calibration → artifacts/threshold.json, artifacts/threshold_transformer.json
```

**Runtime pipeline (inference-time, per event):**

```
POST /ingest (token_id already pre-mapped)
  → SequenceBuffer (rolling deque per stream key)
  → [on stride boundary] InferenceEngine.score(window)
  → RiskResult(score, is_anomaly, threshold, evidence_window)
```

**V2 runtime adds tokenization inline:**
```
POST /v2/ingest (raw log string)
  → _V2LogTokenizer (regex + templates.csv lookup) → token_id
  → Word2Vec lookup → float32 vector
  → rolling buffer → LSTM → Autoencoder → SeverityClassifier
  → V2Result(anomaly_score, is_anomaly, severity, severity_probabilities)
```

**Key data artifacts:**
- `data/processed/sequences_train.parquet` — preprocessed training sequences used by `BaselineFeatureExtractor.fit()`
- `artifacts/vocab.json` (1.5 MB) — token ID to template text mapping
- `artifacts/templates.json` (1.5 MB) — template ID to template text mapping

---

## 7. Models and Artifacts

### Model Files (`models/`)

| File | Size | Type | Role |
|------|------|------|------|
| `models/baseline.pkl` | ~1.6 MB | scikit-learn pickle | IsolationForest for v1 baseline scoring |
| `models/transformer.pt` | ~2.1 MB | PyTorch state dict | Causal transformer for v1 NLL scoring |
| `models/embeddings/word2vec.model` | present | gensim Word2Vec | Token embedding for v2 pipeline |
| `models/behavior/behavior_model.pt` | present | PyTorch LSTM | Context vector for v2 behavior stage |
| `models/anomaly/anomaly_detector.pt` | present | PyTorch Autoencoder | Reconstruction error anomaly scoring (v2) |
| `models/severity/severity_classifier.pt` | present | PyTorch MLP | Severity classification Info/Warning/Critical (v2) |

### Threshold Artifacts (`artifacts/`)

| File | Role |
|------|------|
| `artifacts/threshold.json` | Baseline decision threshold (~0.33) |
| `artifacts/threshold_transformer.json` | Transformer decision threshold (~0.034) |
| `artifacts/threshold_runtime.json` | Optional runtime-calibrated thresholds (loaded if present) |
| `artifacts/vocab.json` | Token ID → template text, used by explain() in InferenceEngine |
| `artifacts/templates.json` | Template ID → template text, used by v2 tokenizer |
| `artifacts/n8n_outbox/` | Dry-run alert payloads (100+ JSON files from prior runs) |

### Training Performance (v1 models, from codebase documentation)

- Baseline IsolationForest: F1=0.96 on BGL dataset, F1=0.047 on HDFS dataset
- Best overall F1≈0.38 at threshold 0.307 (combined evaluation)
- HDFS dataset: 10.9M normal + 288K anomaly events
- BGL dataset: 348K normal + 4.4M anomaly events

---

## 8. Observability

### Prometheus (`prometheus/`)

**Config file:** `prometheus/prometheus.yml`
- Scrape interval: 15s
- Target: `api:8000` (Docker service name), path `/metrics`
- Alert rules file: `prometheus/alerts.yml`

**Alert rules defined in `prometheus/alerts.yml`:**

| Alert | Severity | Condition | For |
|-------|----------|-----------|-----|
| `ServiceDown` | critical | `up{job="anomaly-api"} == 0` | 1m |
| `ServiceUnhealthy` | warning | `service_health < 1.0` | 2m |
| `HighIngestErrorRate` | warning | `rate(ingest_errors_total[5m]) > 0.1` | 2m |
| `IngestStalled` | warning | `rate(ingest_events_total[5m]) == 0` | 5m |

### Application Metrics (`src/observability/metrics.py`)

| Metric | Type | Description |
|--------|------|-------------|
| `ingest_events_total` | Counter | Events received at POST /ingest |
| `ingest_windows_total` | Counter | Windows emitted by InferenceEngine |
| `alerts_total` | Counter (labeled by severity) | Alerts fired |
| `ingest_errors_total` | Counter | Unhandled errors in /ingest handler |
| `ingest_latency_seconds` | Histogram | End-to-end /ingest handler latency |
| `scoring_latency_seconds` | Histogram | Model scoring latency per window |
| `service_health` | Gauge | 1.0=healthy, 0.5=degraded, 0.0=unhealthy |

### Grafana (`grafana/`)

- Pre-built dashboard: `grafana/dashboards/stage08_api_observability.json` (26 KB)
- Visualizes: ingest throughput, window emission rate, alert counts by severity, latency percentiles, service health gauge, error rates
- Provisioned automatically on container startup via:
  - `grafana/provisioning/dashboards/` — dashboard provider config
  - `grafana/provisioning/datasources/` — Prometheus datasource config
- Default credentials: admin / admin

### Monitoring Flow

```
API Container (/metrics endpoint)
  ← Prometheus (scrapes every 15s)
    → evaluates alert rules
  ← Grafana (queries Prometheus)
    → dashboard panels rendered
```

---

## 9. Deployment and Runtime Setup

### Dockerfile (`docker/Dockerfile`)

- Base image: `python:3.11-slim`
- Installs `curl` for healthcheck
- Installs `requirements/requirements.txt`
- Copies: `src/`, `scripts/`, `templates/`
- Creates empty dirs: `data/intermediate`, `models`, `artifacts`
- Exposed port: 8000
- Healthcheck: `GET http://localhost:8000/health` every 15s, 5s timeout, 30s start delay, 5 retries
- Default CMD: `uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000`

**Note:** `models/` and `artifacts/` are mounted as Docker volumes, not baked into the image. Trained artifacts must be present on the host.

### Docker Compose (`docker/docker-compose.yml`)

**Services:**

| Service | Image | Port | Notes |
|---------|-------|------|-------|
| `api` | built from repo | 8000 | `DEMO_MODE=true`, `WINDOW_SIZE=10`, `STRIDE=1`, `DEMO_WARMUP_ENABLED=true` |
| `prometheus` | `prom/prometheus:v2.51.0` | 9090 | Mounts `prometheus/` config dir, 7d retention |
| `grafana` | `grafana/grafana:10.4.2` | 3000 | Admin pass configurable, mounts provisioning + dashboards |

**Volume mounts on `api` service:**
- `./models` → `/app/models`
- `./artifacts` → `/app/artifacts`
- `./data/intermediate` → `/app/data/intermediate`

**Startup dependencies:**
- `prometheus` depends on `api`
- `grafana` depends on `prometheus`

**Compose-specific env overrides:**
- `WINDOW_SIZE=10`, `STRIDE=1` — smaller windows for faster demo alert generation
- `DEMO_MODE=true` — fallback scoring enabled when models missing
- `DEMO_WARMUP_ENABLED=true` — sends 75 synthetic events on startup to pre-populate alerts

### Running Locally (without Docker)

```bash
pip install -r requirements/requirements.txt
python main.py                          # starts on 0.0.0.0:8000
python main.py --model baseline         # baseline only
python main.py --model ensemble         # default
python main.py --disable-auth           # disables API key check
```

---

## 10. Configuration

All configuration is handled by `src/api/settings.py` (Pydantic settings model). Environment variables take precedence; CLI flags push overrides into env before Settings is instantiated.

**Key environment variables:**

| Variable | Default | Effect |
|----------|---------|--------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `API_KEY` | `""` | API key (empty = auth permissive with warning) |
| `DISABLE_AUTH` | `false` | Bypass auth middleware entirely |
| `PUBLIC_ENDPOINTS` | `/health,/metrics,/,/query` | Comma-separated unauthenticated paths |
| `METRICS_ENABLED` | `true` | Enable Prometheus /metrics endpoint |
| `MODEL_MODE` | `ensemble` | `baseline`, `transformer`, `ensemble`, or `v2` |
| `WINDOW_SIZE` | `50` | Rolling window token count |
| `STRIDE` | `10` | Emit every N events after first full window |
| `ALERT_BUFFER_SIZE` | `200` | Max recent alerts in memory |
| `ALERT_COOLDOWN_SECONDS` | `60` | Minimum seconds between alerts per stream |
| `DEMO_MODE` | `false` | Return synthetic score when models missing |
| `DEMO_SCORE` | `2.0` | Score returned in demo mode (always anomalous) |
| `DEMO_WARMUP_ENABLED` | `false` | Send synthetic events on startup |
| `DEMO_WARMUP_EVENTS` | `75` | Number of warmup events |
| `DEMO_WARMUP_INTERVAL_SECONDS` | `0` | 0 = run once; >0 = repeat interval |
| `N8N_WEBHOOK_URL` | `""` | n8n webhook target (empty = always dry-run) |
| `N8N_DRY_RUN` | `true` | Write alerts to outbox instead of POSTing |
| `N8N_TIMEOUT_SECONDS` | `5` | Webhook request timeout |

---

## 11. Testing and Validation

### Test Structure

```
tests/
├── unit/                           # Fast unit tests (no model files needed)
│   ├── test_sequence_buffer.py     # SequenceBuffer windowing logic
│   ├── test_tokenizer.py           # Log tokenization
│   ├── test_calibrator.py          # Threshold calibration
│   ├── test_synth_generation.py    # Synthetic log generation
│   ├── test_sequences.py           # Sequence dataclass
│   ├── test_log_preprocessor.py    # Log preprocessing (v2/legacy)
│   ├── test_log_dataset.py         # LogDataset wrapper
│   ├── test_behavior_model.py      # LSTM behavior model (v2)
│   ├── test_anomaly_detector.py    # Autoencoder anomaly detector (v2)
│   ├── test_severity_classifier.py # MLP severity classifier (v2)
│   ├── test_proactive_engine.py    # ProactiveEngine (legacy)
│   ├── test_inference_engine_smoke.py  # Inference engine smoke
│   ├── test_runtime_calibration.py # Runtime threshold calibration
│   ├── test_explain_decode.py      # Evidence decode logic
│   └── test_placeholder.py        # Placeholder
├── integration/
│   └── test_smoke_api.py           # FastAPI TestClient smoke
├── system/
│   ├── test_end_to_end_pipeline.py     # Full pipeline E2E
│   ├── test_streaming_simulation.py    # Streaming scenario simulation
│   ├── test_performance_validation.py  # Throughput validation
│   └── test_model_availability_fallback.py  # Fallback behavior when models absent
├── test_stage_06_alert_policy.py   # AlertPolicy rules and severity
├── test_stage_06_dedup_cooldown.py # Alert deduplication + cooldown
├── test_stage_06_n8n_outbox.py     # n8n outbox dry-run integration
├── test_stage_07_ingest_integration.py  # POST /ingest full flow
├── test_stage_07_auth.py           # X-API-Key auth enforcement
├── test_stage_07_metrics.py        # Prometheus metric increments
├── test_pipeline_smoke.py          # Pipeline smoke test
└── helpers_stage_07.py             # MockPipeline, mock risk results
```

### Test Markers

- `slow` — tests requiring trained model files or large data; skipped in CI fast suite
- `integration` — tests using FastAPI `TestClient` for route-level assertions

### CI Test Command

```bash
pytest --tb=short -q -m "not slow"
```

### Coverage Areas

- Rolling window logic: stride/boundary correctness, LRU eviction
- Alert deduplication: cooldown per stream key, severity mapping
- Authentication: key validation, public endpoint bypass
- Prometheus: counter increments on ingest, window emission, errors
- n8n outbox: dry-run JSON file creation, webhook fallback
- Full ingest → alert flow via FastAPI TestClient
- Model fallback behavior: graceful degradation when artifacts missing

---

## 12. CI/CD

**File:** `.github/workflows/ci.yml`

**Triggers:** push to `main` or `dev`; pull request to `main`

### Jobs

#### 1. `tests` — Lint + Unit Tests (Python 3.11)
- Install `requirements/requirements.txt` + `requirements/requirements-dev.txt`
- Flake8 lint: max-line-length=120, informational (exit-zero, non-blocking)
- Run pytest: `pytest --tb=short -q -m "not slow"` (skips model-dependent tests)

#### 2. `security` — depends on `tests`
- `pip-audit`: CVE scan against `requirements/requirements.txt`
- Trivy filesystem scan: HIGH and CRITICAL severity, exit-code=0 (non-blocking)

#### 3. `docker` — depends on `tests`
- Builds image: `docker build -f docker/Dockerfile -t anomaly-api:ci .`
- Starts compose stack: `docker compose -f docker/docker-compose.yml up -d`
- Waits up to 90s for `GET /health` to return HTTP 200
- Smoke tests (sequential):
  1. `GET /health` → HTTP 200
  2. `GET /metrics` → HTTP 200
  3. `POST /ingest` × 10 events → HTTP 200 each; then `GET /alerts` → count ≥ 1
  4. `GET /` → HTTP 200 (demo UI)
  5. `POST /query` → validates `answer` and `sources` keys in response
- Tear down: `docker compose down -v` (always, including on failure)

---

## 13. Practical Runbook

### Minimal steps to run the project

```bash
# Option A: Docker (recommended)
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up

# Option B: Local
pip install -r requirements/requirements.txt
python main.py --disable-auth
```

### Minimal steps to verify it works

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Ingest an event (auth required by default; use --disable-auth or set key)
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"service":"test","token_id":10}'

# View recent alerts
curl http://localhost:8000/alerts

# Check Prometheus metrics
curl http://localhost:8000/metrics | grep ingest
```

### Minimal steps to view alerts / metrics / dashboards

| Resource | URL | Credentials |
|----------|-----|-------------|
| API health | `http://localhost:8000/health` | None |
| Demo UI | `http://localhost:8000/` | None |
| Prometheus | `http://localhost:9090` | None |
| Grafana | `http://localhost:3000` | admin / admin |

Grafana dashboard is auto-provisioned as `stage08_api_observability`. It appears in the Dashboards sidebar after startup.

To trigger demo alerts quickly in DEMO_MODE: post 10+ events to `/ingest` with any `token_id`. With `WINDOW_SIZE=10, STRIDE=1` (compose defaults), the first alert fires after 10 events.

---

## 14. Key Strengths of the Repository

- **Clean separation of concerns:** API, inference, alerting, observability, security, and data processing are each isolated into dedicated modules.
- **Dual pipeline design:** v1 (IsolationForest + Transformer ensemble) and v2 (Word2Vec + LSTM + Autoencoder + Severity MLP) coexist without shared state. Switching between them requires only an environment variable.
- **Demo/production separation:** `DEMO_MODE` flag provides reliable CI behavior without trained artifacts, while production mode fails safe (returns score=0.0 when models are absent).
- **Comprehensive observability:** 7 Prometheus metrics, 4 Prometheus alert rules, pre-built Grafana dashboard, all provisioned automatically in the compose stack.
- **Production-ready deployment patterns:** Docker healthchecks, volume mounts for artifacts, proper ASGI factory pattern, LRU-capped stream buffers.
- **Test coverage breadth:** Unit, integration, and system-level tests covering routing, authentication, metric increments, alert deduplication, cooldown enforcement, and end-to-end pipeline flows.
- **Security awareness:** API key middleware with configurable public endpoints, dry-run-by-default webhook integration, dependency CVE scanning in CI.
- **Self-documenting legacy code:** `ProactiveEngine` is clearly marked `STATUS: LEGACY` in its own source header, preventing confusion about what is active in production.

---

## 15. Known Gaps, Constraints, or Tradeoffs

- **HDFS detection quality:** Baseline model achieves F1=0.047 on the HDFS dataset (as documented in code comments). The system is more effective on BGL-style logs (F1=0.96). This is acknowledged as a dataset-specific limitation, not a bug.
- **Pre-tokenized API input (v1):** `POST /ingest` requires a pre-computed `token_id`, meaning callers must have already run template mining offline. This simplifies the runtime pipeline but requires an out-of-band preprocessing step for real log data. V2 addresses this with raw log string input.
- **In-memory alert buffer only:** Alerts are stored in a `collections.deque` in RAM. No database backend; restarting the service clears all alert history. The n8n outbox provides persistence as a side effect of dry-run mode.
- **Single-process, single-instance:** No horizontal scaling logic, no distributed coordination. Stream buffers and alert cooldown state are per-process.
- **RAG stub:** The `POST /query` endpoint (`src/api/ui.py`) is a hardcoded keyword-match knowledge base, not a real retrieval-augmented generation system.
- **Demo UI is minimal:** `templates/index.html` serves as a demo showcase rather than a production operations console.
- **Auth key has no rotation mechanism:** A single static `API_KEY` env variable with no built-in key rotation or expiry.
- **Security scans are non-blocking:** Both `pip-audit` and Trivy use `exit-code: 0` / `|| true`, meaning discovered vulnerabilities do not fail CI.
- **No rate limiting:** The API has no per-client rate limiting beyond the alert cooldown mechanism (which is anomaly-scoped, not request-scoped).

---

## 16. Final Status Summary

**Complete and working:**
- V1 inference pipeline (baseline, transformer, ensemble modes) with trained artifacts present
- FastAPI REST API with authentication, health checks, and Prometheus metrics
- Alert management with deduplication, cooldown, severity classification
- Docker Compose stack with full Prometheus + Grafana observability
- GitHub Actions CI with lint, tests, security scans, and docker smoke tests
- n8n webhook integration in safe dry-run mode (outbox contains real alert history)
- Comprehensive test suite covering unit, integration, and system tests

**Optional / conditionally active:**
- V2 pipeline (Word2Vec + LSTM + Autoencoder + Severity MLP): model artifacts present; activated by setting `MODEL_MODE=v2`; routes wired in app.py; not the default mode
- Demo warmup task: triggered by `DEMO_WARMUP_ENABLED=true` (set in compose defaults, not in production defaults)
- Runtime calibration: `artifacts/threshold_runtime.json` loaded if present; not required

**Experimental / not wired:**
- `ProactiveEngine` (`src/engine/proactive_engine.py`): self-documented as legacy; present for reference and test coverage; not connected to any API route

**Concluding summary:**
Predictive Log Anomaly Engine v2 is a well-structured, complete system engineering submission demonstrating an end-to-end AIOps anomaly detection pipeline. The v1 inference stack (ensemble of IsolationForest and causal transformer) is the active runtime and is production-packaged with Docker, Prometheus, Grafana, and CI. The v2 neural pipeline (LSTM + Autoencoder + MLP) is fully implemented with trained artifacts and API routes, activated via configuration. The codebase demonstrates strong engineering discipline: clean module boundaries, safe defaults, observable behavior, and thorough test coverage.

---

## Appendix — Most Important Files

| File | Why It Matters |
|------|---------------|
| `src/api/app.py` | Application factory: wires all middleware, routers, and lifespan hooks |
| `src/api/routes.py` | Defines POST /ingest, GET /alerts, GET /health, GET /metrics — the core API surface |
| `src/api/pipeline.py` | Container connecting InferenceEngine + AlertManager + MetricsRegistry |
| `src/api/settings.py` | Single source of truth for all runtime configuration |
| `src/runtime/inference_engine.py` | Active scoring engine: model loading, windowed scoring, ensemble logic, explain |
| `src/runtime/sequence_buffer.py` | Rolling window per stream key; stride/boundary logic |
| `src/runtime/pipeline_v2.py` | V2 four-stage pipeline: tokenizer → Word2Vec → LSTM → Autoencoder → Severity |
| `src/runtime/inference_engine_v2.py` | V2 engine wrapper with alert buffering and cooldown |
| `src/alerts/manager.py` | Alert dedup, cooldown enforcement, severity classification |
| `src/alerts/n8n_client.py` | Safe dry-run webhook client; outbox pattern |
| `src/modeling/baseline/model.py` | IsolationForest wrapper |
| `src/modeling/transformer/model.py` | GPT-style causal transformer architecture |
| `src/modeling/baseline/extractor.py` | 204-feature extractor for baseline model |
| `src/observability/metrics.py` | Prometheus registry + MetricsMiddleware |
| `src/security/auth.py` | X-API-Key middleware |
| `src/engine/proactive_engine.py` | Legacy orchestrator (not wired); reference only |
| `docker/docker-compose.yml` | Complete 3-service stack (api + prometheus + grafana) |
| `docker/Dockerfile` | API container definition with healthcheck |
| `prometheus/prometheus.yml` | Scrape config targeting api:8000 |
| `prometheus/alerts.yml` | ServiceDown, Unhealthy, ErrorRate, Stalled alert rules |
| `grafana/dashboards/stage08_api_observability.json` | Pre-built observability dashboard |
| `.github/workflows/ci.yml` | Full CI pipeline: lint → test → security → docker smoke |
| `models/baseline.pkl` | Trained IsolationForest (v1 active scoring) |
| `models/transformer.pt` | Trained causal transformer (v1 active scoring) |
| `artifacts/vocab.json` | Token ID → template text (used by explain() and v2 tokenizer) |
| `artifacts/threshold.json` | Baseline decision threshold (~0.33) |
| `artifacts/threshold_transformer.json` | Transformer decision threshold (~0.034) |
| `scripts/stage_07_run_api.py` | Primary API server entrypoint with CLI argument handling |
| `main.py` | Project root entrypoint; delegates to stage_07_run_api |
