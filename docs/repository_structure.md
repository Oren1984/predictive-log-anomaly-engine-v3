# Repository Structure — Predictive Log Anomaly Engine V3

## 1. Overview

This repository implements a production-oriented log anomaly detection platform with a FastAPI runtime, ML scoring pipeline(s), alerting, and observability integrations.

At a high level, the system ingests tokenized or raw log events, scores risk using baseline/transformer/v2 components, emits alerts with dedup/cooldown policy, and exposes runtime metrics for Prometheus and Grafana.

## 2. Top-Level Structure

Top-level directories and files are grouped by role:

- `src/` — core application logic (active)
- `scripts/` — operational entrypoints and data-pipeline scripts (active)
- `tests/` — unit/integration/system validation (active)
- `docker/` — Dockerfile and compose orchestration (active)
- `prometheus/` — scrape and alert rule configuration (active)
- `grafana/` — dashboard provisioning and JSON dashboards (active)
- `templates/` — API-served HTML templates (active)
- `training/` — model training entrypoints (active)
- `main.py` — root entrypoint delegating to `scripts/run_api.py` (active)
- `requirements/` — dependency manifests (active)

- `data/` — raw/processed/intermediate datasets used by training/runtime (active + generated)
- `models/` — trained model artifacts loaded by runtime (runtime-required)
- `artifacts/` — runtime assets (thresholds, vocab, templates) plus n8n outbox (runtime-required + generated)
- `reports/` — generated evaluation/runtime reports (generated)

- `notebooks/` — interactive analysis/walkthrough notebooks (optional)
- `demo/` — standalone demo scripts (optional)
- `static-demo/` — static web demo assets (optional)
- `examples/` — integration examples (optional)

- `archive/` — historical code/assets retained for traceability (archive)
- `docs/reports/` and `docs/prompts/` — historical/supporting docs (archive/supporting)

## 3. Core System Breakdown

### `src/api/`

Implements the FastAPI application layer.

- `app.py`: `create_app()` factory, lifecycle startup, middleware wiring, router registration.
- `routes.py`: primary endpoints (`/ingest`, `/alerts`, `/health`, `/metrics`).
- `routes_v2.py`: versioned v2 endpoints (`/v2/ingest`, `/v2/alerts`).
- `routes_v3.py`: versioned v3 endpoints (explanation/model-info/versioned ingest).
- `pipeline.py`: shared runtime container that wires inference, alerting, metrics, and semantic enrichment.
- `settings.py`: environment-driven service configuration.

Why it exists: to provide stable API contracts and own request-time orchestration.

### `src/runtime/`

Implements runtime scoring mechanics.

- `inference_engine.py`: baseline/transformer/ensemble scoring, thresholds, rolling windows.
- `sequence_buffer.py`: per-stream rolling window buffering.
- `pipeline_v2.py`: v2 four-stage ML pipeline from raw log to anomaly+severity.
- `inference_engine_v2.py`: v2 wrapper with cooldown and alert buffering.
- `types.py`: typed risk result payloads.

Why it exists: to isolate streaming inference logic from API transport and persistence concerns.

### `src/modeling/`

Contains model implementations used by runtime and training.

- `baseline/`: feature extractor and baseline anomaly model.
- `transformer/`: next-token transformer model and scorer.
- `behavior_model.py`, `anomaly_detector.py`, `severity_classifier.py`: v2-stage models.
- `embeddings/`: embedding-related helpers.

Why it exists: to centralize model definitions independent of API wiring.

### `src/alerts/`

Alert domain and delivery behavior.

- `models.py`: alert and policy models.
- `manager.py`: dedup + cooldown emission logic.
- `n8n_client.py`: webhook/outbox dispatch (safe dry-run default).

Why it exists: to separate risk scoring from alert lifecycle and integrations.

### `src/observability/`

Prometheus-focused instrumentation.

- `metrics.py`: registry, counters/histograms/gauges, middleware.
- `logging.py`: logging utilities.

Why it exists: to expose measurable runtime health and behavior.

### `src/security/`

Authentication middleware and security controls.

Why it exists: to enforce API key/public-path policy without coupling to business logic.

### `src/semantic/`

V3 semantic/explanation layer.

- `config.py`: `SEMANTIC_ENABLED` and related flags.
- loaders/embedders/similarity/explainer modules for optional enrichment.

Why it exists: to add enrichment and explainability as an additive layer.

### `src/data_layer/`

Data structures and loading utilities consumed by runtime/modeling.

Why it exists: to keep data-access transformations explicit and reusable.

## 4. Execution Flow

Primary runtime startup and request flow:

1. `main.py`
2. delegates to `scripts/run_api.py`
3. launches uvicorn with `src.api.app:create_app`
4. app startup initializes `Pipeline` (`src/api/pipeline.py`) and loads artifacts
5. request enters `src/api/routes.py` (`POST /ingest`)
6. event is passed to `src/runtime/inference_engine.py`
7. `src/alerts/manager.py` emits/suppresses alert according to policy/cooldown
8. `src/alerts/n8n_client.py` writes dry-run outbox JSON or posts webhook
9. `/metrics` exposes Prometheus text format via `src/observability/metrics.py`

## 5. Supporting Layers

- `docker/`: reproducible runtime packaging and local multi-service startup.
  - `docker/Dockerfile` builds API image from repo root context.
  - `docker/docker-compose.yml` wires API + Prometheus + Grafana.

- `prometheus/`:
  - `prometheus.yml` scrape configuration.
  - `alerts.yml` alerting rules.

- `grafana/`:
  - `provisioning/` datasources + dashboard provisioning.
  - `dashboards/` curated dashboards.

- `scripts/`:
  - `run_api.py` service launcher.
  - `data_pipeline/` staged dataset/template/sequence/training prep scripts.

- `tests/`:
  - unit tests for components.
  - integration tests for API behavior.
  - system tests for end-to-end and performance-oriented scenarios.

## 6. Data & Artifacts

### Runtime-required (must exist for full behavior)

- `models/`:
  - `transformer.pt`
  - subfolders for anomaly/behavior/embeddings/severity model assets

- `artifacts/`:
  - `templates.json`, `vocab.json`
  - threshold files (`threshold.json`, `threshold_transformer.json`, optional runtime thresholds)

- `data/intermediate/`:
  - runtime/training intermediate files such as `templates.csv` used by v2 pipeline.

### Generated (rotated/cleaned as needed)

- `reports/`: generated outputs (kept lightweight; archive snapshots under `archive/generated/reports/<date>/`).
- `artifacts/n8n_outbox/`: dry-run webhook payload files (archive snapshots under `archive/generated/n8n_outbox/<date>/`).

### Archive

- `archive/generated/`: timestamped snapshots of generated outputs.

## 7. Optional / Demo / Historical

- `notebooks/`: exploratory and explanatory notebooks (not required for runtime).
- `demo/`: standalone demos for visualization/simulation.
- `static-demo/`: static UI demo assets.
- `archive/`: legacy scripts and historical implementation paths retained for traceability.
- `docs/reports/`: historical report documents.

## 8. Key Files

- `main.py`:
  thin entrypoint that delegates to `scripts/run_api.py`.

- `scripts/run_api.py`:
  CLI/env aware launcher that sets env overrides and runs uvicorn.

- `docker/docker-compose.yml`:
  local stack definition for API + Prometheus + Grafana.

- `docker/Dockerfile`:
  image build recipe using `requirements/requirements.txt`.

- `requirements/requirements.txt` and `requirements/requirements-dev.txt`:
  runtime vs development/test dependency boundaries.

## 9. How to Navigate the Repo

A practical reading order for new contributors:

1. `README.md` for high-level context.
2. `main.py` and `scripts/run_api.py` for startup flow.
3. `src/api/app.py` and `src/api/routes.py` for request lifecycle.
4. `src/api/pipeline.py` for component wiring.
5. `src/runtime/inference_engine.py` and `src/runtime/pipeline_v2.py` for scoring internals.
6. `src/alerts/manager.py` and `src/alerts/n8n_client.py` for alert behavior.
7. `src/observability/metrics.py` for instrumentation.
8. `tests/` for expected behavior contracts.

## 10. Summary

The repository is organized around a clear runtime core (`src/`) with explicit supporting layers for deployment, observability, testing, and model/data assets. Historical and optional materials are kept without polluting active runtime paths, which supports both maintainability and traceability in the current cleaned state.
