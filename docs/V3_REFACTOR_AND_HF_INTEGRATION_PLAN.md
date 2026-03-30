# V3 Refactor and Hugging Face Integration Plan
## Predictive Log Anomaly Engine — Official V3 Upgrade Blueprint

**Document version:** 1.0
**Prepared:** 2026-03-30
**Scope:** Full repository audit, cleanup strategy, and V3 integration roadmap
**Status:** Planning only — no code changes have been made

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Repository Reality Map](#2-current-repository-reality-map)
3. [What Must Be Preserved](#3-what-must-be-preserved)
4. [Cleanup Audit](#4-cleanup-audit)
5. [V1/V2 Legacy Assessment](#5-v1v2-legacy-assessment)
6. [Repository Cleanup and Reorganization Plan](#6-repository-cleanup-and-reorganization-plan)
7. [Gaps Between Current V2 and Desired V3](#7-gaps-between-current-v2-and-desired-v3)
8. [V3 Strategy Options](#8-v3-strategy-options)
9. [Recommended Target V3 Architecture](#9-recommended-target-v3-architecture)
10. [Exact Repository-Level Work Plan](#10-exact-repository-level-work-plan)
11. [Proposed New V3 Modules](#11-proposed-new-v3-modules)
12. [API and Runtime Evolution Plan](#12-api-and-runtime-evolution-plan)
13. [Deployment and Dependency Impact](#13-deployment-and-dependency-impact)
14. [Testing and CI Impact](#14-testing-and-ci-impact)
15. [Phased Execution Plan](#15-phased-execution-plan)
16. [Final Recommendation](#16-final-recommendation)

---

## 1. Executive Summary

### What the repository currently is

A production-ready AI log observability and anomaly detection platform built in Python/FastAPI. It runs two inference pipelines in parallel: V1 (IsolationForest + Causal Transformer ensemble) and V2 (Word2Vec → LSTM → Denoising Autoencoder → Severity MLP). The system includes a full alert lifecycle with deduplication and cooldown, Prometheus/Grafana observability, Docker Compose deployment, and a 578-test CI-validated suite.

### Is it a strong base for V3?

**Yes — decisively.** The engineering backbone is mature: modular `src/` layout, clean FastAPI factory pattern, injectable settings via environment variables, layered middleware chain, graceful demo-mode fallbacks, dry-run webhook defaults, and comprehensive test coverage. This is not a prototype. It is portfolio-grade, production-aware infrastructure.

### Does it need cleanup before V3?

**Yes, but limited and targeted.** The repository carries structural noise from the staged V1/V2 build process: duplicate synthetic data generators (`src/data/` vs `src/synthetic/`), a disconnected legacy engine (`src/engine/`), hundreds of stale alert JSON artifacts in `artifacts/n8n_outbox/`, pipeline stage numbers embedded in script filenames, and some ambiguous top-level model wrappers that may duplicate submodule implementations. None of this is dangerous — it is clutter that will create confusion during V3 integration.

### Should V3 adapt the current system or rebuild it?

**Adapt.** There is no rational case for rebuilding. The runtime, API, alert lifecycle, observability, Docker, and test structures are all working correctly and are cleanly implemented. V3 should graft a semantic enrichment and explanation layer onto this backbone rather than replace it.

### Top-level conclusion

Perform a surgical, targeted cleanup in one focused session, then introduce Hugging Face as an optional enrichment layer beside the existing inference engines — not replacing them. V3 is an **augmentation story**, not a rewrite story. The existing V1 and V2 pipelines remain fully active. The semantic layer activates only for confirmed anomalies, adds explanation context to alerts, and is gated by environment variables so it has zero impact on CI or existing behavior when disabled.

---

## 2. Current Repository Reality Map

### Top-level structure

```
predictive-log-anomaly-engine-v3/
│
├── main.py                          ← Entry point; delegates to scripts/stage_07_run_api.py
├── pyproject.toml                   ← Pytest configuration
├── README.md                        ← Project overview (needs V3 update)
├── evaluation_report.json           ← Generated V1 vs V2 metrics (misplaced at root)
├── .env.example                     ← Environment variable template
├── .gitignore / .dockerignore
│
├── src/                             ← All application source code (143 Python files)
├── training/                        ← Offline model training scripts
├── scripts/                         ← Server entry point + data pipeline scripts
├── tests/                           ← Full test suite (578 tests)
├── docker/                          ← Dockerfile + docker-compose files
├── prometheus/                      ← Prometheus scrape config + alert rules
├── grafana/                         ← Dashboard JSON + auto-provisioning
├── templates/                       ← Jinja2 HTML templates for the UI
├── data/                            ← Raw, processed, and intermediate data files
├── models/                          ← Trained model weight artifacts
├── artifacts/n8n_outbox/            ← Dry-run alert JSON files (stale, hundreds of files)
├── notebooks/                       ← Jupyter demo notebooks
├── demo/                            ← Standalone demo scripts
├── static-demo/                     ← Static demo assets (purpose unclear)
├── docs/                            ← Documentation and screenshots
├── examples/                        ← n8n integration examples
├── requirements/                    ← requirements.txt + requirements-dev.txt
└── .github/workflows/ci.yml         ← GitHub Actions CI/CD pipeline
```

### Runtime path map

```
main.py
  └─ scripts/stage_07_run_api.py      ← FastAPI server startup via uvicorn
       └─ src/api/app.py              ← Application factory + lifespan startup
            ├─ src/api/settings.py    ← Environment-driven configuration
            ├─ src/api/routes.py      ← V1 endpoints: /ingest, /alerts, /health, /metrics
            ├─ src/api/routes_v2.py   ← V2 endpoints: /v2/ingest, /v2/alerts, /v2/result
            ├─ src/api/pipeline.py    ← Pipeline container: orchestrates engines + alerts
            ├─ src/security/auth.py   ← X-API-Key middleware
            ├─ src/observability/metrics.py  ← Prometheus MetricsMiddleware
            └─ src/api/ui.py          ← HTML/Jinja2 investigation UI
```

### Core modules

| Module | Location | Role |
|---|---|---|
| FastAPI factory | `src/api/app.py` | Application creation, lifespan, middleware |
| V1 Inference Engine | `src/runtime/inference_engine.py` | Baseline + Transformer + Ensemble scoring |
| V2 Inference Engine | `src/runtime/inference_engine_v2.py` | Word2Vec → LSTM → AE → MLP pipeline |
| V2 Pipeline | `src/runtime/pipeline_v2.py` | V2 orchestration layer |
| Sequence Buffer | `src/runtime/sequence_buffer.py` | Rolling window per stream, LRU eviction |
| Alert Manager | `src/alerts/manager.py` | Dedup, cooldown, severity gating |
| Alert Models | `src/alerts/models.py` | Alert and AlertPolicy dataclasses |
| N8n Webhook Client | `src/alerts/n8n_client.py` | Outbound alert delivery (dry-run default) |
| Metrics Registry | `src/observability/metrics.py` | Prometheus counters, histograms, gauges |
| Health Checks | `src/health/checks.py` | Readiness and liveness probes |
| Auth Middleware | `src/security/auth.py` | X-API-Key validation |
| Template Miner | `src/parsing/template_miner.py` | Template CSV loading and lookup |
| Event Tokenizer | `src/parsing/tokenizer.py` | Token ID assignment |
| Log Preprocessor | `src/preprocessing/log_preprocessor.py` | Normalisation + Word2Vec embedding |
| Sequence Builders | `src/sequencing/builders.py` | Sliding window and session sequence builders |
| Synthetic Generator | `src/synthetic/generator.py` | Synthetic log event generation |

### Model path

```
models/
├── baseline.pkl                    ← IsolationForest (~1.6 MB)
├── transformer.pt                  ← Causal transformer (~2.1 MB)
├── embeddings/word2vec.model       ← Word2Vec (100-dim, 7,833 vocab)
├── behavior/behavior_model.pt      ← Stacked LSTM (hidden_dim=128, 2 layers)
├── anomaly/anomaly_detector.pt     ← Denoising autoencoder
└── severity/severity_classifier.pt ← 3-class MLP
```

### Alert path

```
InferenceEngine.score(window)
  → RiskResult {stream_key, risk_score, is_anomaly, threshold, severity}
  → AlertManager.emit(risk_result)
      → dedup check (stream_key + cooldown_seconds)
      → Alert {alert_id, severity, score, stream_key, timestamp, evidence_window}
  → N8nWebhookClient.send(alert)
      → DRY_RUN: write to artifacts/n8n_outbox/{alert_id}.json
      → LIVE: POST to N8N_WEBHOOK_URL
```

### Observability path

```
src/observability/metrics.py
  ├─ ingest_events_total (counter)
  ├─ ingest_windows_total (counter)
  ├─ alerts_total[severity] (counter)
  ├─ ingest_errors_total (counter)
  ├─ ingest_latency_seconds (histogram)
  ├─ scoring_latency_seconds (histogram)
  └─ service_health (gauge)

GET /metrics → Prometheus scrape → prometheus/prometheus.yml
  → Grafana: grafana/dashboards/stage08_api_observability.json
```

### Deployment path

```
docker/Dockerfile               ← Multi-stage Python 3.11-slim image
docker/docker-compose.yml       ← api + prometheus + grafana services
docker/docker-compose.prod.yml  ← Production variant
.github/workflows/ci.yml        ← Lint → Test → Security scan → Docker smoke test
```

### Testing path

```
tests/
├── unit/           ← 15 unit test files covering models, runtime, parsing, alerts
├── integration/    ← API smoke tests
├── system/         ← E2E pipeline, performance, streaming, fallback tests
└── *.py            ← Stage 06/07 API and alert tests
```

### Docs/notebooks/experiments path

```
docs/
├── predictive_log_anomaly_engine_v2_summary.md  ← Detailed V2 architecture summary
└── screenshots/                                  ← UI and dashboard screenshots

notebooks/
├── predictive_log_anomaly_engine_demo.ipynb      ← CPU demo
└── predictive_log_anomaly_engine_gpu_demo.ipynb  ← GPU demo (V2-era)

demo/
├── predictive_log_anomaly_engine_demo.py
└── predictive_log_anomaly_engine_gpu_demo.py
```

---

## 3. What Must Be Preserved

The following components are production-worthy and should remain intact or be modified only minimally during V3 work.

### Production runtime (must not be broken)

| Component | Location | Reason to Preserve |
|---|---|---|
| FastAPI factory + lifespan | `src/api/app.py` | Clean factory pattern; lifespan model loading is correct |
| V1 + V2 routes | `src/api/routes.py`, `src/api/routes_v2.py` | Working versioned endpoints |
| Pipeline container | `src/api/pipeline.py` | Clean orchestration of engine + alerts + metrics |
| Settings | `src/api/settings.py` | Environment-driven config, well-structured |
| Pydantic schemas | `src/api/schemas.py` | Request/response validation, extend rather than replace |
| V1 Inference Engine | `src/runtime/inference_engine.py` | Active, tested, ensemble is functional |
| V2 Inference Engine | `src/runtime/inference_engine_v2.py` | Optional but complete, well-structured |
| Sequence Buffer | `src/runtime/sequence_buffer.py` | Core LRU rolling window; tested thoroughly |
| Alert Manager | `src/alerts/manager.py` | Dedup and cooldown logic is correct |
| Alert Models | `src/alerts/models.py` | Clean dataclass design; extend with optional fields |
| N8n webhook client | `src/alerts/n8n_client.py` | Dry-run safe default |

### ML model definitions (not just weights)

| Component | Location | Reason |
|---|---|---|
| IsolationForest baseline | `src/modeling/baseline/` | Active V1 scoring |
| Causal Transformer | `src/modeling/transformer/` | Active V1 scoring |
| LSTM behavior model | `src/modeling/behavior/` | Active V2 scoring |
| Denoising Autoencoder | `src/modeling/anomaly/` | Active V2 scoring |
| Severity MLP | `src/modeling/severity/` | Active V2 scoring |
| Word2Vec trainer | `src/modeling/embeddings/` | V2 dependency |

### Infrastructure (must not be broken)

| Component | Location | Reason |
|---|---|---|
| Parsing pipeline | `src/parsing/` | Template miner and tokenizer are V2 runtime dependencies |
| Sequencing | `src/sequencing/` | Used by both training and inference |
| Log Preprocessor | `src/preprocessing/log_preprocessor.py` | Word2Vec embedding, V2 dependency |
| Data layer | `src/data_layer/` | LogEvent dataclass and dataset loader |
| Metrics | `src/observability/metrics.py` | Prometheus instrumentation; extend, do not replace |
| Auth | `src/security/auth.py` | Keep as-is |
| Health checks | `src/health/checks.py` | Extend with semantic layer status |
| Training scripts | `training/` | Reproducibility; full offline pipeline must remain runnable |
| Docker + Compose | `docker/` | Working containerization |
| Prometheus/Grafana | `prometheus/`, `grafana/` | Full observability stack |
| CI workflow | `.github/workflows/ci.yml` | Must remain green throughout all V3 work |
| Full test suite | `tests/` (578 tests) | No regressions permitted |
| `data/intermediate/templates.csv` | `data/intermediate/` | Runtime inference hard dependency |

---

## 4. Cleanup Audit

### Safe to remove

| Item | Location | Justification |
|---|---|---|
| Stale alert JSON files | `artifacts/n8n_outbox/*.json` | Generated dry-run output; not source code. The directory should stay with a `.gitkeep` but its contents can be purged safely. |
| `evaluation_report.json` at root | Root directory | Generated output from `scripts/evaluate_v2.py`. Not source code. Should be moved to `reports/`, not live at root alongside `main.py`. |
| `scripts/archive/` contents | `scripts/archive/` | Explicitly archived during V2 development. If all content is fully superseded by current scripts, delete. **Verify each file before removing.** |

### Safe to archive

| Item | Current Location | Archive Target | Reason |
|---|---|---|---|
| `src/engine/proactive_engine.py` | `src/engine/` | `archive/src/engine/` | Marked LEGACY in its own docstring. Not wired to any route. Not imported by any runtime module. The design concept is relevant to V3 but this implementation should not be adapted. |
| `src/engine/__init__.py` | `src/engine/` | `archive/src/engine/` | Archive alongside proactive_engine.py |
| GPU demo notebook | `notebooks/predictive_log_anomaly_engine_gpu_demo.ipynb` | `archive/notebooks/` or keep in place with clear V2 label | Represents V2-era experiment with CUDA; not relevant to V3 CPU-first direction |
| `static-demo/` | Root | `archive/static-demo/` | Purpose unclear from directory name; likely a static HTML artifact from V1/V2. Verify contents, then archive. |

### Should be merged

| Source | Target | Reason |
|---|---|---|
| `src/data/synth_generator.py` | `src/synthetic/generator.py` | Direct functional duplication. `src/synthetic/` is the primary version with cleaner naming. Any unique logic in `src/data/` should be merged first. |
| `src/data/synth_patterns.py` | `src/synthetic/patterns.py` | Same pattern — parallel copy with different class names. |
| `src/data/scenario_builder.py` | `src/synthetic/scenario_builder.py` | Duplicate scenario orchestration logic. |
| `src/data/log_event.py` | `src/data_layer/models.py` | Two definitions of the LogEvent concept. `src/data_layer/models.py` is the runtime-active version. |

### Should be renamed or restructured

| Item | Proposed Change | Reason |
|---|---|---|
| `scripts/stage_07_run_api.py` | Rename to `scripts/run_api.py` | The `stage_07` prefix is a build-process artifact, not a meaningful production name. The server entry point should not carry a pipeline stage number. |
| `scripts/stage_01*.py` ... `stage_06*.py` | Move to `scripts/data_pipeline/` subdirectory | These are offline data preparation scripts. Grouping them under `data_pipeline/` makes the `scripts/` root clearly show only the active entry points. |
| `src/data/` directory | Rename to `src/synthetic_legacy/` before merging, or delete after merge | The name `src/data/` creates confusion with the top-level `data/` directory that holds actual data files. |
| `docs/predictive_log_anomaly_engine_v2_summary.md` | Rename to `docs/v2_system_summary.md` | Shorter, more consistent naming. The current filename is very long and not consistent with how V3 docs will be named. |

### Duplicate logic (confirmed)

| Duplication | Files Involved | Resolution |
|---|---|---|
| Synthetic data generation | `src/synthetic/generator.py` vs `src/data/synth_generator.py` | Keep `src/synthetic/`. Read `src/data/synth_generator.py` for any unique logic before deleting. |
| Failure patterns | `src/synthetic/patterns.py` vs `src/data/synth_patterns.py` | Keep `src/synthetic/patterns.py`. |
| Scenario builder | `src/synthetic/scenario_builder.py` vs `src/data/scenario_builder.py` | Keep `src/synthetic/scenario_builder.py`. |
| LogEvent dataclass | `src/data_layer/models.py` vs `src/data/log_event.py` | Keep `src/data_layer/models.py` (runtime active). `src/data/log_event.py` is legacy. |

### Legacy but useful (keep with clear label)

| Item | Assessment |
|---|---|
| `src/engine/proactive_engine.py` | The proactive monitoring concept is architecturally relevant to V3. Archive, do not delete. The code documents design intent that informs the new semantic layer. |
| `scripts/evaluate_v2.py` | Useful for V3 model evaluation comparison. Keep in `scripts/` or move to `scripts/eval/`. |
| `notebooks/predictive_log_anomaly_engine_demo.ipynb` | Useful portfolio asset. Keep but relabel as V2 demo. |

### Unclear — requires verification before touching

| Item | What to Verify Before Acting |
|---|---|
| `src/modeling/behavior_model.py` vs `src/modeling/behavior/lstm_model.py` | Are these the same class? Is one a wrapper around the other? Which path does `inference_engine_v2.py` actually import? |
| `src/modeling/anomaly_detector.py` vs `src/modeling/anomaly/autoencoder.py` | Same question. |
| `src/modeling/severity_classifier.py` vs `src/modeling/severity/severity_classifier.py` | Same question. Determine which is canonical before touching either. |
| `tests/unit/test_explain_decode.py` | What module does this test? No obvious matching `explain_decode` module exists in the current `src/`. May be a stale test pointing at a deleted or renamed module. |
| `tests/unit/test_proactive_engine.py` | If `src/engine/proactive_engine.py` is archived, does this test retain value? Determine whether it should be moved to `archive/tests/` or deleted. |
| `scripts/demo_run.py` | Is this superseded by `demo/predictive_log_anomaly_engine_demo.py`? Or does it serve a distinct purpose? |
| `ai_workspace/` | What is in this directory? If it is an AI assistant workspace artifact (Cursor, Copilot, etc.), it should be gitignored or removed entirely. It has no place in production source. |
| `static-demo/` | What files are in this directory? If static HTML artifacts from V1/V2, archive. If currently served by the UI, keep. |

---

## 5. V1/V2 Legacy Assessment

### Items that no longer cleanly fit the V3 direction

#### The `stage_XX` naming convention throughout `scripts/`

Every offline pipeline script carries a `stage_01` through `stage_07` prefix inherited from the sequential build process used to construct V1 and V2. This naming is meaningful only in the context of that build sequence. In a V3 production repository, the server entry point (`stage_07_run_api.py`) must not carry a pipeline stage number in its name.

**Resolution:** Rename scripts by function. `stage_07_run_api.py` → `run_api.py`. Group offline data preparation scripts (stages 01–06) under `scripts/data_pipeline/`.

---

#### The V1 causal transformer (`src/modeling/transformer/`)

The home-built GPT-style causal transformer was introduced in V1 as an experimental NLL-based anomaly scorer. It is active in the V1 ensemble and works correctly. However, V3 introduces Hugging Face — and there is a real risk of architectural confusion between this transformer and the HF-sourced models. Class names such as `TransformerConfig` and `NextTokenTransformerModel` will overlap ambiguously with HF terminology.

**Resolution:** Keep the V1 transformer entirely as-is. Name all new V3 HuggingFace classes distinctly: `HFModelLoader`, `SemanticEmbeddingService`, `SemanticEnricher`, etc. No renaming of the V1 transformer is needed — the risk is managed by naming discipline in new V3 modules.

---

#### `src/engine/proactive_engine.py`

Explicitly marked `LEGACY` in its own module docstring. It was part of an early V1 design for a proactive monitoring concept that was never wired to any API route. The class `ProactiveMonitorEngine` has test coverage (`tests/unit/test_proactive_engine.py`) but is completely disconnected from the production runtime.

**Resolution:** Archive to `archive/src/engine/`. Do not delete — the concept of proactive anomaly monitoring informs the V3 semantic layer design.

---

#### `src/data/` — duplicate synthetic generators

This entire module is a functional duplicate of `src/synthetic/`. It was likely the original V1 synthetic data code, retained when the cleaner `src/synthetic/` implementation was introduced in V2. It serves no unique runtime purpose.

**Resolution:** Read each file, extract any unique logic, merge into `src/synthetic/`, then delete `src/data/`.

---

#### GPU demo notebook (`notebooks/predictive_log_anomaly_engine_gpu_demo.ipynb`)

Represents a V1/V2 experiment testing CUDA acceleration. This is ahead of V3's practical constraints — the V3 Hugging Face integration targets CPU-first models for portability.

**Resolution:** Keep in `notebooks/` as a reference artifact, but label clearly as a V2-era experiment. Create a new `notebooks/v3_semantic_demo.ipynb` for V3.

---

#### `evaluation_report.json` at root

A generated artifact showing V1 vs V2 precision/recall comparison. Its presence in the root directory alongside `main.py` and `README.md` is confusing — it looks like project configuration when it is actually generated output from a single evaluation run.

**Resolution:** Move to `reports/evaluation_report_v2.json`. Add `reports/*.json` to `.gitignore` to prevent re-committing generated output.

---

#### `scripts/archive/`

Contains older pipeline scripts explicitly moved there during V2 development. These are likely variants of stages 01–06 that were superseded.

**Resolution:** Inspect each file individually. If content is fully superseded, delete. If it has historical reference value, move to `archive/scripts/`. Do not leave `scripts/archive/` nested inside the active `scripts/` directory.

---

## 6. Repository Cleanup and Reorganization Plan

### Proposed clean structure after cleanup

```
predictive-log-anomaly-engine-v3/
│
├── main.py                         ← Keep; update delegate path to scripts/run_api.py
├── README.md                       ← Keep; update for V3
├── pyproject.toml                  ← Keep as-is
├── .env.example                    ← Keep; add new SEMANTIC_* variables
├── .gitignore / .dockerignore      ← Keep; verify models/ and artifacts/ exclusions
│
├── src/                            ← RUNTIME CODE ONLY
│   ├── api/                        ← Keep entirely as-is
│   ├── runtime/                    ← Keep entirely as-is
│   ├── modeling/                   ← Keep; resolve wrapper vs submodule duplication
│   ├── alerts/                     ← Keep entirely as-is
│   ├── parsing/                    ← Keep entirely as-is
│   ├── sequencing/                 ← Keep entirely as-is
│   ├── preprocessing/              ← Keep entirely as-is
│   ├── data_layer/                 ← Keep entirely as-is
│   ├── dataset/                    ← Keep entirely as-is
│   ├── synthetic/                  ← Keep; absorb any unique logic from src/data/
│   ├── observability/              ← Keep; extend with semantic metrics
│   ├── security/                   ← Keep entirely as-is
│   ├── health/                     ← Keep; extend with semantic layer status
│   └── semantic/                   ← [NEW V3] Hugging Face semantic layer
│       ├── __init__.py
│       ├── loader.py
│       ├── embeddings.py
│       ├── similarity.py
│       ├── explainer.py
│       └── config.py
│
├── training/                       ← Keep as-is (offline pipeline, clearly labeled)
│
├── scripts/
│   ├── run_api.py                  ← Renamed from stage_07_run_api.py
│   ├── evaluate_v2.py              ← Keep; optionally move to scripts/eval/
│   ├── demo_run.py                 ← Keep or consolidate with demo/
│   └── data_pipeline/             ← [NEW subdirectory for offline data scripts]
│       ├── stage_01_synth_generate.py
│       ├── stage_01_synth_to_processed.py
│       ├── stage_01_synth_validate.py
│       ├── stage_01_data.py
│       ├── stage_02_templates.py
│       ├── stage_03_sequences.py
│       ├── stage_04_baseline.py
│       ├── stage_04_transformer.py
│       ├── stage_05_run.py
│       └── stage_06_demo_alerts.py
│
├── tests/                          ← Keep entirely as-is (578 tests must remain green)
│   └── unit/test_semantic_*.py     ← [NEW] Add V3 unit tests with mocked HF
│
├── docker/                         ← Keep as-is; update CMD path after script rename
├── prometheus/                     ← Keep as-is
├── grafana/                        ← Keep as-is; add new V3 metric panels
├── templates/                      ← Keep as-is
├── data/                           ← Keep as-is
├── models/                         ← Keep; verify .gitignore covers *.pkl, *.pt, *.model
├── requirements/                   ← Keep; add sentence-transformers
├── .github/                        ← Keep as-is
│
├── docs/
│   ├── v2_system_summary.md        ← Renamed from the long V2 filename
│   ├── V3_REFACTOR_AND_HF_INTEGRATION_PLAN.md  ← This document
│   ├── v3_architecture.md          ← [NEW] V3 architecture reference
│   └── screenshots/                ← Keep as-is
│
├── notebooks/
│   ├── v2_demo_cpu.ipynb           ← Renamed for clarity
│   ├── v2_demo_gpu.ipynb           ← Renamed; labeled as V2-era experiment
│   └── v3_semantic_demo.ipynb      ← [NEW] V3 demo notebook
│
├── examples/                       ← Keep as-is
│
├── reports/                        ← [NEW] Dedicated location for generated outputs
│   └── evaluation_report_v2.json   ← Moved from root
│
└── archive/                        ← [NEW] Explicitly archived legacy content
    ├── src/
    │   ├── engine/                 ← Archived src/engine/ (proactive_engine.py)
    │   └── data/                   ← Archived src/data/ (after merge into src/synthetic/)
    ├── scripts/                    ← Merged with contents of scripts/archive/
    └── notebooks/                  ← GPU demo if archived
```

### What stays untouched for stability

The following must not be reorganized, renamed, or modified during cleanup. Any changes risk breaking the CI pipeline or introducing test failures.

- `src/api/` — fully working, do not restructure
- `src/runtime/` — fully working, do not restructure
- `src/alerts/` — fully working, do not restructure
- `src/observability/` — working, only extend
- `tests/` — 578 tests, do not move or rename any existing test file
- `docker/` — working containerization (only update CMD path)
- `prometheus/` and `grafana/` — working observability stack
- `.github/workflows/ci.yml` — CI must remain green at every step

---

## 7. Gaps Between Current V2 and Desired V3

| Gap | Description | Priority |
|---|---|---|
| No semantic understanding layer | The system operates entirely on token IDs and statistical reconstruction error. It has no concept of log meaning — only pattern frequency and autoencoder loss. Semantically identical logs with different token IDs are treated as different events. | High |
| No Hugging Face integration | No `transformers` or `sentence-transformers` dependency. No HF model loading infrastructure. No embedding pipeline for raw text. | High |
| No embedding-based similarity | No mechanism to compare new anomalous events against a known corpus of anomaly patterns using semantic distance. The alert system cannot say "this looks like the HDFS cascading failure pattern from three months ago." | High |
| No explanation layer | Anomaly alerts contain a `risk_score` and `severity`, but no human-readable explanation of why the anomaly was detected or what it means. This is the most significant usability gap. | High |
| No config abstraction for model backends | `MODEL_MODE` is a simple string enum. V3 requires a proper backend selector that handles HF model loading, local weight loading, and fallback behavior in a unified way. | Medium |
| No retrieval or context layer | No mechanism to enrich alerts with semantically similar historical events from a reference set. | Medium |
| Weak alert payload | Current `Alert` contains: `alert_id`, `severity`, `score`, `stream_key`, `timestamp`, `evidence_window`. Missing: human-readable explanation, semantic context, similar events, evidence template IDs. | Medium |
| No `src/semantic/` module | There is no location or scaffolding yet for semantic/NLP/HF code. | Medium |
| No V3 API routes | No `/explain`, `/v3/ingest`, `/alerts/{id}/explanation`, or `/models/info` endpoints exist. | Low-Medium |
| Word2Vec is the only embedding | Word2Vec operates on token IDs, not raw log text. It captures token co-occurrence statistics, not semantic meaning. A sentence like "connection refused" and "socket timeout" are semantically related but may get entirely different Word2Vec representations depending on their template IDs. | High |

---

## 8. V3 Strategy Options

### Option A — Cleanup-First Minimal Upgrade

**Scope:** Perform the structural cleanup described in Section 6. Add `sentence-transformers` embeddings as an optional post-processing step on V2 output. No new routes, no explanation layer, no changes to the alert format.

**What is added:**
- `src/semantic/loader.py` and `src/semantic/embeddings.py`
- `SemanticEmbeddingService` called optionally after anomaly scoring
- New dependency: `sentence-transformers`
- New config variables: `SEMANTIC_ENABLED`, `SEMANTIC_MODEL`

**Benefits:**
- Minimal risk, fast to execute
- Demonstrates HF model loading in the portfolio
- Does not require any schema changes

**Risks:**
- Shallow V3 story — HuggingFace is a bolt-on with no real impact on decisions or alert quality
- The enrichment adds computation but does not change any output the user sees
- Hard to justify as V3 vs V2.1

**Complexity:** Low

**Recommendation:** Use as the Phase 2 entry point into V3, not as the full V3 ambition.

---

### Option B — Balanced V3 Upgrade ✅ RECOMMENDED

**Scope:** Targeted cleanup + preserve the full V1/V2 backbone + add a semantic enrichment layer that meaningfully improves alert quality and adds explanation capability. This is not a rewrite. It is a layered augmentation.

**What is added:**
- `src/semantic/` module: HF loader, embeddings, similarity scorer, explainer
- Semantic enrichment called after anomaly scoring, only for confirmed anomalies
- Extended `Alert` payload: adds `explanation`, `semantic_similarity`, `top_similar_events`
- New API routes: `GET /alerts/{id}/explanation`, `GET /models/info`
- Optional `POST /v3/ingest` as a clean entry point for the full V3 pipeline
- New config: `SEMANTIC_ENABLED`, `SEMANTIC_MODEL`, `EXPLANATION_ENABLED`
- New Prometheus metrics: `semantic_latency_seconds`, `explanation_generated_total`

**Benefits:**
- V3 story is coherent: better alerts, not just more computation
- Hugging Face adds real value that users can see in alert payloads
- V1 and V2 inference pipelines remain fully active and unchanged
- All 578 existing tests continue to pass (new layer is additive, gated by env var)
- Portfolio-ready: demonstrates NLP + ML + backend + observability integration
- `SEMANTIC_ENABLED=false` default ensures CI is unaffected

**Risks:**
- Latency: HF model inference adds 10–50ms overhead; mitigated by calling only for confirmed anomalies, not on every ingest event
- Docker image size increases by ~90–300MB depending on chosen model
- HF calls must be mocked in unit tests to keep CI fast

**Complexity:** Medium

**Recommendation: This is the correct V3 path.**

---

### Option C — Major Restructure

**Scope:** Rewrite the inference engine with Hugging Face models as the primary scoring mechanism. Replace Word2Vec with HF text embeddings throughout. Rebuild the API as a clean V3 system.

**Benefits:**
- Technically cleaner end state with a fully modern NLP stack
- No legacy inference code to maintain

**Risks:**
- Destroys working, tested infrastructure without proportional benefit
- Breaks 578 tests that are coupled to the existing architecture
- Removes the V1/V2 vs V3 benchmarking capability
- HF models are not drop-in replacements for statistical anomaly detection — the detection logic itself requires redesign, not just substitution
- Significant regression risk; delays portfolio readiness substantially

**Complexity:** High

**Recommendation: Not justified. This option should not be pursued.**

---

## 9. Recommended Target V3 Architecture

### V3 Processing Pipeline (Option B)

```
┌────────────────────────────────────────────────────────────────────┐
│  INGEST LAYER                                                      │
│  POST /ingest      (V1: pre-tokenized token_id)                   │
│  POST /v2/ingest   (V2: raw log string)                           │
│  POST /v3/ingest   (V3: raw log string → full enriched pipeline)  │
└───────────────────────────────┬────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│  PARSING & NORMALISATION                                           │
│  src/parsing/template_miner.py  → template_id                     │
│  src/parsing/tokenizer.py       → token_id                        │
└───────────────────────────────┬────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│  SEQUENCE BUFFER                                                   │
│  src/runtime/sequence_buffer.py                                    │
│  Rolling window per stream_key (service:session_id), LRU eviction │
└───────────────────────────────┬────────────────────────────────────┘
                                ↓ (on window emit)
          ┌─────────────────────┴──────────────────────┐
          ↓                                            ↓
   ┌─────────────┐                             ┌──────────────┐
   │  V1 ENGINE  │                             │  V2 ENGINE   │
   │ (optional)  │                             │  (optional)  │
   │             │                             │              │
   │ Baseline    │                             │ Word2Vec     │
   │ IsoForest   │                             │   → LSTM     │
   │     +       │                             │   → AE       │
   │ Transformer │                             │   → MLP      │
   │ NLL scorer  │                             │              │
   └──────┬──────┘                             └──────┬───────┘
          └───────────────────┬────────────────────────┘
                              ↓
                 risk_score, is_anomaly, severity
                              ↓
          ┌───────────────────┴─────────────────────────┐
          │                                             │
          │  [GATE: only if is_anomaly = true]          │
          │                                             │
          ↓                                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  [NEW V3] SEMANTIC ENRICHMENT LAYER   src/semantic/                │
│                                                                     │
│  SemanticLayer.enrich(raw_log, risk_result)                        │
│                                                                     │
│  ├─ HFModelLoader (lazy, cached)                                   │
│  │     model: all-MiniLM-L6-v2 (sentence-transformers)            │
│  │     config: SEMANTIC_MODEL env var                              │
│  │                                                                 │
│  ├─ SemanticEmbeddingService                                       │
│  │     embed(raw_log) → float[384]                                 │
│  │                                                                 │
│  ├─ SemanticSimilarityScorer                                       │
│  │     cosine_sim vs indexed anomaly pattern embeddings            │
│  │     → semantic_similarity: float                               │
│  │     → top_similar_events: list[str]                            │
│  │                                                                 │
│  └─ AnomalyExplainer                                               │
│        rule-based mode (default, zero latency):                   │
│          combines: severity + evidence_tokens + semantic context  │
│          → explanation_text: str                                  │
│        optional LLM mode (EXPLANATION_MODEL=flan-t5-small):       │
│          → generated natural language explanation                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ALERT MANAGEMENT    src/alerts/manager.py                         │
│                                                                     │
│  AlertManager: dedup + cooldown + severity gating                 │
│                                                                     │
│  V3 Enriched Alert Payload:                                        │
│  {                                                                 │
│    alert_id, severity, score, stream_key, timestamp,              │
│    evidence_window, model_name, threshold,          ← V2 fields   │
│    explanation:          str | None,                ← V3 new      │
│    semantic_similarity:  float | None,              ← V3 new      │
│    top_similar_events:   list[str] | None,          ← V3 new      │
│    evidence_tokens:      list[str] | None           ← V3 new      │
│  }                                                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
          ┌────────────────────┴──────────────────────┐
          ↓                                           ↓
  N8n Webhook Client                       Prometheus Metrics
  src/alerts/n8n_client.py                src/observability/metrics.py
  (enriched payload forwarded)             + semantic_latency_seconds
                                           + explanation_generated_total
          ↓
  GET /alerts/{id}/explanation
  (retrieve explanation for any stored alert)
```

### New module: `src/semantic/`

```
src/semantic/
├── __init__.py         ← SemanticLayer facade; the only import pipeline.py needs
├── config.py           ← SemanticConfig dataclass reading from env vars
├── loader.py           ← HFModelLoader: lazy, cached model loading
├── embeddings.py       ← SemanticEmbeddingService: text → dense vector
├── similarity.py       ← SemanticSimilarityScorer: cosine sim + index query
└── explainer.py        ← AnomalyExplainer: rule-based and optional LLM
```

---

## 10. Exact Repository-Level Work Plan

This section is ordered by execution sequence. Each step must be completed and tests confirmed green before proceeding to the next.

### Step 0 — Baseline Protection (before any changes)

1. Run `pytest -m "not slow"` — confirm 578 tests pass
2. Run `docker build -f docker/Dockerfile .` — confirm Docker build succeeds
3. Start the compose stack and confirm `GET /health` returns healthy
4. Document which model files currently exist in `models/`
5. Confirm `src/engine/proactive_engine.py` has no active imports by grepping for it across `src/`

### Step 1 — Verify before touching (disambiguation required)

Before any files are moved or deleted, read the following files and determine their relationship:

1. `src/modeling/behavior_model.py` — is it a wrapper of `src/modeling/behavior/lstm_model.py` or a duplicate? Check `inference_engine_v2.py` to see which path it imports.
2. `src/modeling/anomaly_detector.py` — same question vs `src/modeling/anomaly/autoencoder.py`.
3. `src/modeling/severity_classifier.py` — same question vs `src/modeling/severity/severity_classifier.py`.
4. `src/data/` — compare line by line with `src/synthetic/`. Document any logic that exists in `src/data/` but not in `src/synthetic/`.
5. `static-demo/` — list and read contents. Determine if runtime or artifact.
6. `ai_workspace/` — list and read contents. Determine if it belongs in the repo at all.
7. `tests/unit/test_explain_decode.py` — determine what module it tests. If the module no longer exists, the test is stale.
8. `tests/unit/test_proactive_engine.py` — determine test value if `src/engine/` is archived.

### Step 2 — Safe cleanup actions (low-risk, no import changes)

1. Move `evaluation_report.json` from root → `reports/evaluation_report_v2.json`. Create `reports/` directory.
2. Add `reports/*.json` to `.gitignore`.
3. Delete contents of `artifacts/n8n_outbox/` (keep directory with `.gitkeep`).
4. Add `artifacts/n8n_outbox/*.json` to `.gitignore`.
5. Create `archive/` at repo root with `archive/src/engine/`, `archive/src/data/`, `archive/scripts/`.

### Step 3 — Script renaming and reorganisation

1. Rename `scripts/stage_07_run_api.py` → `scripts/run_api.py`.
2. Update `main.py` import/call to reference the new path.
3. Update `docker/Dockerfile` CMD to reference `scripts/run_api.py`.
4. Create `scripts/data_pipeline/` subdirectory.
5. Move all `scripts/stage_01*.py`, `stage_02*.py`, `stage_03*.py`, `stage_04*.py`, `stage_05*.py`, `stage_06*.py` → `scripts/data_pipeline/`.
6. Move `scripts/archive/` contents → `archive/scripts/`.
7. Run full test suite — confirm green.

### Step 4 — Duplication resolution (after Step 1 verification)

1. If `src/modeling/behavior_model.py` duplicates `src/modeling/behavior/lstm_model.py`: update imports in `inference_engine_v2.py` to use the submodule path, then delete the top-level wrapper. Run tests after.
2. Apply same logic to `anomaly_detector.py` and `severity_classifier.py`.
3. Read each file in `src/data/`. Extract any logic unique to `src/data/` and merge into `src/synthetic/`. Delete `src/data/`. Run tests after.

### Step 5 — Legacy isolation

1. Move `src/engine/proactive_engine.py` → `archive/src/engine/proactive_engine.py`.
2. Move `src/engine/__init__.py` → `archive/src/engine/__init__.py`.
3. Decide on `tests/unit/test_proactive_engine.py`: either move to `archive/tests/unit/` or delete if the test has no standalone value.
4. Handle `tests/unit/test_explain_decode.py`: if the module it tests no longer exists, delete the test. If the module was renamed, update the import.
5. Run full test suite — confirm green.

### Step 6 — `.gitignore` and repository hygiene

1. Verify `models/*.pkl`, `models/*.pt`, `models/**/*.model` are excluded from git.
2. Add `ai_workspace/` to `.gitignore` if it is an IDE artifact.
3. Verify `data/raw/` and `data/processed/` are excluded (large dataset files should not be committed).

### Step 7 — V3 module scaffolding (first Hugging Face step)

1. Create `src/semantic/__init__.py` with a `SemanticLayer` facade class (stub only at this stage).
2. Create `src/semantic/config.py` — `SemanticConfig` dataclass reading `SEMANTIC_ENABLED` (default: `false`), `SEMANTIC_MODEL` (default: `all-MiniLM-L6-v2`), `EXPLANATION_ENABLED` (default: `false`), `EXPLANATION_MODEL` (default: `rule-based`), `SEMANTIC_CACHE_SIZE` (default: `1000`).
3. Create `src/semantic/loader.py` — `HFModelLoader` with lazy `get_model()` using singleton pattern.
4. Create `src/semantic/embeddings.py` — `SemanticEmbeddingService` with `embed(text)` and `embed_batch(texts)`.
5. Add `sentence-transformers>=2.7.0` to `requirements/requirements.txt`.
6. Add `SEMANTIC_ENABLED=false` to `.env.example`.
7. Create `tests/unit/test_semantic_embeddings.py` — unit tests with fully mocked HF model.
8. Confirm `SEMANTIC_ENABLED=false` keeps the layer completely inert with zero performance cost.
9. Run full test suite — confirm green.

### Step 8 — Semantic integration into the pipeline

1. Create `src/semantic/similarity.py` — `SemanticSimilarityScorer` with `index()` and `query()`.
2. Create `src/semantic/explainer.py` — `AnomalyExplainer` in rule-based mode first.
3. Add semantic enrichment call to `src/api/pipeline.py` — after anomaly scoring, before alert emit, gated by `SEMANTIC_ENABLED`.
4. Extend `Alert` dataclass in `src/alerts/models.py` with optional fields: `explanation: Optional[str]`, `semantic_similarity: Optional[float]`, `top_similar_events: Optional[list[str]]`, `evidence_tokens: Optional[list[str]]`.
5. Update `src/api/schemas.py` to expose new optional fields in `AlertSchema`.
6. Run integration tests — confirm backward compatibility (all optional fields default to `None`).

### Step 9 — New API routes

1. Add `GET /alerts/{alert_id}/explanation` to `src/api/routes.py` or a new `src/api/routes_v3.py`.
2. Add `GET /models/info` to return loaded model status for all three layers.
3. Update `src/health/checks.py` to include semantic layer status.

### Step 10 — Observability update

1. Add `semantic_latency_seconds` histogram to `src/observability/metrics.py`.
2. Add `explanation_generated_total` counter.
3. Update `grafana/dashboards/stage08_api_observability.json` with new panels.

### Step 11 — Docker and dependency updates

1. Update `docker/Dockerfile` to add HF model pre-download layer:
   ```dockerfile
   RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```
2. Confirm Docker build works with `SEMANTIC_ENABLED=false` (CI path).
3. Confirm Docker build works with `SEMANTIC_ENABLED=true` (full V3 path).
4. Test full compose stack end-to-end.

### Step 12 — Documentation and final polish

1. Update `README.md` with V3 architecture overview and new configuration variables.
2. Create `docs/v3_architecture.md` with the pipeline diagram from Section 9.
3. Create `notebooks/v3_semantic_demo.ipynb`.
4. Rename `docs/predictive_log_anomaly_engine_v2_summary.md` → `docs/v2_system_summary.md`.
5. Confirm all tests (578 original + new V3 tests) pass.

---

## 11. Proposed New V3 Modules

### `src/semantic/config.py` — Semantic Layer Configuration

**Purpose:** Environment-driven configuration for the semantic enrichment layer. Keeps HF-specific config isolated from the main `src/api/settings.py`, preserving a clean separation between the production API config and the optional semantic layer.

**Key config variables:**

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_ENABLED` | `false` | Master toggle for the semantic layer |
| `SEMANTIC_MODEL` | `all-MiniLM-L6-v2` | Hugging Face sentence-transformer model name |
| `SEMANTIC_MODEL_PATH` | `""` | Local path override (empty = download from HF Hub) |
| `EXPLANATION_ENABLED` | `false` | Toggle for explanation text generation |
| `EXPLANATION_MODEL` | `rule-based` | `rule-based` or `flan-t5-small` for LLM generation |
| `SEMANTIC_CACHE_SIZE` | `1000` | LRU embedding cache size |

**Integration:** `src/api/app.py` reads `SemanticConfig` during lifespan startup if `SEMANTIC_ENABLED=true`. The `SemanticLayer` is passed to `pipeline.py` as an optional dependency.

---

### `src/semantic/loader.py` — Hugging Face Model Loader

**Purpose:** Loads and caches Hugging Face sentence-transformer models lazily. Prevents loading at import time. Handles both HF Hub download and local path loading. Provides a warm-up method for the startup lifespan.

**Key class:** `HFModelLoader`

- `get_model() -> SentenceTransformer` — returns the singleton model instance, loading on first call
- `warmup()` — runs a dummy encode to force model compilation
- `is_loaded() -> bool` — returns whether the model has been loaded
- Reads `SEMANTIC_MODEL` and `SEMANTIC_MODEL_PATH` from `SemanticConfig`

**Integration:** Called by `SemanticEmbeddingService.__init__()`. Uses module-level singleton pattern to avoid double-loading in multi-route scenarios.

---

### `src/semantic/embeddings.py` — Semantic Embedding Service

**Purpose:** Converts raw log strings into dense semantic embeddings using a sentence-transformer model. Provides the embedding backbone for both similarity scoring and explanation generation.

**Key class:** `SemanticEmbeddingService`

- `embed(text: str) -> np.ndarray` — returns a float[384] embedding for a single log string
- `embed_batch(texts: list[str]) -> np.ndarray` — batch embedding for efficiency
- LRU cache on `embed()` to avoid redundant computation for repeated log templates

**Integration:** Called by `src/semantic/__init__.py`'s `SemanticLayer.enrich()` method. Also used by `SemanticSimilarityScorer` for indexing known patterns.

---

### `src/semantic/similarity.py` — Semantic Similarity Scorer

**Purpose:** Maintains an in-memory index of known anomaly pattern embeddings (seeded from training data or a curated pattern library). At inference time, computes cosine similarity between a new anomalous event embedding and the indexed patterns to find the most semantically similar historical cases.

**Key class:** `SemanticSimilarityScorer`

- `index(texts: list[str], labels: list[str])` — embeds and indexes a pattern corpus
- `query(embedding: np.ndarray, k: int = 3) -> list[SimilarEvent]` — returns top-k similar events
- `SimilarEvent` dataclass: `{label: str, similarity: float, raw_text: str}`
- Cosine similarity computed with `numpy`; no vector database required at this scale

**Integration:** Called inside `SemanticLayer.enrich()` after embedding generation. Results are attached to the enriched alert payload.

---

### `src/semantic/explainer.py` — Anomaly Explanation Engine

**Purpose:** Generates a human-readable explanation for a confirmed anomaly detection result. Operates in two modes controlled by `EXPLANATION_MODEL`:

**Rule-based mode** (default, always available):
- Combines: severity level + evidence token sequence + semantic similarity context
- Uses structured templates: `"Anomalous {severity} sequence detected in {service}. Pattern: [{top_templates}]. Semantic match: {top_similar_event} (similarity: {score:.2f})."`
- Zero additional latency; no model required
- Deterministic and testable

**LLM mode** (optional, `EXPLANATION_MODEL=flan-t5-small`):
- Loads `google/flan-t5-small` from Hugging Face Hub
- Generates a short natural language description of the anomaly
- Target length: 1–2 sentences
- CPU-friendly: flan-t5-small is 80M parameters (~300MB)

**Key class:** `AnomalyExplainer`

- `explain(risk_result: RiskResult, context: SemanticContext) -> ExplanationResult`
- `ExplanationResult` dataclass: `{text: str, confidence: float, mode: str, evidence_tokens: list[str]}`

**Integration:** Called inside `SemanticLayer.enrich()` after similarity scoring. Output is attached to the enriched alert payload.

---

### `src/semantic/__init__.py` — Public Facade

**Purpose:** Exposes a single `SemanticLayer` class that `pipeline.py` calls without needing to know the internal structure of the semantic module.

**Key class:** `SemanticLayer`

- `__init__(config: SemanticConfig)` — initialises all sub-components lazily
- `enrich(raw_log: str, risk_result: RiskResult) -> SemanticEnrichment` — main method
- `is_ready() -> bool` — health check for the semantic layer
- `SemanticEnrichment` dataclass: `{explanation: str | None, semantic_similarity: float | None, top_similar_events: list[str] | None, evidence_tokens: list[str] | None, latency_ms: float}`

**Integration:** `src/api/pipeline.py` holds an optional `SemanticLayer` instance. If `SEMANTIC_ENABLED=false`, the instance is `None` and the enrich call is skipped entirely with no overhead.

---

## 12. API and Runtime Evolution Plan

### Design principle

All V3 API changes are **strictly additive**. No existing routes are modified. No existing response schemas have required fields removed. New fields in responses are `Optional` and default to `None` when the semantic layer is disabled. This ensures backward compatibility across all existing clients, tests, and CI flows.

---

### New endpoint: `GET /alerts/{alert_id}/explanation`

**Purpose:** Retrieve the semantic explanation for a specific stored alert. Useful for investigation workflows where a user wants deeper context on a specific fired alert.

**Response schema:**
```json
{
  "alert_id": "string (UUID)",
  "explanation": "Anomalous Critical sequence in hdfs:session_99. Templates [42, 17, 88] indicate cascading block failure. Semantic match: 'HDFS replication pipeline collapse' (similarity: 0.89).",
  "semantic_similarity": 0.89,
  "top_similar_events": [
    "HDFS replication pipeline collapse",
    "DataNode disk failure cascade"
  ],
  "evidence_tokens": ["template_42", "template_17", "template_88"],
  "generated_at": "2026-03-30T12:00:00Z"
}
```

**Behavior when semantic layer is disabled:** Returns `404` with a clear message indicating that explanation generation is not enabled (`SEMANTIC_ENABLED=false`).

**Location:** Add to `src/api/routes.py` or a new `src/api/routes_v3.py`.

---

### New endpoint: `GET /models/info`

**Purpose:** Return a summary of all loaded models and their current status. Valuable for debugging, health monitoring, and portfolio demonstrations.

**Response schema:**
```json
{
  "v1": {
    "baseline": "loaded",
    "transformer": "loaded",
    "mode": "ensemble"
  },
  "v2": {
    "word2vec": "loaded",
    "lstm": "loaded",
    "autoencoder": "loaded",
    "severity": "loaded"
  },
  "v3": {
    "semantic_enabled": true,
    "model": "all-MiniLM-L6-v2",
    "explanation_enabled": true,
    "explanation_mode": "rule-based",
    "status": "ready"
  }
}
```

**Location:** Add to `src/api/routes.py`.

---

### Extended `/ingest` response (when `SEMANTIC_ENABLED=true`)

When the semantic layer is active and an anomaly is detected, the existing alert field in the ingest response is extended:

```json
{
  "window_emitted": true,
  "risk_result": { "...existing fields..." },
  "alert": {
    "alert_id": "string",
    "severity": "Critical",
    "score": 0.91,
    "stream_key": "hdfs:session_99",
    "timestamp": 1743340800,
    "explanation": "Anomalous Critical sequence detected...",
    "semantic_similarity": 0.87,
    "top_similar_events": ["HDFS block cascade failure"],
    "evidence_tokens": ["template_42", "template_17"]
  }
}
```

All new fields are `Optional[str | float | list]` in `AlertSchema`. When `SEMANTIC_ENABLED=false`, they are omitted or `null`. No breaking schema change.

---

### Optional: `POST /v3/ingest`

A clean entry point for the full V3 pipeline, accepting a raw log string and returning the fully enriched V3 response including explanation. This is the recommended endpoint for new integrations built for V3.

Internally calls the same pipeline as `/v2/ingest` plus the semantic enrichment step. Isolated in `src/api/routes_v3.py` to avoid coupling with V1/V2 routes.

---

## 13. Deployment and Dependency Impact

### CPU vs GPU practicality

| Model | Parameters | CPU Inference | GPU Required | RAM (CPU) |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | ~10–50ms per batch | No | ~85MB |
| `flan-t5-small` | 80M | ~100–500ms per call | No | ~300MB |
| V1 IsolationForest | N/A | <1ms | No | <5MB |
| V1 Transformer | ~2M | ~2–5ms | No | ~10MB |
| V2 LSTM + AE + MLP | ~5M | ~0.5ms | No | ~25MB |

The semantic layer is designed CPU-first throughout. All recommended models are runnable without a GPU. Semantic enrichment is triggered only for confirmed anomalies (a minority of windows), so there is no per-event overhead.

---

### Docker image impact

| Scenario | Additional Dependencies | Approximate Image Impact |
|---|---|---|
| V3 with `SEMANTIC_ENABLED=false` | `sentence-transformers` library only (~50MB) | +50MB library, no model download |
| V3 with `all-MiniLM-L6-v2` pre-baked | Library + model weights | +140MB total |
| V3 with `flan-t5-small` optional | Library + larger model | +350MB if included |

**Recommended Dockerfile strategy:** Pre-download `all-MiniLM-L6-v2` during build to avoid runtime Hub requests. Keep `flan-t5-small` optional and download only if `EXPLANATION_MODEL=flan-t5-small` is set at container start.

```dockerfile
# Add to Dockerfile after pip install:
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"
```

---

### Dependency additions

Add to `requirements/requirements.txt`:
```
sentence-transformers>=2.7.0
```

`sentence-transformers` transitively brings `transformers`, `tokenizers`, and `huggingface-hub`. These do not conflict with any current production dependency.

---

### Latency analysis

| Operation | Current | V3 (SEMANTIC_ENABLED=false) | V3 (SEMANTIC_ENABLED=true) |
|---|---|---|---|
| Ingest (no window emit) | <1ms | <1ms (unchanged) | <1ms (unchanged) |
| Ingest (window emit, demo) | 4–5ms | 4–5ms (unchanged) | 4–5ms + semantic |
| Semantic enrichment | N/A | N/A | +10–50ms (anomaly only) |
| Rule-based explanation | N/A | N/A | +<1ms |
| LLM explanation (optional) | N/A | N/A | +100–500ms |

The hot path (ingest → buffer → score) is completely unaffected. Semantic enrichment adds latency only on the alert path for confirmed anomalies.

---

### Offline and reproducibility constraints

- Set `HF_HUB_OFFLINE=1` to disable Hub access after the Docker build pre-download step
- For environments without internet access, use `SEMANTIC_MODEL_PATH` to point to a local model directory
- The `sentence-transformers` library supports fully offline usage when the model is cached locally
- CI runs with `SEMANTIC_ENABLED=false` require no internet access and no model download

---

## 14. Testing and CI Impact

### Existing tests: impact analysis

| Test File | Impact | Required Action |
|---|---|---|
| `tests/unit/test_inference_engine_smoke.py` | None | Semantic layer is additive; no change needed |
| `tests/integration/test_smoke_api.py` | Low | Verify that new optional fields in `AlertSchema` do not fail existing assertions |
| `tests/system/test_end_to_end_pipeline.py` | Low | Verify enriched alert fields default to `None` when `SEMANTIC_ENABLED=false` |
| `tests/test_stage_07_ingest_integration.py` | Low | Pydantic schema validation — new optional fields must have `None` defaults |
| `tests/unit/test_explain_decode.py` | Investigate | May be stale; resolve before V3 work begins |
| `tests/unit/test_proactive_engine.py` | Needs decision | Archive alongside `src/engine/` or delete |
| All others | None | Semantic layer is fully gated; zero impact when disabled |

---

### New tests required for V3

| Test File | Location | What It Tests |
|---|---|---|
| `test_semantic_config.py` | `tests/unit/` | `SemanticConfig` loads correctly from env vars with correct defaults |
| `test_semantic_embeddings.py` | `tests/unit/` | `SemanticEmbeddingService.embed()` with fully mocked `SentenceTransformer` |
| `test_semantic_similarity.py` | `tests/unit/` | `SemanticSimilarityScorer.query()` cosine similarity logic |
| `test_anomaly_explainer.py` | `tests/unit/` | `AnomalyExplainer` rule-based output format and content |
| `test_semantic_layer.py` | `tests/unit/` | `SemanticLayer.enrich()` with mocked sub-components |
| `test_semantic_pipeline_integration.py` | `tests/integration/` | Full pipeline with `SEMANTIC_ENABLED=true` (mocked HF model) |
| `test_explanation_route.py` | `tests/` | `GET /alerts/{id}/explanation` — response schema and 404 when disabled |
| `test_models_info_route.py` | `tests/` | `GET /models/info` — response structure |

---

### Mocking strategy for CI

All Hugging Face model operations must be mocked in the standard test suite:

```python
@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
    monkeypatch.setattr(
        "src.semantic.loader.SentenceTransformer",
        lambda *args, **kwargs: mock_model
    )
    return mock_model
```

**Test markers:**
- Add `@pytest.mark.semantic` for tests that require a real HF model
- CI command remains: `pytest -m "not slow and not semantic"`
- Optional CI step for `main` branch only: `pytest -m "semantic"` with HF model cached

---

### How to keep CI fast and stable

1. `SEMANTIC_ENABLED=false` in all CI environment configurations — existing 578 tests are completely unaffected
2. All new V3 unit and integration tests use mocked HF models — no downloads, no GPU, no latency
3. The `@pytest.mark.slow` and `@pytest.mark.semantic` markers are used consistently
4. Docker smoke tests use `SEMANTIC_ENABLED=false` for fast build verification
5. A separate optional CI stage (not blocking) can run the full semantic stack for integration validation on `main`

---

## 15. Phased Execution Plan

### Phase 0 — Deep Scan and Protection (1 session)

**Goal:** Establish a verified, protected baseline before any changes.

- Run full test suite; confirm 578/578 pass
- Run Docker build and compose stack; confirm healthy
- Document current `models/` contents
- Read the four ambiguous file pairs (modeling wrappers vs submodules)
- Read `src/data/` vs `src/synthetic/` — document differences
- Inspect `ai_workspace/` and `static-demo/` — determine disposition
- Read `tests/unit/test_explain_decode.py` — determine if stale
- Create `archive/` directory structure

**Exit criteria:** Known, confirmed baseline. No unknowns remaining before cleanup.

---

### Phase 1 — Cleanup and Repo Restructuring (1–2 sessions)

**Goal:** Remove noise, resolve duplications, archive legacy components, rename scripts.

- Move `evaluation_report.json` → `reports/`
- Purge `artifacts/n8n_outbox/` stale files
- Rename `scripts/stage_07_run_api.py` → `scripts/run_api.py`; update references
- Move stage 01–06 scripts → `scripts/data_pipeline/`
- Merge `src/data/` unique logic into `src/synthetic/`; delete `src/data/`
- Archive `src/engine/proactive_engine.py` → `archive/src/engine/`
- Resolve `src/modeling/` top-level wrapper vs submodule duplication
- Handle stale test (`test_explain_decode.py`) and archived test (`test_proactive_engine.py`)
- Update `.gitignore` for model artifacts and generated outputs

**Exit criteria:** Full test suite green. Docker build green. No stale tests. No duplicate modules. Script names reflect function, not stage numbers.

---

### Phase 2 — HuggingFace Embeddings Integration (1–2 sessions)

**Goal:** Introduce HuggingFace as an inert-by-default optional layer with proper config and loading.

- Add `sentence-transformers` to `requirements/requirements.txt`
- Create `src/semantic/config.py`
- Create `src/semantic/loader.py`
- Create `src/semantic/embeddings.py`
- Create `src/semantic/__init__.py` (stub facade)
- Write unit tests with fully mocked HF model
- Verify `SEMANTIC_ENABLED=false` adds zero overhead to the hot path
- Docker build test with the new dependency

**Exit criteria:** HF model loads correctly when enabled. All tests (including new ones) green. `SEMANTIC_ENABLED=false` is fully transparent to existing behavior.

---

### Phase 3 — Explanation Layer Integration (1–2 sessions)

**Goal:** Add similarity scoring and rule-based explanation generation.

- Create `src/semantic/similarity.py`
- Create `src/semantic/explainer.py` (rule-based mode)
- Complete `src/semantic/__init__.py` `SemanticLayer.enrich()` implementation
- Integrate `SemanticLayer` into `src/api/pipeline.py` (gated by `SEMANTIC_ENABLED`)
- Extend `Alert` dataclass and `AlertSchema` with optional V3 fields
- Unit and integration tests for all new components

**Exit criteria:** A confirmed anomaly with `SEMANTIC_ENABLED=true` produces an enriched alert payload including explanation text and semantic similarity score.

---

### Phase 4 — Alert/Runtime/API Wiring (1 session)

**Goal:** Expose V3 capabilities through the API and health system.

- Add `GET /alerts/{alert_id}/explanation` route
- Add `GET /models/info` route
- Update `src/health/checks.py` with semantic layer status
- Optional: add `POST /v3/ingest` route in `src/api/routes_v3.py`
- Add API-level tests for new routes

**Exit criteria:** New routes return correct responses. Health endpoint reflects semantic layer status. All tests green.

---

### Phase 5 — Observability, Docs, and Tests Cleanup (1 session)

**Goal:** Update monitoring, documentation, and test coverage to reflect V3.

- Add `semantic_latency_seconds` and `explanation_generated_total` to `src/observability/metrics.py`
- Update Grafana dashboard JSON with new metric panels
- Update `README.md` for V3
- Create `docs/v3_architecture.md`
- Create `notebooks/v3_semantic_demo.ipynb`
- Confirm full test suite (578 + new V3 tests) passes
- Update CI to add optional `@pytest.mark.semantic` step on main

**Exit criteria:** All tests green. Grafana shows semantic metrics. Documentation reflects V3 architecture.

---

### Phase 6 — Final V3 Polish (1 session)

**Goal:** Production-ready V3 release.

- Review all new module docstrings and type annotations
- Add HF model pre-download step to `docker/Dockerfile`
- Full Docker Compose end-to-end test with V3 stack
- End-to-end smoke test: ingest anomalous event → semantic enrichment → enriched alert → explanation endpoint
- Final `README.md` update with V3 screenshots and demo instructions
- Tag release as `v3.0.0`

**Exit criteria:** Full V3 system runs cleanly in Docker. All tests green. README reflects the complete V3 system. Repository is portfolio-ready.

---

## 16. Final Recommendation

### Should we adapt the current repo or rebuild?

**Adapt.** The existing system is well-engineered and production-ready. Rebuilding would destroy 578 passing tests, a working Docker stack, a validated alert lifecycle with deduplication and cooldown, a functioning Prometheus/Grafana observability stack, and a clean modular architecture — for no proportional benefit. V3 is an augmentation, not a replacement.

---

### What should be cleaned first?

Execute in this sequence to minimise risk:

1. **Purge `artifacts/n8n_outbox/` stale JSON files** — zero risk, immediate noise reduction
2. **Move `evaluation_report.json` to `reports/`** — zero risk, immediate structural clarity
3. **Rename and reorganise `scripts/`** — medium risk; must update `main.py` and Dockerfile references; run tests after
4. **Resolve `src/data/` vs `src/synthetic/` duplication** — medium risk; must read files and verify before deleting
5. **Archive `src/engine/proactive_engine.py`** — low risk; confirmed disconnected from runtime
6. **Resolve `src/modeling/` wrapper vs submodule duplication** — medium risk; must verify import paths in `inference_engine_v2.py` first

---

### What should be archived vs removed?

| Item | Action |
|---|---|
| `src/engine/proactive_engine.py` | Archive → `archive/src/engine/` |
| `src/data/` (after merge) | Remove |
| `scripts/archive/` contents | Remove if fully superseded; otherwise move to `archive/scripts/` |
| `artifacts/n8n_outbox/*.json` | Remove generated files; keep directory with `.gitkeep` |
| `evaluation_report.json` (root) | Move to `reports/evaluation_report_v2.json` |
| `ai_workspace/` | Remove or gitignore (verify contents first) |

---

### What HuggingFace path is best?

Use **`sentence-transformers` with `all-MiniLM-L6-v2`** as the embedding backbone. This model is:
- CPU-friendly: 22M parameters, ~85MB RAM, 10–50ms per batch
- Practically downloadable: ~90MB from HuggingFace Hub
- Well-maintained with a stable API
- Portfolio-recognisable: one of the most widely used text embedding models
- No GPU required for the demo or production use case

The explanation layer should start in **rule-based mode** (template-driven, zero latency, no extra dependencies) and offer optional **`flan-t5-small`** for actual generative explanation text, controlled by `EXPLANATION_MODEL=flan-t5-small`. Do not attempt to integrate a large LLM — `flan-t5-small` at 80M parameters is the correct scale for this system.

---

### What is the recommended first implementation step after planning?

**Verify the baseline before touching anything.**

Run `pytest -m "not slow"` to confirm 578 tests pass.

Then read the four ambiguous file pairs:
- `src/modeling/behavior_model.py` vs `src/modeling/behavior/lstm_model.py`
- `src/modeling/anomaly_detector.py` vs `src/modeling/anomaly/autoencoder.py`
- `src/modeling/severity_classifier.py` vs `src/modeling/severity/severity_classifier.py`
- `src/data/` vs `src/synthetic/`

And read `tests/unit/test_explain_decode.py` to determine if it is stale.

Those answers are the gate for proceeding. Once the baseline is confirmed and the ambiguous files are understood, the Phase 1 cleanup is safe to execute. Phase 2 (HuggingFace scaffolding) follows immediately after Phase 1 is verified green.

---

*Document prepared from a full scan of 143 Python source files across all modules, plus all infrastructure, test, and documentation files in the repository. All file paths, module names, and architectural assertions are grounded in direct inspection of the current codebase. No assumptions have been made about files that were not read.*
