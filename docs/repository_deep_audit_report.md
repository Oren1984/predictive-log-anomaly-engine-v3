# Predictive Log Anomaly Engine V3 — Repository Deep Audit Report

## 1. Executive Summary

This repository is a production-oriented FastAPI log anomaly platform with three active runtime surfaces:
- V1 path: tokenized event ingestion and anomaly scoring (baseline / transformer / ensemble) via `src/runtime/inference_engine.py`.
- V2 path: raw-log ingestion with Word2Vec + LSTM + Autoencoder + severity pipeline via `src/runtime/pipeline_v2.py` and `src/runtime/inference_engine_v2.py`.
- V3 semantic overlay: optional semantic explanation/similarity enrichment (disabled by default) via `src/semantic/*` and `src/api/routes_v3.py`.

What appears to be the real active system:
- API runtime and wiring in `src/api/*`.
- Runtime/modeling logic in `src/runtime/*`, `src/modeling/*`, `src/alerts/*`, `src/observability/*`, `src/security/*`.
- Container/ops path in `docker/*`, `prometheus/*`, `grafana/*`, `.github/workflows/ci.yml`.
- Test safety net in `tests/*`.

What appears old / duplicated / legacy / archive-like:
- Explicit legacy tree under `archive/*` (detached from active imports).
- Large generated artifacts (`artifacts/n8n_outbox/*`, `reports/*.csv|*.json|*.jsonl`).
- Stage-era naming residue and stale references in docs/scripts comments.
- Duplicate domain model surfaces (`src/data/*` vs `src/data_layer/*`).

General cleanup opportunities:
- Keep runtime code intact; prioritize docs/reference cleanup and archive hygiene.
- Resolve stale script/test references around Stage 05 calibration/demo files.
- Formalize generated-vs-source boundaries (reports/outbox/intermediate data).

Audit confidence: high for runtime/deployment paths; medium for optional demo/notebook value; medium-low where external manual workflows are implied.

---

## 2. Repository Top-Level Map

| Name | Type | Purpose | Status | Recommendation |
|---|---|---|---|---|
| `.github/` | Folder | CI workflow (`ci.yml`) for tests/security/docker smoke | Active | Keep |
| `.claude/` | Folder | Local assistant/tooling config | Optional | Keep but document as local-only |
| `.git/` | Folder | VCS metadata | Active | Keep |
| `.pytest_cache/` | Folder | Local pytest cache | Generated | Delete candidate (local) |
| `archive/` | Folder | Legacy scripts/engine/tests intentionally sidelined | Archive Candidate | Archive (already archived) |
| `artifacts/` | Folder | Runtime artifacts: thresholds, vocab/templates, outbox | Active + Generated | Keep, split generated policy in docs |
| `data/` | Folder | Raw/intermediate/processed/synth datasets | Active + Generated | Keep but document lifecycle |
| `demo/` | Folder | Standalone visual demo scripts | Optional | Keep but document non-runtime role |
| `docker/` | Folder | Container definitions and compose orchestration | Active | Keep |
| `docs/` | Folder | Architecture/process reports and prompts | Likely Active (mixed) | Keep but prune stale/internal docs |
| `examples/` | Folder | n8n integration examples | Optional | Keep but update stale script paths |
| `grafana/` | Folder | Provisioning + active dashboards + archived exports | Active + Archive Candidate | Keep, move old JSON exports under clearer archive label |
| `hf_cache/` | Folder | Hugging Face model cache mount target | Generated/Optional | Keep (runtime cache target) |
| `models/` | Folder | Trained model artifacts used by runtime | Active | Keep |
| `notebooks/` | Folder | Demo/explainer notebooks (CPU/GPU/V3 semantic) | Optional | Keep but label notebook intent/version clearly |
| `prometheus/` | Folder | Scrape config + alert rules | Active | Keep |
| `reports/` | Folder | Generated evaluation/calibration outputs | Generated | Keep but treat as generated artifacts |
| `requirements/` | Folder | Runtime and dev dependencies | Active | Keep |
| `scripts/` | Folder | Entrypoints, evaluation, data pipeline scripts | Active (mixed) | Keep, fix stale stage wrappers/paths |
| `src/` | Folder | Main application source code | Active | Keep |
| `static-demo/` | Folder | Static showcase website (non-runtime) | Optional / Likely Legacy companion | Keep or archive after product decision |
| `templates/` | Folder | Runtime-served UI HTML template | Active | Keep |
| `tests/` | Folder | Unit/integration/system tests | Active | Keep |
| `training/` | Folder | Model training scripts for V2 stack | Active | Keep |
| `main.py` | File | Thin root entrypoint delegating to `scripts/run_api.py` | Active | Keep |
| `pyproject.toml` | File | Pytest config/markers | Active | Keep |
| `README.md` | File | Primary operator/developer docs | Active (partially stale refs) | Keep but update stale links/paths |
| `.env.example` | File | Environment variable reference | Active | Keep |
| `.gitignore` | File | Ignore policy | Active | Keep |
| `.dockerignore` | File | Docker build context exclusions | Active (contains stale comments) | Keep but clean comments/typos |

---

## 3. Deep Folder-by-Folder Analysis

### 3.1 `src/`

Purpose:
- Primary runtime and API implementation.

Main contents and usage:
- `src/api/`: app factory, routes (`/ingest`, `/v2/*`, `/v3/*`), schemas, settings, UI router.
- `src/runtime/`: v1/v2 inference engines and sequence buffering.
- `src/modeling/`: baseline/transformer/v2 modeling implementations.
- `src/alerts/`: alert policy/dedup + n8n dispatch client.
- `src/observability/`: Prometheus middleware/registry.
- `src/security/`: API key middleware.
- `src/health/`: health composition.
- `src/semantic/`: V3 optional semantic layer.
- `src/parsing/`, `src/sequencing/`, `src/preprocessing/`, `src/dataset/`, `src/data_layer/`: data pipeline/runtime support.
- `src/synthetic/`: synthetic event generators actively used by tests and `scripts/demo_run.py`.
- `src/data/`: secondary `LogEvent` dataclass package; only test-level usage observed.

Evidence of usage:
- `main.py` -> `scripts/run_api.py` -> `src.api.app:create_app` path.
- CI smoke tests hit `/health`, `/metrics`, `/ingest`, `/alerts`, `/query`.
- Tests import broadly from `src/*`.

Status:
- Active overall.
- `src/data/` is likely duplicate/legacy-adjacent compared with `src/data_layer/`.

Risk if removed:
- High (core runtime).

Recommendation:
- Keep all core modules.
- Manual review for `src/data/` consolidation with `src/data_layer/`.

Confidence:
- High.

### 3.2 `scripts/`

Purpose:
- Runtime startup, evaluation, demo runner, and offline data pipeline/training-support scripts.

Main contents:
- `run_api.py`: active API launcher used by `main.py`.
- `evaluate_v2.py`: comparative evaluator (v1 vs v2).
- `demo_run.py`: in-process mock demo of API flow.
- `data_pipeline/`: stage scripts for dataset preparation/modeling/demo alerts.
- `00_check_env.ps1`: environment check stub (currently only comments).
- `smoke_test.sh`: local docker smoke script.

Evidence of usage:
- `main.py` imports `scripts.run_api`.
- README references `scripts/evaluate_v2.py`.
- CI directly uses docker/HTTP smoke path, not `smoke_test.sh`.

Issues found:
- `scripts/data_pipeline/stage_05_run.py` imports `scripts.stage_05_runtime_demo` and `scripts.stage_05_runtime_benchmark`, but those files exist only under `archive/scripts/`.
- Multiple script headers still advertise old paths (`scripts/stage_*`) after move to `scripts/data_pipeline/`.

Status:
- Active (mixed quality).

Risk if removed:
- Medium-High (some scripts are core operational helpers).

Recommendation:
- Keep.
- Manual review/fix Stage 05 wrapper and stale path references.

Confidence:
- High.

### 3.3 `training/`

Purpose:
- V2 model artifact training pipeline.

Main contents:
- `train_embeddings.py`, `train_behavior_model.py`, `train_autoencoder.py`, `train_severity_model.py`.

Evidence:
- README explicitly instructs these commands.
- `src/runtime/pipeline_v2.py` error messages direct users to these training modules.

Status:
- Active.

Risk if removed:
- High for retraining/reproducibility; runtime can still run only if artifacts already exist.

Recommendation:
- Keep.

Confidence:
- High.

### 3.4 `tests/`

Purpose:
- Behavioral and regression validation for unit/integration/system levels.

Main contents:
- Unit tests for core models/runtime/semantic components.
- Integration tests for API endpoints.
- System tests for end-to-end and fallback behavior.

Evidence:
- CI runs `pytest -m "not slow"`.
- API and semantic routes have direct test coverage.

Issues found:
- `tests/unit/test_runtime_calibration.py` dynamically imports `scripts/stage_05_runtime_calibrate.py`, which is not in active `scripts/` (exists in `archive/scripts/`).
- This test is marked `slow`, so CI does not catch this path mismatch.

Status:
- Active with some slow-test drift.

Risk if removed:
- High for quality assurance.

Recommendation:
- Keep; fix stale slow-test import paths in future maintenance.

Confidence:
- High.

### 3.5 `docker/`, `prometheus/`, `grafana/`

Purpose:
- Deployment and observability stack.

Main contents:
- `docker/Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`.
- Prometheus scrape and alert rules.
- Grafana datasource/dashboard provisioning and active dashboards.

Evidence:
- CI builds Docker image and runs compose smoke tests.
- Compose mounts prometheus/grafana folders and exposes services.

Status:
- Active.
- `grafana/archive/dashboards/` appears historical export storage.

Risk if removed:
- High for deployment/monitoring visibility.

Recommendation:
- Keep core files.
- Keep archived dashboard exports but isolate as archive-only docs asset.

Confidence:
- High.

### 3.6 `models/`, `artifacts/`, `data/`, `reports/`, `hf_cache/`

Purpose:
- Runtime model artifacts and generated data/evaluation outputs.

Usage:
- `models/*` and `artifacts/{threshold*,vocab.json,templates.json}` are loaded by runtime engines.
- `data/intermediate/templates.csv` is required by v2 tokenizer at runtime.
- `reports/*` and `artifacts/n8n_outbox/*` are generated outputs.
- `hf_cache/` is a runtime cache mount for semantic model downloads.

Evidence:
- Explicit model paths hardcoded in `src/runtime/*` and training scripts.
- Compose mounts `models`, `artifacts`, `data/intermediate`, `hf_cache`.

Status:
- Mixed: active + generated.

Risk if removed:
- `models/`, key `artifacts/*.json`, and required `data/intermediate/templates.csv`: High.
- `artifacts/n8n_outbox/*`, many `reports/*`: Low-Medium (diagnostic loss).

Recommendation:
- Keep core artifacts.
- Mark generated outputs for periodic cleanup/rotation.

Confidence:
- High.

### 3.7 `docs/`

Purpose:
- Human-readable architecture/process/report documentation.

Main contents:
- Primary large planning docs.
- `docs/reports/*`: phase-by-phase implementation reports.
- `docs/prompts/*`: execution prompt artifacts.

Evidence:
- Not runtime-referenced, but important for project history and governance.

Issues:
- Multiple stale references to old script names/removed folders.
- README references `docs/V3_ARCHITECTURE.md`, but file not found.

Status:
- Likely active for project communication, but mixed with generated process artifacts.

Risk if removed:
- Low runtime risk, Medium project-knowledge risk.

Recommendation:
- Keep; prune/label generated phase reports and stale prompt artifacts.

Confidence:
- Medium-High.

### 3.8 `notebooks/`, `demo/`, `static-demo/`, `examples/`

Purpose:
- Demo/presentation/integration companion assets.

Usage evidence:
- Not required by runtime or CI.
- Referenced in README/docs.
- `static-demo` explicitly documents itself as presentation layer.

Status:
- Optional.
- `static-demo/` and demo scripts are useful but non-core.

Risk if removed:
- Low runtime risk, Medium portfolio/demo risk.

Recommendation:
- Keep but clearly mark as non-production and optional.
- Update stale path references in docs/examples.

Confidence:
- Medium.

### 3.9 `archive/`

Purpose:
- Explicit legacy storage: old scripts, legacy engine code, and legacy tests.

Evidence:
- No imports from active code paths into `archive/` observed.
- Contains stage-era scripts and `proactive_engine` legacy implementation.

Status:
- Archive Candidate (already archived).

Risk if removed:
- Low runtime risk; Medium historical-reference risk.

Recommendation:
- Keep archived (or externalize snapshot) until historical retention decision.

Confidence:
- High.

---

## 4. Important File Analysis

| Path | What it does | Required? | Duplicate/Outdated signal | If removed | Recommendation |
|---|---|---|---|---|---|
| `main.py` | Root app entrypoint to `scripts/run_api.py` | Yes | None | Breaks primary local startup pattern | Keep |
| `README.md` | Main docs + quickstart + V3 section | Yes | References missing `docs/V3_ARCHITECTURE.md` | Runtime unaffected; onboarding/documentation degraded | Keep but update |
| `pyproject.toml` | Pytest config/markers | Yes | None | Test discovery/markers may break | Keep |
| `.env.example` | Env var reference for runtime controls | Yes | Generally current | Operator confusion if removed | Keep |
| `.gitignore` | Generated/large file ignore policy | Yes | Good but project-history dependent | Risk of committing artifacts | Keep |
| `.dockerignore` | Build context pruning | Yes | Stale comment references missing `docker/.dockerignore`; malformed tail token | Larger/slower builds if removed; confusion if stale | Keep but clean |
| `.github/workflows/ci.yml` | CI gates: tests/security/docker smoke | Yes | None | Loss of automated quality checks | Keep |
| `docker/Dockerfile` | Runtime image build | Yes | None | Container build path fails | Keep |
| `docker/docker-compose.yml` | Dev/demo stack (api+prometheus+grafana) | Yes | None | Main stack startup breaks | Keep |
| `docker/docker-compose.prod.yml` | Production override | Likely Yes | None | Production hardening path lost | Keep |
| `src/api/app.py` | FastAPI app factory + lifespan setup | Yes | None | API runtime fails | Keep |
| `src/api/routes.py` | V1 ingest/alerts/health/metrics | Yes | None | Core API loss | Keep |
| `src/api/routes_v2.py` | V2 raw log endpoints | Optional-but-active | None | V2 API unavailable | Keep |
| `src/api/routes_v3.py` | V3 semantic endpoints | Optional-but-active | None | V3 API unavailable | Keep |
| `src/api/pipeline.py` | Core event processing and semantic enrichment hook | Yes | None | Ingest pipeline breaks | Keep |
| `src/runtime/inference_engine.py` | V1 runtime scoring | Yes | None | V1 runtime breaks | Keep |
| `src/runtime/pipeline_v2.py` | V2 pipeline orchestration | Likely Active | None | V2 route execution breaks | Keep |
| `scripts/run_api.py` | API launcher | Yes | None | `main.py` path breaks | Keep |
| `scripts/evaluate_v2.py` | Comparative evaluator | Optional | Output path/docs drift vs report file names | No runtime break; evaluation workflow loss | Keep but document expected outputs |
| `scripts/data_pipeline/stage_05_run.py` | Stage wrapper for runtime demo+benchmark | Unclear / Likely Legacy in current state | Imports modules that only exist in `archive/scripts/` | Script currently fails if used | Manual review (fix or archive) |
| `tests/unit/test_runtime_calibration.py` | Slow calibration validation | Optional CI path | Imports non-existent active script path (`scripts/stage_05_runtime_calibrate.py`) | No fast-CI impact; slow suite drift | Keep but fix reference or archive test |
| `src/README.md` | Source tree documentation | Optional | Mentions non-existent `core` module; stale map | Runtime unaffected; contributor confusion | Keep but update |
| `data/README.md` | Data folder documentation | Optional | References old script path (`scripts/10_download_data.py`) | Runtime unaffected; data process confusion | Keep but update |
| `examples/n8n/n8n_flow_stub.md` | Integration walkthrough | Optional | Uses old script path (`scripts/stage_06_demo_alerts.py`) | Runtime unaffected | Keep but update |
| `reports/evaluation_report_v2.json` | Historical generated evaluation output | Optional/Generated | Contains paths to a different repo location (`predictive-log-anomaly-engine-v2`) | Runtime unaffected | Keep as generated evidence or archive |
| `templates/index.html` | Runtime dashboard UI page served by API | Yes | None | `/` UI breaks | Keep |
| `static-demo/index.html` | Static presentation site | Optional | Non-runtime companion | Runtime unaffected | Keep or archive by product decision |

---

## 5. Active System Flow

### 5.1 Main runtime path (actual current system)

1. Entry:
- `main.py` delegates to `scripts/run_api.py`.

2. Server boot:
- `scripts/run_api.py` launches `uvicorn` with app factory `src.api.app:create_app`.

3. App factory and startup:
- `src/api/app.py` creates `Pipeline`, middleware, route registration.
- Lifespan loads base pipeline models; conditionally loads v2 engine when `MODEL_MODE` contains `v2`.

4. Backend flow:
- `/ingest` -> `src/api/routes.py` -> `Pipeline.process_event()` -> `InferenceEngine.ingest()` -> `AlertManager.emit()` -> optional n8n outbox/webhook.
- `/v2/ingest` -> `src/api/routes_v2.py` -> `InferenceEngineV2.process_log()`.
- `/v3/ingest` and explanation/info routes via `src/api/routes_v3.py`; semantic enrichment gated by `SEMANTIC_ENABLED`.

5. Frontend flow:
- `GET /` served by `src/api/ui.py` from `templates/index.html`.
- `POST /query` uses static in-process RAG stub in `src/api/ui.py`.

6. Notebook/demo flow:
- Notebooks under `notebooks/` and scripts under `demo/` are companion demos, not runtime dependencies.

7. Docker flow:
- `docker/docker-compose.yml` starts API + Prometheus + Grafana and mounts required model/artifact/data/cache paths.
- CI validates this path with health/metrics/ingest smoke tests.

8. Config flow:
- Environment-driven `Settings` in `src/api/settings.py`.
- `.env.example` documents controls.

9. Observability flow:
- `src/observability/metrics.py` exposes Prometheus metrics at `/metrics`.
- `prometheus/prometheus.yml` scrapes API.
- `grafana/provisioning/*` auto-loads dashboards.

10. Test flow:
- CI runs fast suite (`pytest -m "not slow"`) and docker smoke.
- Slow tests include additional artifact-heavy checks.

Conclusion:
- The truly active V3-capable system is `main.py` + `scripts/run_api.py` + `src/*` + docker/prometheus/grafana + tests.

---

## 6. Legacy / Duplicate / Archive Candidates

| Path | Why it looks legacy/duplicate | Evidence | Recommendation | Risk if removed |
|---|---|---|---|---|
| `archive/` | Explicit legacy holding area | No active imports from code runtime/tests fast path | Keep archived (or external snapshot) | Low |
| `archive/src/engine/proactive_engine.py` | Marked legacy in file header; not wired to app | Active app uses `src/runtime/*` and `src/api/pipeline.py` | Archive/keep for historical reference | Low |
| `archive/scripts/stage_05_runtime_*.py` | Stage-era scripts detached from current active scripts | Active wrapper references them indirectly but outside active path | Archive candidate; decide retention | Low-Medium |
| `scripts/data_pipeline/stage_05_run.py` | Broken/stale wrapper imports missing active modules | Imports `scripts.stage_05_runtime_demo` and `scripts.stage_05_runtime_benchmark` not present outside archive | Manual review: fix or archive | Medium |
| `tests/unit/test_runtime_calibration.py` | Slow test points to non-active script path | Imports `scripts/stage_05_runtime_calibrate.py` which exists only in archive | Manual review/fix | Medium |
| `src/data/` vs `src/data_layer/` | Duplicate domain modeling surfaces | `src/data/log_event.py` and `src/data_layer/models.py` both define `LogEvent`; little usage of `src/data` | Merge candidate after review | Medium |
| `grafana/archive/dashboards/*` | Timestamped exports alongside active dashboards | Active provisioning uses `grafana/dashboards/*` | Keep in archive subtree or externalize | Low |
| `docs/prompts/*` | Process execution artifacts, not product docs | Not runtime-referenced | Keep but consider archival grouping | Low |
| `docs/reports/PHASE_*` | Implementation process logs, not operator docs | Large volume, historical process state | Keep but tag as historical | Low |
| `static-demo/` | Presentation site not tied to runtime | Self-described companion site only | Keep or archive by product decision | Low |

---

## 7. Safe Cleanup Candidates

These items appear safe to archive/remove after manual verification (no code changes performed in this audit):

| Path | Reason | Why likely safe | What could break | Confidence |
|---|---|---|---|---|
| `artifacts/n8n_outbox/*.json` | Generated dry-run payload accumulation | Runtime only needs directory, not historical payload history | Loss of historical demo evidence | High |
| `.pytest_cache/` | Test cache only | Regenerated automatically | Nothing functional | High |
| `__pycache__/` folders | Python bytecode cache | Regenerated automatically | Nothing functional | High |
| `reports/*.csv`, `reports/*.json`, `reports/*.jsonl` | Generated outputs | Not imported by runtime | Loss of historical evaluation artifacts | High |
| `grafana/archive/dashboards/*.json` | Historical exported dashboards | Not provisioning-targeted | Loss of historical dashboard snapshots | Medium-High |
| `docs/prompts/*` | Internal phase execution prompts | Not runtime or CI dependencies | Loss of process traceability | Medium |

Note: `static-demo/` is likely safe from runtime perspective but may be valuable for portfolio/demo; classify as manual decision, not immediate safe delete.

---

## 8. High-Risk Files — Do Not Remove

Must keep (core runtime/deployment integrity):
- `main.py`
- `scripts/run_api.py`
- `src/api/app.py`
- `src/api/pipeline.py`
- `src/api/routes.py`
- `src/runtime/inference_engine.py`
- `src/runtime/pipeline_v2.py`
- `src/alerts/*`
- `src/observability/metrics.py`
- `src/security/auth.py`
- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `prometheus/prometheus.yml`
- `prometheus/alerts.yml`
- `grafana/provisioning/*`
- `.github/workflows/ci.yml`
- `requirements/requirements.txt`
- `requirements/requirements-dev.txt`
- `models/*` required artifacts and core `artifacts/{threshold.json,threshold_transformer.json,vocab.json,templates.json}`
- `data/intermediate/templates.csv` (required by v2 runtime tokenizer)
- `templates/index.html` and `src/api/ui.py` (if runtime UI is required)

---

## 9. Documentation Gaps

Key gaps found:
- Missing file reference: README mentions `docs/V3_ARCHITECTURE.md`, but file not found.
- `src/README.md` module map includes `core` module that is not present.
- `data/README.md` references old script path (`scripts/10_download_data.py`) not present.
- `examples/n8n/n8n_flow_stub.md` references old script path (`scripts/stage_06_demo_alerts.py`), now under `scripts/data_pipeline/`.
- Many stage scripts still contain legacy header usage examples with pre-move paths.
- Slow-test calibration docs/tests reference non-active script location.
- `.dockerignore` comment references non-existent `docker/.dockerignore` and contains malformed trailing ignore token.

Recommendation:
- Run a docs-only consistency pass (no code refactor) to align paths, current entrypoints, and artifact expectations.

---

## 10. Final Cleanup Proposal

### Phase 1: Safe documentation updates only
- Update README and folder READMEs with current script paths and existing docs links.
- Mark optional/demo/historical areas explicitly (`demo/`, `static-demo/`, `docs/reports/`, `docs/prompts/`).
- Add generated-artifacts policy (what is source-of-truth vs disposable).

### Phase 2: Archive candidates
- Rotate or archive `artifacts/n8n_outbox/*.json` and generated `reports/*` evidence outputs.
- Keep only active Grafana dashboards in primary dashboard folder; move historical exports to clearer archive label.

### Phase 3: Manual review items
- Resolve `scripts/data_pipeline/stage_05_run.py` stale imports (fix or archive).
- Resolve `tests/unit/test_runtime_calibration.py` import target mismatch.
- Decide on `src/data/` vs `src/data_layer/` consolidation strategy.
- Decide long-term retention of `static-demo/`.

### Phase 4: Possible deletion after verification
- Delete disposable caches (`.pytest_cache`, `__pycache__`, regenerated local outputs).
- Optionally prune historical process prompt/report files if governance retention allows.

---

## 11. Final Classification Table

| Path | Purpose | Status | Risk if removed | Recommendation | Confidence |
|---|---|---|---|---|---|
| `src/` | Core application logic | Active | High | Keep | High |
| `scripts/run_api.py` | Runtime launcher | Active | High | Keep | High |
| `scripts/data_pipeline/` | Offline prep/eval/demo scripts | Likely Active (mixed) | Medium-High | Keep + fix stale Stage 05 wrapper | High |
| `training/` | Model training/retraining | Active | High (for reproducibility) | Keep | High |
| `tests/` | Quality and regression safety net | Active | High | Keep + fix slow-test drift | High |
| `docker/` | Container build/run definitions | Active | High | Keep | High |
| `prometheus/` | Metrics scrape/rules | Active | High (ops visibility) | Keep | High |
| `grafana/dashboards/` | Active dashboards | Active | Medium | Keep | High |
| `grafana/archive/` | Historical dashboard exports | Archive Candidate | Low | Archive/retain as historical | High |
| `archive/` | Legacy code and scripts | Archive Candidate | Low | Keep archived | High |
| `models/` | Runtime model artifacts | Active | High | Keep | High |
| `artifacts/*.json` (core thresholds/vocab/templates) | Runtime thresholds and mappings | Active | High | Keep | High |
| `artifacts/n8n_outbox/` | Generated alert payloads | Generated | Low | Rotate/archive periodically | High |
| `data/intermediate/templates.csv` | V2 tokenizer dependency | Active | High | Keep | High |
| `data/raw/`, `data/processed/`, `data/synth/` | Datasets and generated data | Active + Generated | Medium | Keep with lifecycle policy | High |
| `reports/` | Generated evaluations/calibration outputs | Generated | Low-Medium | Keep as artifacts or archive | High |
| `docs/` | Human docs/history | Likely Active (mixed) | Low runtime / Medium knowledge | Keep, prune stale refs | Medium-High |
| `docs/reports/PHASE_*` | Process history reports | Optional historical | Low | Keep or archive as historical | Medium |
| `docs/prompts/` | Internal phase prompts | Optional | Low | Archive candidate after process close | Medium |
| `notebooks/` | Demo and explanation notebooks | Optional | Low runtime / Medium showcase | Keep, label intent/version | Medium |
| `demo/` | Standalone visual demos | Optional | Low runtime | Keep optional | Medium |
| `static-demo/` | Static showcase website | Optional / Likely legacy companion | Low runtime | Keep or archive by product choice | Medium |
| `examples/n8n/` | Integration examples | Optional | Low | Keep but update script paths | High |
| `.github/workflows/ci.yml` | CI pipeline | Active | High | Keep | High |
| `.env.example` | Runtime config template | Active | Medium | Keep | High |
| `.dockerignore` | Build context exclusions | Active (stale bits) | Medium | Keep but clean comments/typos | High |
| `.pytest_cache/`, `__pycache__/` | Local caches | Generated | Low | Safe delete candidate | High |

---

## Audit Method and Evidence Notes

Method used in this audit:
- Import/reference tracing across `src`, `scripts`, `tests`.
- Entrypoint/deployment verification through `main.py`, compose files, Dockerfile, CI workflow.
- Observability flow verification via prometheus/grafana configs and API metrics routes.
- Documentation/reference consistency checks for path drift and missing files.
- Legacy comparison between active tree and `archive/`.

Uncertain items were marked accordingly where runtime linkage is not definitive (especially optional/demo/historical assets).
