# Notebooks V3 Audit Report

**Date:** 2026-03-30
**Repository:** `C:\Users\ORENS\predictive-log-anomaly-engine-v3`
**Auditor:** Claude Code (automated deep audit)
**Notebooks audited:**
- NB1: `notebooks/predictive_log_anomaly_engine_demo.ipynb` (21 cells, ~473 KB)
- NB2: `notebooks/predictive_log_anomaly_engine_gpu_demo.ipynb` (19 cells, ~332 KB)

---

## Executive Summary

Both notebooks were written to describe the **V2** system and have not been updated to reflect the **V3 semantic layer** that was integrated in Phases 6–8. Neither notebook is broken at execution time — neither imports project-internal modules, so no cells will crash. However, both notebooks contain significant **descriptive inaccuracies** and **omissions** that misrepresent the current system to any reader or reviewer.

The most critical gaps are:

1. **Three V3 API endpoints are entirely absent** from NB1 (the main demo): `POST /v3/ingest`, `GET /v3/alerts/{id}/explanation`, `GET /v3/models/info`.
2. **Three new Prometheus metrics are absent** from NB1's metrics table: `semantic_enrichments_total`, `semantic_enrichment_latency_seconds`, `semantic_model_ready`.
3. **Both notebooks label the project "v2"** in titles, footers, and architectural summaries; the repo is `predictive-log-anomaly-engine-v3` and the V3 semantic layer is fully integrated.
4. **The V3 post-alert enrichment step is invisible** in all pipeline flow diagrams across both notebooks.
5. **NB2 (GPU demo) does not mention `sentence-transformers`** as a third GPU-eligible model introduced in V3.
6. **Severity label vocabularies are conflated**: NB1 uses `Info / Warning / Critical` (V2 `SeverityClassifier` labels) interchangeably with V1 `AlertPolicy` labels (`critical / high / medium / low`) without distinguishing them.

No broken imports or wrong file paths were found in either notebook.

---

## Repository Ground Truth (Audit Baseline)

### Active API Endpoints

| Method | Path | File |
|--------|------|------|
| POST | `/ingest` | `src/api/routes.py` |
| GET | `/alerts` | `src/api/routes.py` |
| GET | `/health` | `src/api/routes.py` |
| GET | `/metrics` | `src/api/routes.py` |
| POST | `/v2/ingest` | `src/api/routes_v2.py` |
| GET | `/v2/alerts` | `src/api/routes_v2.py` |
| POST | `/v3/ingest` | `src/api/routes_v3.py` |
| GET | `/v3/alerts/{alert_id}/explanation` | `src/api/routes_v3.py` |
| GET | `/v3/models/info` | `src/api/routes_v3.py` |

### Prometheus Metrics (full list — `MetricsRegistry`)

| Metric | Type | Notes |
|--------|------|-------|
| `ingest_events_total` | Counter | |
| `ingest_windows_total` | Counter | |
| `alerts_total` | Counter | label: `severity` |
| `ingest_errors_total` | Counter | |
| `ingest_latency_seconds` | Histogram | |
| `scoring_latency_seconds` | Histogram | |
| `service_health` | Gauge | |
| `semantic_enrichments_total` | Counter | **V3 — new** |
| `semantic_enrichment_latency_seconds` | Histogram | **V3 — new** |
| `semantic_model_ready` | Gauge | **V3 — new** |

### Alert Severity Labels

| Source | Labels |
|--------|--------|
| V1 `AlertPolicy.classify_severity()` | `critical`, `high`, `medium`, `low` |
| V2 `SeverityClassifier` (ML model) | `info`, `warning`, `critical` |

These are **two separate systems**. NB1 conflates them.

### V3 Alert Enrichment Fields (added to `Alert.to_dict()` and `AlertSchema`)

`explanation` · `evidence_tokens` · `semantic_similarity` · `top_similar_events`
All `None` when `SEMANTIC_ENABLED=false` (default). Populated by `Pipeline._enrich_alert()`.

### Model Artifacts Confirmed Present

- `models/baseline.pkl`
- `models/transformer.pt`
- `models/embeddings/word2vec.model`
- `models/behavior/behavior_model.pt`
- `models/anomaly/anomaly_detector.pt`
- `models/severity/severity_classifier.pt`
- `artifacts/vocab.json`, `artifacts/templates.json`, `artifacts/threshold.json`
- `artifacts/threshold_transformer.json`, `artifacts/threshold_runtime.json`

---

## Notebook 1: `predictive_log_anomaly_engine_demo.ipynb`

### Cell-Level Issue Inventory

#### Cell 0 — Title / Metadata Table

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-01 | `OUTDATED_ARCH` | MUST_FIX | `Project: Predictive Log Anomaly Engine v2` | Should be `v3` — repo is `predictive-log-anomaly-engine-v3` with integrated V3 layer |
| N1-02 | `OUTDATED_ARCH` | MUST_FIX | `Architecture: V1 Ensemble + V2 LSTM Autoencoder Pipeline (parallel)` | Missing V3 semantic enrichment. Should note all three layers. |
| N1-03 | `MISSING_V3` | MUST_FIX | No row or note for `SEMANTIC_ENABLED` / V3 semantic layer in metadata | V3 semantic config (`SEMANTIC_ENABLED`, `SEMANTIC_MODEL`, cache size) should appear or be referenced |

**Valid content:** Runtime stack (FastAPI · PyTorch · Prometheus · Grafana), inference modes list, test count claim.

---

#### Cell 1 — Table of Contents

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N1-04 | `MISSING_V3` | NICE_TO_HAVE | No TOC entry for V3 semantic layer, V3 endpoints, or V3 metrics |

**Valid:** Overall TOC structure and section numbering.

---

#### Cell 2 — Executive Overview

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-05 | `OUTDATED_ARCH` | MUST_FIX | Describes "two parallel inference pipelines: V1 ensemble and V2 neural pipeline" | Three layers now: V1, V2, and V3 semantic enrichment on top of confirmed alerts |
| N1-06 | `MISSING_V3` | MUST_FIX | No mention of `explanation`, `evidence_tokens`, `semantic_similarity`, `top_similar_events` fields | These are now part of every fired alert when `SEMANTIC_ENABLED=true` |

---

#### Cell 3 — Problem Context

No issues. Conceptual, technology-agnostic. **VALID — do not change.**

---

#### Cell 4 — Why Log Anomaly Detection Matters

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-07 | `STALE_MODEL` | MUST_FIX | Severity categories listed as `info / warning / critical` | V1 alert path uses `critical / high / medium / low`. `info/warning/critical` are V2 ML classifier labels. Must disambiguate. |
| N1-08 | `MISSING_V3` | NICE_TO_HAVE | "Sub-10 ms end-to-end V2 inference latency" unqualified | When `SEMANTIC_ENABLED=true`, V3 enrichment adds additional latency (observable via `semantic_enrichment_latency_seconds`). Qualify as CPU-only, semantic-disabled path. |

---

#### Cell 5 — Imports and Setup

No issues. Imports only `numpy`, `matplotlib`, `datetime` — all standard. **VALID.**

---

#### Cell 6 — System Architecture (text description)

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-09 | `MISSING_V3` | MUST_FIX | Architecture table has two rows: V1 Ensemble and V2 Neural. No V3 row. | Add: `V3 Semantic Layer \| cross-cutting \| SBERT embeddings + rule-based explanation + cosine similarity` |
| N1-10 | `MISSING_V3` | MUST_FIX | Three V3 API endpoints not listed anywhere | `POST /v3/ingest`, `GET /v3/alerts/{id}/explanation`, `GET /v3/models/info` are all registered and functional |

---

#### Cell 7 — Architecture Diagram (matplotlib)

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N1-11 | `MISSING_V3` | NICE_TO_HAVE | Pipeline diagram ends at Alert Manager with no V3 enrichment stage |

Not a runtime blocker; visualization-only. **NICE_TO_HAVE** to add "Semantic Enrichment (V3)" box post-Alert Manager.

---

#### Cell 8 — V2 AI Pipeline Overview

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-12 | `STALE_MODEL` | MUST_FIX | Severity Classifier output labels: `info / warning / critical` | These are V2 ML classifier labels. V1 `AlertPolicy` labels are `critical / high / medium / low`. Both coexist — must label clearly which is which. |
| N1-13 | `MISSING_V3` | MUST_FIX | Pipeline ends at step 8: "AlertManager emits alert". No V3 step. | Step 9 is missing: "V3 Semantic Enrichment → `explanation` + `evidence_tokens` + `semantic_similarity` attached to alert dict (when `SEMANTIC_ENABLED=true`)" |

---

#### Cell 9 — V2 Pipeline Diagram (matplotlib)

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N1-14 | `MISSING_V3` | NICE_TO_HAVE | Diagram ends at Alert with no V3 stage |

**Valid:** Footnote "All four models trained on HDFS dataset — Artifacts in `models/` directory" — confirmed correct, all artifacts present.

---

#### Cell 10 — Runtime and API Flow *(most critical cell)*

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-15 | `STALE_ENDPOINT` | MUST_FIX | Endpoint table lists only V1 and V2 routes. Three V3 endpoints missing. | Add `POST /v3/ingest`, `GET /v3/alerts/{id}/explanation`, `GET /v3/models/info` with response schemas |
| N1-16 | `STALE_METRIC` | MUST_FIX | Prometheus metrics table has 6 entries — missing 4. | Add: `ingest_errors_total` (Counter), `semantic_enrichments_total` (Counter), `semantic_enrichment_latency_seconds` (Histogram), `semantic_model_ready` (Gauge) |

**Valid:** The 6 metrics already listed are all correct. V2 request/response schema fields. Public endpoint list (minor: `/query` omitted but not misleading).

---

#### Cell 11 — Observability Stack

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N1-17 | `MISSING_V3` | NICE_TO_HAVE | No mention of V3 semantic layer, `SEMANTIC_ENABLED` env var, or `hf_cache/` volume |

**Valid:** Port assignments (8000, 9090, 3000), `docker compose` command, Grafana dashboard reference.

---

#### Cell 12 — System Validation Snapshot

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N1-18 | `MISSING_V3` | NICE_TO_HAVE | No row for `Semantic Layer` in the component health table. `GET /health` now returns a `semantic` component. |

---

#### Cell 13 — Health Status Chart (matplotlib)

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N1-19 | `MISSING_V3` | NICE_TO_HAVE | `components` list in chart omits "Semantic Layer (V3)" bar |

---

#### Cells 14–17 — Walkthrough, Score Distribution, Throughput

No technical inaccuracies. Synthetic demo data. Thresholds are illustrative. **VALID — do not change.**

---

#### Cell 18 — Severity Donut + Pipeline Latency

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-20 | `STALE_MODEL` | MUST_FIX | Severity donut uses labels `Info`, `Warning`, `Critical` | If depicting V1 alert pipeline output, correct labels are `critical`, `high`, `medium`, `low`. If depicting V2 classifier output, `info`/`warning`/`critical` are correct. Must label clearly. |

---

#### Cell 19 — Key Outcomes

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-21 | `MISSING_V3` | MUST_FIX | "Dual pipeline architecture: V1 ensemble and V2 neural pipeline" | Should read "Triple-layer architecture: V1 ensemble, V2 neural pipeline, and V3 semantic enrichment layer" |

---

#### Cell 20 — Final Summary

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N1-22 | `OUTDATED_ARCH` | MUST_FIX | Footer: "Predictive Log Anomaly Engine **v2**" | Should be `v3` |
| N1-23 | `MISSING_V3` | MUST_FIX | Summary describes multi-stage inference, severity classification, alert management, Prometheus/Grafana — no mention of semantic enrichment | V3 semantic layer is a differentiating feature that should appear in the final summary |

---

### NB1 — Summary Counts

| Category | MUST_FIX | NICE_TO_HAVE |
|----------|----------|--------------|
| `OUTDATED_ARCH` | 5 | 0 |
| `MISSING_V3` | 6 | 7 |
| `STALE_ENDPOINT` | 1 | 0 |
| `STALE_METRIC` | 1 | 0 |
| `STALE_MODEL` | 2 | 1 |
| `WRONG_PATH` | 0 | 0 |
| `BROKEN_IMPORT` | 0 | 0 |
| **Total** | **15** | **8** |

---

## Notebook 2: `predictive_log_anomaly_engine_gpu_demo.ipynb`

### Cell-Level Issue Inventory

#### Cell 0 — Title / Metadata Table

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N2-01 | `OUTDATED_ARCH` | MUST_FIX | `GPU-Accelerated Stages: LSTM Behavior Model · Autoencoder Detector` | When `SEMANTIC_ENABLED=true`, the V3 `sentence-transformers` model (`all-MiniLM-L6-v2`) is also a PyTorch model and GPU-eligible. |
| N2-02 | `OUTDATED_ARCH` | NICE_TO_HAVE | `CPU-Only Stages: Tokenization · Word2Vec Lookup · Alert Manager` | `RuleBasedExplainer` (V3, always CPU) and `SemanticEmbedder` (V3, CPU by default, GPU-eligible when enabled) are both missing |

---

#### Cell 1 — Table of Contents

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N2-03 | `MISSING_V3` | NICE_TO_HAVE | No TOC entry for V3 semantic model GPU applicability |

---

#### Cell 2 — Overview Text

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N2-04 | `OUTDATED_ARCH` | MUST_FIX | "every model in the V2 pipeline is inherently GPU-compatible. The LSTM Behavior Model and the Autoencoder Detector can be targeted to a CUDA device with a single `.to(device)` call" | Accurate but incomplete. V3 `sentence-transformers` is also a PyTorch model and can be `.to('cuda')`. Should note this as a third GPU-eligible model. |

---

#### Cell 3 — Why GPU Matters (text)

No technical inaccuracies in the GPU fundamentals described. **VALID.**

---

#### Cell 4 — Relationship to Main System

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N2-05 | `OUTDATED_ARCH` | MUST_FIX | Code block: `Standard System (CPU) └─ All four V2 pipeline stages` | Should be "V2 pipeline stages + optional V3 semantic enrichment stage (when `SEMANTIC_ENABLED=true`)" |

---

#### Cell 5 — Imports and Setup

No issues. Imports only `numpy` and `matplotlib`. **VALID.**

---

#### Cell 6 — Execution Path Comparison (text)

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N2-06 | `MISSING_V3` | MUST_FIX | GPU stages: Tokenization (CPU), Word2Vec (CPU), LSTM (CUDA), Autoencoder (CUDA), Alert Manager (CPU). V3 semantic model absent. | Add row: `Semantic Embedder (V3)` — CPU in standard path, GPU-eligible when `SEMANTIC_ENABLED=true` |

---

#### Cell 7 — Execution Path Diagram (matplotlib)

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N2-07 | `MISSING_V3` | MUST_FIX | `cpu_stages` and `gpu_stages` both have exactly 5 entries; V3 semantic embedding absent | Add sixth entry: CPU path `('Semantic Embedder (V3)', cpu_color, False)` with annotation `SEMANTIC_ENABLED=true`; GPU path `('Semantic Embedder (V3, CUDA)', gpu_color, True)` |

---

#### Cell 8 — Runtime Considerations / GPU Code Example

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N2-08 | `MISSING_V3` | NICE_TO_HAVE | No mention of V3 `SemanticEmbedder` as a third GPU-eligible model. Pattern: `SentenceTransformer(...).to(device)` |

**Valid:** Class names `SystemBehaviorModel` and `AnomalyDetector` confirmed correct in `src/modeling/`. `.to(device)` pattern is accurate. Illustrative `...` placeholders are acceptable.

---

#### Cells 9–10 — Throughput Scaling (text + chart)

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N2-09 | `MISSING_V3` | NICE_TO_HAVE | Latency breakdown chart stages: LSTM Behavior, Word2Vec, AE Scoring, Severity Cls., Alert Mgr. No "Semantic Embedder" bar. |

**Valid:** Explicit disclaimer that all figures are conceptual/illustrative. **VALID — keep disclaimer as-is.**

---

#### Cells 11–12 — Tradeoffs Table

No technical inaccuracies. **VALID.**

---

#### Cell 13 — GPU Applicability Chart

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N2-10 | `MISSING_V3` | NICE_TO_HAVE | `stages` list: Alert Manager, Severity Classifier, Autoencoder Scoring, LSTM Behavior Model, Embedding Lookup, Tokenization — no Semantic Embedder (V3) row |

---

#### Cell 14 — When GPU Helps

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N2-11 | `MISSING_V3` | NICE_TO_HAVE | High-throughput semantic embedding (SBERT per alert when `SEMANTIC_ENABLED=true`) is a valid GPU use case not mentioned |

---

#### Cells 15–16 — When CPU Is Enough / Deployment Positioning

| ID | Category | Severity | Issue |
|----|----------|----------|-------|
| N2-12 | `MISSING_V3` | NICE_TO_HAVE | GPU override snippet in cell 16 has no `SEMANTIC_ENABLED` or `HF_HOME` env vars |

**Valid:** `docker/docker-compose.prod.yml` path reference — confirmed exists. `scripts/evaluate_v2.py --max-sessions` reference — confirmed exists.

---

#### Cell 17 — Portfolio Value Table

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N2-13 | `OUTDATED_ARCH` | MUST_FIX | Portfolio value table rows cover framework fluency, architecture awareness, engineering judgement, deployment awareness, MLOps breadth, honest evaluation — no V3 semantic dimension | Add row: `Semantic / NLP layer` → "sentence-transformers SBERT embeddings, cosine similarity ranking, rule-based explainability — demonstrates NLP engineering depth" |

---

#### Cell 18 — Final Summary / Footer

| ID | Category | Severity | Wrong | Correct |
|----|----------|----------|-------|---------|
| N2-14 | `OUTDATED_ARCH` | MUST_FIX | Footer: "Predictive Log Anomaly Engine **v2** · PyTorch (CUDA-compatible) · LSTM + Autoencoder" | Should be `v3`. Add "sentence-transformers" or "V3 Semantic Layer" to tags. |
| N2-15 | `MISSING_V3` | MUST_FIX | "The V2 pipeline is built on PyTorch — which means GPU support is structurally present throughout the codebase" | Accurate but ignores V3. `sentence-transformers` is also PyTorch-based and GPU-compatible. Mention V3 semantic model as third GPU-eligible stage. |

---

### NB2 — Summary Counts

| Category | MUST_FIX | NICE_TO_HAVE |
|----------|----------|--------------|
| `OUTDATED_ARCH` | 5 | 1 |
| `MISSING_V3` | 3 | 7 |
| `STALE_ENDPOINT` | 0 | 0 |
| `STALE_METRIC` | 0 | 0 |
| `STALE_MODEL` | 0 | 0 |
| `WRONG_PATH` | 0 | 0 |
| `BROKEN_IMPORT` | 0 | 0 |
| **Total** | **8** | **8** |

---

## Cross-Notebook Findings

### No Broken Imports in Either Notebook

Neither notebook imports any project-internal module (`src.*`). All imports are from `numpy`, `matplotlib`, and the Python standard library. **No runtime failures from imports.**

### No Wrong Paths in Either Notebook

- `models/` artifact paths described in NB1 cell 9 — all confirmed present.
- `scripts/evaluate_v2.py` in NB2 cell 16 — confirmed present.
- `docker/docker-compose.prod.yml` in NB2 cell 16 — confirmed present.
- Docker port assignments (8000/9090/3000) — confirmed correct.

### Severity Label Confusion (NB1 only)

This is the most technically subtle issue. The system has **two severity classification systems** in production:

| System | Implementation | Labels | Used Where |
|--------|---------------|--------|------------|
| V1 Alert Policy | `AlertPolicy.classify_severity()` in `src/alerts/models.py` | `critical`, `high`, `medium`, `low` | V1 and V3 ingest alert pipeline |
| V2 ML Classifier | `SeverityClassifier` in `src/modeling/severity_classifier.py` | `info`, `warning`, `critical` | V2 pipeline only |

NB1 cells 4, 8, and 18 use `Info / Warning / Critical` without specifying they apply to the V2 ML classifier — not to the V1/V3 alert pipeline output. A reviewer comparing the notebook's severity label to a live `/ingest` alert response (which returns `critical/high/medium/low`) will find a mismatch.

---

## What Is Still Valid — Do Not Change

| Notebook | Cells | Content |
|----------|-------|---------|
| NB1 | 3 | Problem context — fully agnostic, timeless |
| NB1 | 14–17 | Walkthrough, score distribution, demo throughput charts — explicitly synthetic |
| NB1 | 10 (partial) | The 6 Prometheus metrics already listed are all correct |
| NB1 | 10 (partial) | V1 and V2 endpoint descriptions |
| NB1 | 11 | Port assignments, docker compose command |
| NB2 | 3 | GPU fundamentals — technically accurate |
| NB2 | 5 | Imports — clean |
| NB2 | 8 (partial) | `SystemBehaviorModel` and `AnomalyDetector` class names — confirmed correct |
| NB2 | 9–12 | Performance curves disclaimer ("conceptual, not measured") — keep as-is |
| NB2 | 16 (partial) | `docker-compose.prod.yml` and `evaluate_v2.py` references — confirmed present |

---

## Must-Fix List (Priority Order)

| Priority | Notebook | Cell(s) | Issue ID(s) | Description |
|----------|----------|---------|-------------|-------------|
| 1 | NB1 | 10 | N1-15 | Add three V3 endpoints to endpoint table |
| 2 | NB1 | 10 | N1-16 | Add four missing Prometheus metrics to metrics table |
| 3 | NB1 | 0, 20 | N1-01, N1-22 | Update version label from v2 → v3 |
| 4 | NB1 | 2, 6, 19 | N1-05, N1-09, N1-21 | Update architecture descriptions to include V3 layer |
| 5 | NB1 | 8 | N1-12, N1-13 | Add V3 enrichment step to pipeline flow; fix severity label disambiguation |
| 6 | NB1 | 4, 18 | N1-07, N1-20 | Disambiguate V1 vs V2 severity label vocabularies |
| 7 | NB1 | 20 | N1-23 | Add V3 semantic layer to final summary |
| 8 | NB2 | 0 | N2-01 | Add V3 `sentence-transformers` as third GPU-eligible model |
| 9 | NB2 | 2, 4 | N2-04, N2-05 | Update overview and CPU path description to include V3 stage |
| 10 | NB2 | 6, 7 | N2-06, N2-07 | Add Semantic Embedder stage to execution path comparison |
| 11 | NB2 | 17, 18 | N2-13, N2-14, N2-15 | Update portfolio value table and footer to reflect v3 |

---

## Nice-to-Have Improvements

| Notebook | Cell(s) | Description |
|----------|---------|-------------|
| NB1 | 1 | Add V3 TOC entry |
| NB1 | 7, 9, 13 | Add V3 semantic stage to architecture/pipeline/health diagrams |
| NB1 | 11 | Add note about `SEMANTIC_ENABLED` and `hf_cache/` volume |
| NB1 | 12 | Add Semantic Layer row to health snapshot |
| NB2 | 1 | Add V3 GPU applicability to TOC |
| NB2 | 8 | Add `SemanticEmbedder.to(device)` note |
| NB2 | 10 | Add Semantic Embedder bar to latency breakdown chart |
| NB2 | 13 | Add Semantic Embedder row to GPU applicability chart |
| NB2 | 14 | Add V3 embedding at scale as a GPU use case |
| NB2 | 16 | Add `SEMANTIC_ENABLED` + `HF_HOME` to GPU compose snippet |

---

## Manual Verification Items

These items require human judgment before editing:

1. **Test count claim in NB1 cell 0**: "578 tests" — current count after Phases 7–8 is 557 (not slow). Verify whether the slow tests account for the difference or if this number needs updating.
2. **V2 severity classifier label vocabulary**: Confirm `info / warning / critical` are the actual output labels of `src/modeling/severity_classifier.py`. The audit infers this from the notebook text but did not read the model's label encoder.
3. **GPU SBERT performance claims in NB2**: If any GPU benchmarks are added for V3, they must be measured on actual hardware, not inferred.
4. **Existing cell outputs**: Both notebooks may have rendered outputs (images, JSON) stored in the `.ipynb` JSON. These outputs will become stale after cell edits and should be cleared (`Kernel → Restart & Clear Output`) before re-running.

---

## Risk Notes

| Risk | Impact | Mitigation |
|------|--------|------------|
| Editing large notebook JSON by hand | Cell index misalignment, broken metadata | Use `NotebookEdit` tool or targeted JSON patches |
| Clearing outputs before re-running | If models aren't loaded, code cells that call the live API will fail | Only run API-calling cells with a live instance |
| Severity label fix creates confusion | If both V1 and V2 labels appear side-by-side in the same cell, readers may assume they are the same | Use a clear table with a "Pipeline" column |
| V3 sections in NB1 describing endpoints | Readers may try to call `/v3/alerts/{id}/explanation` without having first fired an alert | Add a prerequisite note |

---

## Recommended Update Order

**Update NB1 first.** Reasons:

1. NB1 is the **main demo notebook** and the primary document that reviewers, collaborators, and portfolio viewers will encounter first.
2. NB1 has the **higher MUST_FIX count** (15 vs 8) and contains the most structurally important inaccuracies (missing endpoints, missing metrics).
3. NB1 covers the full runtime pipeline including the API surface — which is where the V3 changes are most visible.
4. NB2 is a **supplementary GPU analysis notebook** with a narrower audience; its corrections are mostly additive (adding a V3 stage to charts) and can follow NB1 without risk.

---

## Proposed Phased Update Plan

### Phase 0 — Baseline Protection
- Commit both notebooks in their current state to a baseline tag/branch before any edits.
- Clear all stored cell outputs from both notebooks (kernel restart + clear output).
- Verify both notebooks open without errors in Jupyter.

### Phase 1 — Metadata and Version Labels (NB1 + NB2)
- Update version labels: NB1 cell 0, NB1 cell 20, NB2 cell 0, NB2 cell 18 footer.
- Change "v2" → "v3" in all title metadata, header tables, and footer tags.
- No functional changes; purely cosmetic. **Low risk.**
- Issues addressed: N1-01, N1-22, N2-14.

### Phase 2 — Severity Label Disambiguation (NB1)
- Fix cells 4, 8, and 18 to clearly separate V1 alert policy labels (`critical/high/medium/low`) from V2 ML classifier labels (`info/warning/critical`).
- Add a small reference table or parenthetical note at first occurrence in each cell.
- **Medium risk** — touches three cells but changes are additive (no deletions).
- Issues addressed: N1-07, N1-12, N1-20.

### Phase 3 — Architecture Descriptions (NB1 + NB2)
- NB1 cells 2, 6, 8, 19: Add V3 semantic enrichment layer to all pipeline and architecture descriptions.
- NB2 cells 2, 4: Add V3 stage to overview and CPU path description.
- Prose changes only; no code cells. **Low risk.**
- Issues addressed: N1-05, N1-06, N1-09, N1-13, N1-21, N2-04, N2-05.

### ⏸ STOP — Manual Review Point 1
- Review all prose changes against current repo before proceeding.
- Confirm test count (578 vs 557) and update if needed.
- Confirm V2 severity classifier label vocabulary from `src/modeling/severity_classifier.py`.

### Phase 4 — Endpoint and Metrics Tables (NB1 cell 10)
- Add three V3 endpoints to the endpoint table with correct HTTP method, path, and response schema summary.
- Add four missing Prometheus metrics (`ingest_errors_total`, `semantic_enrichments_total`, `semantic_enrichment_latency_seconds`, `semantic_model_ready`) to the metrics table.
- **High value** — the most factually wrong part of the notebook.
- Issues addressed: N1-15, N1-16.

### Phase 5 — Execution Path Diagram (NB2 cells 6–7)
- Add "Semantic Embedder (V3)" as a sixth stage to both the text and the matplotlib bar chart.
- Add `SEMANTIC_ENABLED=true` annotation.
- **Medium risk** — touches a code cell (the chart). Test rendering.
- Issues addressed: N2-06, N2-07.

### ⏸ STOP — Manual Review Point 2
- Run both notebooks in a Jupyter environment (with API running locally if needed for NB1 API cells).
- Verify that all chart cells render correctly after edits.
- Review V3 endpoint descriptions for accuracy against the live API.

### Phase 6 — Final Summaries and Key Outcomes (NB1 cells 19–20, NB2 cells 17–18)
- NB1 cell 19: "Dual pipeline" → "Triple-layer architecture".
- NB1 cell 20 final summary: Add V3 semantic layer sentence.
- NB2 cell 17: Add portfolio value row for semantic/NLP layer.
- NB2 cell 18: Update footer and final summary to acknowledge V3.
- Issues addressed: N1-21, N1-23, N2-13, N2-15.

### Phase 7 — GPU-Specific V3 Additions (NB2)
- NB2 cell 0: Update GPU-accelerated stages to include V3 `sentence-transformers` model.
- NB2 cell 8: Add `SemanticEmbedder.to(device)` code note.
- NB2 cell 13: Add Semantic Embedder row to GPU applicability chart.
- Issues addressed: N2-01, N2-08, N2-10.

### Phase 8 — Nice-to-Have Diagram Updates (NB1 + NB2)
- NB1 cell 7: Add V3 semantic enrichment box to architecture diagram.
- NB1 cell 9: Add V3 stage to V2 pipeline diagram.
- NB1 cell 13: Add Semantic Layer bar to health status chart.
- NB2 cell 10: Add Semantic Embedder to latency breakdown chart.
- **Lower priority** — visual polish only, no factual impact.
- Issues addressed: N1-11, N1-14, N1-19, N2-09.

---

*End of audit report. No notebook changes have been made. Update plan is ready for review.*
