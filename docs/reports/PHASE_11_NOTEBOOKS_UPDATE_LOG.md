---
name: Notebooks V3 Update Log
description: Chronological record of Phase 11 notebook edits — what changed, in which cell, and why
---

# Notebooks V3 Update Log

Tracks all Phase 11 edits to `notebooks/predictive_log_anomaly_engine_demo.ipynb` (NB1) and `notebooks/predictive_log_anomaly_engine_gpu_demo.ipynb` (NB2).

Reference: `docs/NOTEBOOKS_V3_AUDIT_REPORT.md` — full pre-edit audit with cell-level issue inventory.

---

## Phase 0 — Baseline Documentation

**Date:** 2026-03-30
**Notebook edits:** None
**Action:** Pre-edit audit completed and written to `docs/NOTEBOOKS_V3_AUDIT_REPORT.md`. All issues catalogued by cell ID, priority, and phase assignment.

---

## Phase 1 — Version Label & Test Count Updates

**Date:** 2026-03-30
**Goal:** Replace all stale `v2` version labels with `v3`; update test count 578 → 557 everywhere it appears.

### NB1 — `predictive_log_anomaly_engine_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `md-title-0001` | [0] | Project row: `v2` → `v3`; Architecture row: added V3 Semantic Layer; Pipeline Stages: added Semantic Enrichment (optional); Test Coverage: 578 → 557 |
| `md-why-000005` | [4] | Test count bullet: 578 → 557 |
| `md-valid-00013` | [12] | Test Suite row: 578 → 557 |
| `md-outco-00020` | [19] | Test safety net row: 578 → 557; footer prose: 578 → 557 |
| `md-summa-00021` | [20] | Footer: `v2` → `v3`; test count: 578 → 557 |

### NB2 — `predictive_log_anomaly_engine_gpu_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `gpu-md-title-001` | [0] | CPU-Only Stages: added `Semantic Enrichment`; added V3 Semantic Stage row |
| `gpu-md-summ-0019` | [18] | Footer: `v2` → `v3`; added `+ Semantic Layer` to architecture label |

---

## Phase 2 — Severity Label Disambiguation

**Date:** 2026-03-31
**Goal:** Clearly distinguish V1 AlertPolicy labels (`critical/high/medium/low`) from V2 ML Severity Classifier labels (`info/warning/critical`) in all cells where they appear.

### NB1 — `predictive_log_anomaly_engine_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `md-why-000005` | [4] | Severity bullet rewritten: "V2 severity classification (`info` / `warning` / `critical`) — trained ML labels; V1 uses `critical` / `high` / `medium` / `low` (threshold-based AlertPolicy)" |
| `md-v2pi-00009` | [8] | Stage 4 output qualified: `info` / `warning` / `critical` (V2 ML labels); added Note block distinguishing V2 ML classifier from V1 AlertPolicy |
| `code-seve-00018` | [18] | Donut chart title updated to "V2 Severity Distribution (Classifier Output)\n(Simulated Demo Data)" — unambiguously scoped to V2 classifier output |

### NB2 — No changes (severity labelling not present in NB2 cells)

---

## Phase 3 — Architecture Description Updates

**Date:** 2026-03-31
**Goal:** Add V3 semantic enrichment layer to all pipeline/architecture descriptions across both notebooks.

### NB1 — `predictive_log_anomaly_engine_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `md-exec-00003` | [2] | Added V3 Semantic Layer paragraph: optional enrichment stage, `SEMANTIC_ENABLED=true`, `all-MiniLM-L6-v2`, disabled-by-default note |
| `md-arch-00007` | [6] | Added V3 Semantic row to architecture table: `SEMANTIC_ENABLED=true` \| Alert output from V1/V2 \| Explanation + similarity enrichment (optional overlay) |
| `md-v2pi-00009` | [8] | Added step 9 to inference flow: V3 Semantic Layer enriches alert when `SEMANTIC_ENABLED=true` → `explanation`, `evidence_tokens`, `semantic_similarity`, `top_similar_events` |
| `md-outco-00020` | [19] | Added V3 Semantic Layer row to Key Outcomes table; added NLP integration note to "What this demonstrates" list |
| `md-summa-00021` | [20] | Added V3 Semantic Layer paragraph to Final Summary; updated footer to include `LSTM Autoencoder + V3 Semantic Layer` |

### NB2 — `predictive_log_anomaly_engine_gpu_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `gpu-md-over-002` | [2] | Added V3 Semantic Layer paragraph: `sentence-transformers`, GPU support via `.to(device)`, CPU default |
| `gpu-md-rela-004` | [4] | Updated CPU path code block to include "All V2 pipeline stages + optional V3 Semantic Enrichment"; added V3 Semantic Embedder line to GPU Extension block |

---

## STOP — Manual Review After Phase 3

**Status:** STOPPED — awaiting user approval before Phase 4

**Items for reviewer to verify:**
1. Test count (557) is correct in all updated cells ✓ (Phase 1 set this)
2. V2 severity label vocabulary (`info`/`warning`/`critical`) — confirm against `src/modeling/severity_classifier.py`
3. V3 endpoint paths (`/v3/ingest`, `/v3/alerts/{id}/explanation`, `/v3/models/info`) — confirm against `src/api/routes_v3.py`
4. All notebook JSON is valid (no structural corruption from edits)
5. Prose changes in cells 2, 6, 8, 19, 20 (NB1) and cells 2, 4 (NB2) read correctly

---

## Phase 4 — Endpoint & Metrics Tables (NB1 cell [10])

**Date:** 2026-03-31
**Goal:** Add three missing V3 endpoints and four missing Prometheus metrics to NB1 cell [10].

### NB1 — `predictive_log_anomaly_engine_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `md-runt-00011` | [10] | Added "V3 Endpoints (Semantic Layer)" section: `POST /v3/ingest`, `GET /v3/alerts/{alert_id}/explanation`, `GET /v3/models/info` with descriptions and prerequisite note |
| `md-runt-00011` | [10] | Added Sample V3 Alert Response JSON block showing `explanation`, `evidence_tokens`, `semantic_similarity`, `top_similar_events` fields |
| `md-runt-00011` | [10] | Added four missing metrics to Prometheus table: `ingest_errors_total` (Counter), `semantic_enrichments_total` (Counter), `semantic_enrichment_latency_seconds` (Histogram), `semantic_model_ready` (Gauge) |

---

## Phase 5 — Execution Path Diagram (NB2 cells [6], [7])

**Date:** 2026-03-31
**Goal:** Add "Semantic Enrichment (V3)" as a sixth stage to both the text and the matplotlib bar chart in NB2.

### NB2 — `predictive_log_anomaly_engine_gpu_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `gpu-md-exec-006` | [6] | Added V3 Semantic Enrichment note to section intro: "CPU by default and is GPU-capable via `.to(device)`" |
| `gpu-code-exec-007` | [7] | Added sixth entry to `cpu_stages`: `('Semantic Enrichment  (V3)', CPU_COLOR, False)` with comment "optional — CPU default" |
| `gpu-code-exec-007` | [7] | Added sixth entry to `gpu_stages`: `('Semantic Enrichment  (V3)', CPU_COLOR, False)` with comment "GPU-capable via .to(device)" |
| `gpu-code-exec-007` | [7] | Updated chart title to "CPU vs. GPU Execution Path — Full Inference Pipeline (V2 + V3 Semantic)" |

---

## STOP — Manual Review After Phase 5

**Status:** STOPPED — awaiting user approval before Phase 6

**Items for reviewer to verify:**
1. NB1 cell [10] — V3 endpoint paths and response schema fields match live `src/api/routes_v3.py`
2. NB1 cell [10] — all 10 Prometheus metrics listed match `MetricsRegistry` in `src/monitoring/`
3. NB2 cell [7] — chart renders with 6 boxes per column (not 5); Semantic Enrichment box appears at bottom in both CPU and GPU columns
4. NB2 cell [7] — GPU badge bracket still correctly covers only LSTM and Autoencoder rows (rows 2 and 3), not the Semantic Enrichment row
5. Both notebooks open without JSON errors in Jupyter

---

## Phase 6 — Final Summaries

**Date:** 2026-03-31
**Goal:** Update final summaries and key outcome tables in both notebooks to reflect the triple-layer architecture and V3 semantic enrichment.

### NB1 — `predictive_log_anomaly_engine_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `md-outco-00020` | [19] | "Dual pipeline architecture" → row replaced with V3 Semantic Layer row; added NLP integration bullet to "What this demonstrates" |
| `md-summa-00021` | [20] | Added V3 Semantic Layer paragraph; footer updated to `v3 · LSTM Autoencoder + V3 Semantic Layer · 557 Tests Passing` |

### NB2 — `predictive_log_anomaly_engine_gpu_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `gpu-md-port-017` | [17] | Added NLP & semantic enrichment row: `sentence-transformers`, cosine similarity, evidence extraction (V3) |
| `gpu-md-summ-018` | [18] | V3 Semantic Layer paragraph added to final summary; footer updated to `v3 · LSTM + Autoencoder + Semantic Layer` |

---

## Phase 7 — GPU V3 Additions

**Date:** 2026-03-31
**Goal:** Add V3 `sentence-transformers` as a third GPU-eligible model to GPU-specific cells in NB2.

### NB2 — `predictive_log_anomaly_engine_gpu_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `gpu-md-title-001` | [0] | Added `V3 Semantic Stage` row: `all-MiniLM-L6-v2 sentence-transformers — CPU default, GPU-capable`; `Semantic Enrichment` added to CPU-Only Stages |
| `gpu-md-runt-008` | [8] | Added V3 Semantic Layer `.to(device)` code block after existing GPU pattern; shows `SentenceTransformer("all-MiniLM-L6-v2").to(device)` |
| `gpu-code-appl-013` | [13] | Added `'Semantic Embedder (V3)'` to `stages` list; `cpu_pct=82`, `gpu_pct=18` (CPU default, GPU-capable — illustrative) |

---

## Phase 8 — Nice-to-Have Diagram Updates

**Date:** 2026-03-31
**Goal:** Add V3 semantic enrichment stage to three NB1 diagrams; fix stale test count in health chart.

### NB1 — `predictive_log_anomaly_engine_demo.ipynb`

| Cell ID | Index | Change |
|---|---|---|
| `code-arch-00008` | [7] | `figsize` extended `(16,4.2)→(18,4.2)`; `xlim` `(0,16)→(0,18)`; added `(15.8, 2.6, 'Semantic\n(V3)', TEAL)` box to pipeline; added optional annotation; title updated to include "(V1 + V2 + V3 Semantic Layer)" |
| `code-v2pi-00010` | [9] | `figsize` extended `(14,5.5)→(16,5.5)`; `xlim` `(0,14)→(0,16)`; added `(15.3, 4.2, 'V3', 'Semantic', 'Enrichment (optional)', TEAL, 'SEMANTIC_ENABLED=true')` stage; title updated |
| `code-heal-00014` | [13] | Fixed stale `578 passing` → `557 passing`; added `'Semantic Layer  (V3)'` row |

### NB2 — No changes
N2-09 (latency breakdown chart) is not applicable — NB2 cell [10] contains throughput-vs-batch-size charts, not a stage latency breakdown. Existing GPU applicability chart updated in Phase 7 (cell [13]) covers GPU stage coverage for the Semantic Embedder.

---

## Overall Final Status

| Phase | Status | Notebooks touched |
|---|---|---|
| Phase 0 | ✅ Complete | — (audit only) |
| Phase 1 | ✅ Complete | NB1, NB2 |
| Phase 2 | ✅ Complete | NB1 |
| Phase 3 | ✅ Complete | NB1, NB2 |
| Phase 4 | ✅ Complete | NB1 |
| Phase 5 | ✅ Complete | NB2 |
| Phase 6 | ✅ Complete | NB1, NB2 |
| Phase 7 | ✅ Complete | NB2 |
| Phase 8 | ✅ Complete | NB1 |

### Unresolved / Manual Follow-Up

| Item | Notebook | Notes |
|---|---|---|
| Rendered cell outputs are stale | NB1, NB2 | Run "Restart & Clear Output" before final re-run — existing stored outputs reference old chart states |
| N2-09 latency breakdown | NB2 | Cell [10] is throughput chart — if a separate stage latency chart for NB2 is added in future, include Semantic Embedder bar at that time |
| GPU SBERT benchmarks | NB2 | All GPU performance figures remain conceptual/illustrative; real measurements require CUDA hardware |
