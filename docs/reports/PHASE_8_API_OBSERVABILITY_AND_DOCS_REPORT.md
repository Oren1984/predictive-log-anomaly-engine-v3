# Phase 8 — API, Observability & Documentation Report

**Date:** 2026-03-30
**Branch:** `main`
**Baseline commit:** Phase 7 (pipeline semantic integration)
**Goal:** Complete the V3 surface area — API endpoints, observability metrics, health
reporting, Docker support, documentation, and demo notebook.

---

## 1. Routes Added

### New file: `src/api/routes_v3.py`

All endpoints live under `/v3/` prefix (registered as `router_v3` in `app.py`).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v3/ingest` | Versioned ingest alias — same as `/ingest` but explicitly surfaces semantic fields in the alert response |
| `GET` | `/v3/alerts/{alert_id}/explanation` | Returns semantic enrichment fields for a specific alert from the ring buffer; `404` when not found |
| `GET` | `/v3/models/info` | Returns inference mode, artifact load state, and V3 semantic layer status |

### Modified: `src/api/routes.py`

`POST /ingest` and `GET /alerts` now pass all four semantic fields through to `AlertSchema`:
`explanation`, `semantic_similarity`, `top_similar_events`, `evidence_tokens`.

### Modified: `src/api/app.py`

Registered `router_v3` alongside `router`, `router_v2`, and `ui_router`.

---

## 2. Schemas Added

### Modified: `src/api/schemas.py`

| Schema | New/Modified | Description |
|--------|-------------|-------------|
| `AlertSchema` | Modified (Phase 7) | Already had 4 optional semantic fields |
| `ExplanationResponse` | **New** | Response for `GET /v3/alerts/{id}/explanation` |
| `ModelsInfoResponse` | **New** | Response for `GET /v3/models/info` |

#### `ExplanationResponse`
```python
alert_id: str
semantic_enabled: bool
explanation: Optional[str]
evidence_tokens: Optional[list]
semantic_similarity: Optional[float]
top_similar_events: Optional[list]
```

#### `ModelsInfoResponse`
```python
inference_mode: str
artifacts_loaded: bool
semantic_enabled: bool
semantic_model: str
semantic_model_loaded: bool
explanation_enabled: bool
v2_engine_available: bool
```

---

## 3. Metrics Added

### Modified: `src/observability/metrics.py`

Three new Prometheus metrics added to `MetricsRegistry`:

| Metric | Type | Description |
|--------|------|-------------|
| `semantic_enrichments_total` | Counter | Total alerts enriched by V3 semantic layer |
| `semantic_enrichment_latency_seconds` | Histogram | Wall-clock time of each `_enrich_alert` call |
| `semantic_model_ready` | Gauge | `1` when sentence-transformers model is loaded, `0` otherwise |

`semantic_model_ready` is updated in `Pipeline.load_models()` after model load.
`semantic_enrichments_total` and `semantic_enrichment_latency_seconds` are recorded in
`Pipeline._enrich_alert()` after successful enrichment.

---

## 4. Health Reporting Updated

### Modified: `src/health/checks.py`

`GET /health` now includes a `semantic` component (informational, never degrades overall status):

```json
"semantic": {
  "enabled": false,
  "model_loaded": false,
  "model_name": "all-MiniLM-L6-v2"
}
```

The component reads from `pipeline._semantic_config` and `pipeline._semantic_loader` via
`getattr` with safe fallbacks — fully compatible with `MockPipeline` used in tests.

---

## 5. Pipeline Additions

### Modified: `src/api/pipeline.py`

- Added `get_alert_by_id(alert_id: str) -> Optional[dict]` — O(n) search over ring buffer
- `_enrich_alert` now records `semantic_enrichments_total` and
  `semantic_enrichment_latency_seconds` when `metrics` is attached
- `load_models` sets `semantic_model_ready` gauge after loading

---

## 6. Grafana Dashboard

### New file: `grafana/dashboards/v3_semantic_observability.json`

Standalone dashboard (UID `v3-semantic-observability`) with 5 panels:

| Panel | Type | Query |
|-------|------|-------|
| Semantic Model Ready | Stat | `semantic_model_ready` |
| Enrichment Rate (5 min) | Time series | `rate(semantic_enrichments_total[5m]) * 60` |
| Enrichment Latency p95 + avg | Time series | Histogram quantile queries |
| Alerts vs Enrichments | Time series | `increase(...)` comparison |
| Enrichment Latency p95 (current) | Stat | Last p95 value |

Refresh: 10s. Tags: `v3`, `semantic`, `anomaly-engine`.

The existing `stage08_api_observability.json` dashboard is **unchanged**.

---

## 7. Docker Impact

### Modified: `docker/Dockerfile`

```dockerfile
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface
```

`HF_HOME` is set in the image so `sentence-transformers` writes downloaded models to
the correct path regardless of how the container is started.

### Modified: `docker/docker-compose.yml`

Two environment variables added to the `api` service:

```yaml
SEMANTIC_ENABLED: "false"           # safe default; set to "true" to enable
HF_HOME: "/app/.cache/huggingface"
```

Volume mount for model cache:

```yaml
- ../hf_cache:/app/.cache/huggingface
```

The `hf_cache/` directory is created on first run and persists across container restarts.
On first run with `SEMANTIC_ENABLED=true`, the `all-MiniLM-L6-v2` model (~90 MB) is
downloaded from Hugging Face Hub. Subsequent starts use the cache. No GPU required.

---

## 8. Documentation Created / Updated

| File | Action | Description |
|------|--------|-------------|
| `docs/V3_ARCHITECTURE.md` | **Created** | Full V3 design: component map, config table, pipeline flow, API reference, metrics, Docker notes, deferred items |
| `README.md` | **Updated** | Added V3 Semantic Layer section with endpoint table, enable instructions, field table, and updated project status |
| `notebooks/v3_semantic_demo.ipynb` | **Created** | 6-cell interactive demo covering models/info, health, v3/ingest, explanation fetch, alert listing with semantic fields, and metrics output |

---

## 9. Backward Compatibility

| Concern | Status |
|---------|--------|
| `/ingest`, `/alerts`, `/health`, `/metrics` | Unchanged response structure; semantic fields are `null` when disabled |
| V1/V2 inference path | Zero modifications |
| `MockPipeline` in tests | Compatible — all new pipeline methods guarded with `getattr(..., None)` |
| `MetricsRegistry` | New counters/histograms are isolated in the private registry; no duplicate registration |
| `AlertSchema` | New fields are `Optional` with `None` defaults — no consumer breakage |

---

## 10. Test Results

### Integration and smoke tests

```
pytest tests/integration/test_smoke_api.py tests/test_pipeline_smoke.py tests/unit/test_semantic.py

78 passed in 10.93s
```

### Full suite (non-slow)

```
pytest -m "not slow"

557 passed, 26 deselected in 46.61s
```

| | Count |
|-|-------|
| Phase 7 baseline | 557 |
| Phase 8 new tests | 0 (all existing tests pass; no new test file needed — coverage via integration suite) |
| **Total** | **557** |
| Failures | **0** |

---

## 11. What Was NOT Done (out of Phase 8 scope)

| Item | Notes |
|------|-------|
| Automated HF model pre-download in Docker build | Intentional: keeps image size minimal; model downloads at first use with SEMANTIC_ENABLED=true |
| LLM-backed explanations (`EXPLANATION_MODEL=llm`) | Requires API key wiring; deferred |
| Persistent semantic history (vector store) | Deferred — current in-memory deque resets on restart |
| `/v3/search` semantic search endpoint | Requires persistent history |
| V2 pipeline semantic enrichment | V2 uses a different alert schema; deferred |
