# V3 Architecture — Semantic Enrichment Layer

**Version:** Phase 8
**Date:** 2026-03-30

---

## 1. Overview

The V3 semantic layer augments confirmed anomalies with human-readable explanations and
cosine-similarity-based historical context. It is **additive and backward-compatible**: the
V1/V2 inference backbone is untouched; semantic enrichment happens only after an alert has
been confirmed by the existing decision logic.

The layer is **disabled by default** (`SEMANTIC_ENABLED=false`) and safe to run in any
environment that does not have GPU or the `sentence-transformers` library installed.

---

## 2. Component Map

```
src/semantic/
├── config.py          SemanticConfig       — env-var-driven config (5 fields)
├── loader.py          SemanticModelLoader  — lazy, idempotent model loader
├── embeddings.py      SemanticEmbedder     — LRU-cached sentence embedding
├── similarity.py      SemanticSimilarity   — cosine similarity + top-k ranking
└── explainer.py       RuleBasedExplainer   — 3-rule heuristic explanation generator
```

---

## 3. Configuration

All fields read from environment variables. Safe defaults keep the layer inert.

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_ENABLED` | `false` | Master switch. `true` activates the full layer. |
| `SEMANTIC_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model identifier (CPU-friendly). |
| `EXPLANATION_ENABLED` | `false` | Enables explanation sub-system (requires `SEMANTIC_ENABLED`). |
| `EXPLANATION_MODEL` | `rule-based` | Explanation back-end. Only `rule-based` implemented in V3. |
| `SEMANTIC_CACHE_SIZE` | `1000` | LRU cache size for (text → embedding) pairs. |

---

## 4. Pipeline Integration Flow

```
POST /ingest (or /v3/ingest)
  │
  ├── InferenceEngine.ingest()          [V1/V2 — unchanged]
  │     └── score_baseline / score_transformer / ensemble
  │     └── decide() → is_anomaly = (score >= threshold)
  │
  ├── AlertManager.emit()               [V1/V2 — unchanged]
  │     └── cooldown / dedup → Alert
  │
  └── Pipeline._enrich_alert(alert)     [V3 — NEW, gated by SEMANTIC_ENABLED]
        if SEMANTIC_ENABLED=false → return immediately (zero side effects)
        │
        ├── RuleBasedExplainer.explain(evidence_window)
        │     → explanation (str)
        │     → evidence_tokens (list[str])
        │
        ├── SemanticEmbedder.embed(alert_text)
        │     → np.ndarray [384] | None
        │
        ├── SemanticSimilarity.top_k(embedding, history, k=3)
        │     → top_similar_events (list[dict])
        │     → semantic_similarity = top[0]["score"]
        │
        └── _semantic_history.append((label, embedding))
              (deque maxlen=200 — in-memory, resets on restart)
```

---

## 5. RuleBasedExplainer Rules

Applied in order; multiple rules can fire simultaneously.

| Rule | Trigger | Output |
|------|---------|--------|
| Error keyword | `templates_preview` contains error/exception/fail/timeout/… | Lists error templates in `evidence_tokens` |
| Low diversity | < 3 distinct template types in window | Flags in `explanation` |
| High density | `token_count` > 40 | Flags in `explanation` |
| Empty window | `token_count` == 0 | Flags in `explanation` |
| Fallback | No rule fires | Generic "anomaly score exceeded threshold" sentence |

---

## 6. V3 API Endpoints

All endpoints are under `/v3/` prefix and are additive (no existing routes modified).

### `POST /v3/ingest`

Functionally identical to `POST /ingest`. Semantic fields appear in the `alert` object
when `SEMANTIC_ENABLED=true`.

**Response `alert` object additions:**

```json
{
  "explanation": "Detected 2 error-indicative log template(s).",
  "evidence_tokens": ["SSH: exception in handler", "disk timeout on /dev/sda"],
  "semantic_similarity": 0.8421,
  "top_similar_events": [
    {"text": "hdfs:a1b2c3d4", "score": 0.8421},
    {"text": "auth:e5f6g7h8", "score": 0.7203}
  ]
}
```

### `GET /v3/alerts/{alert_id}/explanation`

Returns semantic enrichment fields for a specific alert from the ring buffer.

```json
{
  "alert_id": "550e8400-e29b-41d4-a716-446655440000",
  "semantic_enabled": true,
  "explanation": "Detected 2 error-indicative log template(s).",
  "evidence_tokens": ["SSH: exception in handler"],
  "semantic_similarity": 0.8421,
  "top_similar_events": [...]
}
```

Returns `404` if the alert is no longer in the ring buffer.
Returns `semantic_enabled: false` with null fields when the layer is disabled.

### `GET /v3/models/info`

Returns a diagnostics snapshot of the inference engine and semantic layer.

```json
{
  "inference_mode": "ensemble",
  "artifacts_loaded": true,
  "semantic_enabled": false,
  "semantic_model": "all-MiniLM-L6-v2",
  "semantic_model_loaded": false,
  "explanation_enabled": false,
  "v2_engine_available": false
}
```

---

## 7. Observability Metrics

Three Prometheus metrics are added for the V3 layer.

| Metric | Type | Description |
|--------|------|-------------|
| `semantic_enrichments_total` | Counter | Total alerts enriched (increments each time `_enrich_alert` runs with `SEMANTIC_ENABLED=true`) |
| `semantic_enrichment_latency_seconds` | Histogram | Wall-clock time of each enrichment call |
| `semantic_model_ready` | Gauge | `1` when the sentence-transformers model is loaded, `0` otherwise |

These metrics are exposed at `GET /metrics` alongside existing V1/V2 metrics and are
available in the Grafana dashboard `v3_semantic_observability.json`.

---

## 8. Health Endpoint

`GET /health` now includes a `semantic` component:

```json
{
  "status": "healthy",
  "components": {
    "inference_engine": {"status": "ok", "artifacts_loaded": true},
    "alert_manager":    {"status": "ok"},
    "alert_buffer":     {"status": "ok", "size": 3},
    "semantic": {
      "enabled": false,
      "model_loaded": false,
      "model_name": "all-MiniLM-L6-v2"
    }
  }
}
```

The `semantic` component is informational — it never degrades the overall `status`.

---

## 9. Docker / Deployment

### Environment variables (docker-compose.yml)

```yaml
SEMANTIC_ENABLED: "false"           # set to "true" to enable
HF_HOME: "/app/.cache/huggingface"  # model cache path inside container
```

### Volume mount for model cache

```yaml
volumes:
  - ../hf_cache:/app/.cache/huggingface
```

Persists the downloaded model across container restarts (`~90 MB` for `all-MiniLM-L6-v2`).

### CPU-only default

The default model (`all-MiniLM-L6-v2`) runs fully on CPU without GPU drivers.
No `CUDA_VISIBLE_DEVICES` or `torch` GPU configuration is required.

---

## 10. What Is Deferred (Post-V3)

| Feature | Notes |
|---------|-------|
| LLM-backed explanation (`EXPLANATION_MODEL=llm`) | Requires Anthropic/OpenAI API key |
| Persistent semantic history (across restarts) | Requires a vector store (FAISS / ChromaDB) |
| `/v3/search` semantic log search endpoint | Depends on persistent history |
| Semantic alert routing rules | Depends on reliable embedding quality at scale |
