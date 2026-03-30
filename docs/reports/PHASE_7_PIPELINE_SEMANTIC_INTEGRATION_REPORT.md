# Phase 7 — Pipeline Semantic Integration Report

**Date:** 2026-03-30
**Branch:** `main`
**Baseline commit:** `651d663` (Phase 6 scaffold)
**Goal:** Integrate the V3 semantic enrichment layer into the pipeline in an additive,
backward-compatible way. The existing V1/V2 inference behaviour is preserved exactly;
all new fields are optional and gated by `SEMANTIC_ENABLED`.

---

## 1. Files Created

| File | Purpose |
|------|---------|
| `src/semantic/similarity.py` | `SemanticSimilarity` — cosine similarity computation and top-k ranking |
| `src/semantic/explainer.py` | `RuleBasedExplainer` — rule-based explanation generator from evidence_window |

---

## 2. Files Modified

| File | Change |
|------|--------|
| `src/semantic/__init__.py` | Added exports for `SemanticSimilarity`, `RuleBasedExplainer` |
| `src/alerts/models.py` | Added 4 optional semantic fields to `Alert`; updated `to_dict()` |
| `src/api/schemas.py` | Added 4 optional fields to `AlertSchema` (Pydantic v2) |
| `src/api/pipeline.py` | Imported semantic components; added `_enrich_alert()` method; called after alert confirmation |
| `tests/unit/test_semantic.py` | Added `TestSemanticSimilarity` (8 tests) and `TestRuleBasedExplainer` (10 tests) |

---

## 3. New Optional Fields on Alert

All four fields are added to both `Alert` (dataclass) and `AlertSchema` (Pydantic) with
`default=None`. Existing clients that do not read these keys are unaffected.

| Field | Type | When populated |
|-------|------|----------------|
| `explanation` | `str \| None` | When `SEMANTIC_ENABLED=true`; rule-based text describing the anomaly |
| `semantic_similarity` | `float \| None` | When enabled and ≥1 prior alert in history; cosine similarity score to most similar past alert |
| `top_similar_events` | `list \| None` | When enabled; top-3 most similar historical alerts (`[{"text": str, "score": float}, …]`) |
| `evidence_tokens` | `list \| None` | When enabled; template snippets that triggered explanation rules |

---

## 4. Architecture

### 4.1 `SemanticSimilarity` (`src/semantic/similarity.py`)

Stateless utility class. Two methods:

- `compute(a, b) -> float` — cosine similarity in [-1, 1]; returns 0.0 on zero vectors.
- `top_k(query, candidates, k=3) -> list[dict]` — ranks candidate `(label, embedding)` pairs by similarity; returns top-k as `[{"text": str, "score": float}]`.

### 4.2 `RuleBasedExplainer` (`src/semantic/explainer.py`)

Stateless utility class. Single public method:

- `explain(evidence_window: dict) -> dict` — applies three rules in order:
  1. **Error-keyword scan** — scans `templates_preview` for error/exception/fail/timeout/… keywords
  2. **Template diversity** — flags windows with < 3 distinct template types
  3. **Window density** — flags windows with token_count > 40 or == 0

  Returns `{"explanation": str, "evidence_tokens": list[str]}`.
  Falls back to a generic sentence when no rule fires.

### 4.3 Pipeline integration (`src/api/pipeline.py`)

Enrichment is entirely post-anomaly. The call order is:

```
Pipeline.process_event(event)
  → InferenceEngine.ingest()       [unchanged V1/V2 path]
  → AlertManager.emit()            [unchanged — anomaly gate]
  → Pipeline._enrich_alert(alert)  [NEW — only when alert fired]
      if not semantic_enabled → return immediately (inert)
      RuleBasedExplainer.explain(evidence_window) → explanation, evidence_tokens
      SemanticEmbedder.embed(alert_text)           → embedding | None
      SemanticSimilarity.top_k(embedding, history) → top_similar_events
      alert.semantic_similarity = top[0]["score"]
      _semantic_history.append((label, embedding))
  → N8nWebhookClient.send(alert)
```

The semantic history is a `deque(maxlen=200)` scoped to the `Pipeline` instance —
no disk I/O, no external state.

---

## 5. Backward Compatibility

| Concern | How it is preserved |
|---------|---------------------|
| V1/V2 inference scores | `InferenceEngine._build_result()` is untouched |
| Alert schema | New fields have `default=None`; absent from payloads when null is omitted by the caller |
| API responses | `AlertSchema` new fields all `Optional` with `None` default — no breaking change to existing consumers |
| `Alert.to_dict()` | Always includes all 4 new keys (set to `null`) — no client can break from a missing key |
| `SemanticConfig` defaults | `semantic_enabled=False` → `_enrich_alert` returns immediately without touching any component |
| Import safety | `sentence_transformers` is never imported at module level |

---

## 6. Confirmation: Disabled Flow Is Inert

With `SEMANTIC_ENABLED=false` (the default):

```python
Pipeline._enrich_alert(alert)
  └── if not self._semantic_config.semantic_enabled:
          return   ← exits immediately, zero side effects
```

- No model is loaded.
- No imports of `sentence_transformers` occur.
- Alert fields `explanation`, `semantic_similarity`, `top_similar_events`, `evidence_tokens` remain `None`.
- All existing tests pass with no change in behaviour.

---

## 7. Test Results

### Semantic unit tests

```
pytest tests/unit/test_semantic.py -v

44 passed in 0.50s
```

| Test class | Tests | Result |
|-----------|-------|--------|
| `TestSemanticConfig` | 9 | ✓ all pass |
| `TestSemanticModelLoader` | 6 | ✓ all pass |
| `TestSemanticEmbedderDisabled` | 3 | ✓ all pass |
| `TestSemanticEmbedderEnabled` | 6 | ✓ all pass |
| `TestSemanticPackageImport` | 2 | ✓ all pass |
| `TestSemanticSimilarity` | 8 | ✓ all pass (Phase 7 new) |
| `TestRuleBasedExplainer` | 10 | ✓ all pass (Phase 7 new) |

### Full suite (non-slow)

```
pytest -m "not slow"

557 passed, 26 deselected in 56.19s
```

| | Count |
|-|-------|
| Phase 6 baseline | 539 |
| Phase 7 new semantic tests | +18 |
| **Total** | **557** |
| Failures | **0** |

---

## 8. What Was NOT Done (out of Phase 7 scope)

| Item | Deferred to |
|------|------------|
| `/semantic` or `/explain` API routes | Phase 8+ |
| `explanation_model="llm"` back-end | Phase 8+ |
| Persist semantic history across restarts | Phase 8+ |
| Add `SemanticConfig` fields to `src/api/settings.py` (auto-wiring) | Phase 8+ |
| Hugging Face model hub integration | Phase 8+ |
| V2 pipeline (`src/runtime/pipeline_v2.py`) enrichment | Phase 8+ |
