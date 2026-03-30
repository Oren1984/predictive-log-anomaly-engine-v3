# Phase 6 — V3 Semantic Scaffold Report

**Date:** 2026-03-30
**Branch:** `main`
**Commit baseline:** `d975042`
**Purpose:** Create the initial V3 semantic scaffolding without integrating it into the runtime pipeline or API.

---

## 1. Files Created

### Package: `src/semantic/`

| File | Purpose |
|------|---------|
| `src/semantic/__init__.py` | Package entry point; re-exports `SemanticConfig`, `SemanticModelLoader`, `SemanticEmbedder` |
| `src/semantic/config.py` | `SemanticConfig` dataclass — all five env-var-driven config fields |
| `src/semantic/loader.py` | `SemanticModelLoader` — lazy, idempotent loader for the sentence-transformers model |
| `src/semantic/embeddings.py` | `SemanticEmbedder` — LRU-cached embedding computation; inert when disabled |

### Test file

| File | Tests |
|------|-------|
| `tests/unit/test_semantic.py` | 26 tests across 5 classes |

### Modified files

| File | Change |
|------|--------|
| `requirements/requirements.txt` | Added `sentence-transformers>=2.7.0` |
| `.env.example` | Added V3 semantic section with 5 documented variables |

---

## 2. Configuration Variables

All five variables are defined in `SemanticConfig` and documented in `.env.example`:

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SEMANTIC_ENABLED` | `false` | bool | Master switch. When `false` the entire layer is inert. |
| `SEMANTIC_MODEL` | `all-MiniLM-L6-v2` | str | Sentence-Transformers model identifier. Only loaded when enabled. |
| `EXPLANATION_ENABLED` | `false` | bool | Enables the explanation sub-system. Requires `SEMANTIC_ENABLED=true`. |
| `EXPLANATION_MODEL` | `rule-based` | str | Explanation strategy. `"rule-based"` uses heuristic template matching. |
| `SEMANTIC_CACHE_SIZE` | `1000` | int | LRU cache capacity for (text → embedding) pairs. `0` disables caching. |

---

## 3. Architecture

### Design principles

- **Lazy import:** `sentence_transformers` is imported inside `SemanticModelLoader.load()`, never at module level. The package is safe to import in environments where `sentence-transformers` is not installed (e.g., CI fast suite without GPU).
- **Inert by default:** Every public method checks `semantic_enabled` as its first step and returns a safe null value (`None`, `[]`) without touching the model.
- **No pipeline coupling:** `src/semantic/` has zero imports from `src/api/`, `src/runtime/`, `src/alerts/`, or any other live module. Integration is deferred to a later phase.
- **Idempotent load:** Calling `loader.load()` twice is safe — the second call is a no-op.
- **LRU caching:** `SemanticEmbedder` wraps `_embed_raw` in `functools.lru_cache` with `maxsize=semantic_cache_size`. Cache size is set at construction time from config.

### Call flow when disabled (default)

```
SemanticEmbedder.embed(text)
  └── if not config.semantic_enabled → return None   ← exits immediately
```

### Call flow when enabled

```
SemanticModelLoader.load()
  └── from sentence_transformers import SentenceTransformer
  └── self._model = SentenceTransformer(config.semantic_model)

SemanticEmbedder.embed(text)
  └── if not config.semantic_enabled → return None   (skipped when enabled)
  └── self._embed_cached(text)
        └── if not loader.is_ready → return None
        └── loader.model.encode(text) → np.ndarray [dim]
```

---

## 4. Proof That the Layer Is Disabled by Default

### 4.1 `SemanticConfig` defaults

```python
cfg = SemanticConfig()
assert cfg.semantic_enabled is False      # ✓
assert cfg.explanation_enabled is False   # ✓
```

### 4.2 `SemanticModelLoader` — no-op when disabled

```python
loader = SemanticModelLoader(SemanticConfig())
loader.load()          # no-op: sentence-transformers is never imported
assert loader.is_ready is False   # ✓
assert loader.model is None       # ✓
```

### 4.3 `SemanticEmbedder` — null outputs when disabled

```python
embedder = SemanticEmbedder()
assert embedder.embed("any text") is None    # ✓
assert embedder.embed_batch(["a", "b"]) == []  # ✓
```

### 4.4 Module import is always safe

```python
from src.semantic import SemanticConfig, SemanticModelLoader, SemanticEmbedder
# ↑ no sentence-transformers import occurs, no model is loaded
```

All four claims are covered by tests in `TestSemanticConfig`, `TestSemanticModelLoader`, `TestSemanticEmbedderDisabled`, and `TestSemanticPackageImport`.

---

## 5. Test Results

### Semantic-specific tests

```
pytest tests/unit/test_semantic.py -v

26 passed in 0.59s
```

| Test class | Tests | Result |
|-----------|-------|--------|
| `TestSemanticConfig` | 9 | ✓ all pass |
| `TestSemanticModelLoader` | 6 | ✓ all pass |
| `TestSemanticEmbedderDisabled` | 3 | ✓ all pass |
| `TestSemanticEmbedderEnabled` | 6 | ✓ all pass |
| `TestSemanticPackageImport` | 2 | ✓ all pass |

### Full suite

```
pytest -m "not slow"

539 passed, 26 deselected in 36.86s
```

| | Count |
|-|-------|
| Phase 4 baseline | 513 |
| New semantic tests | +26 |
| **Total** | **539** |
| Failures | **0** |

---

## 6. What Was NOT Done (per Phase 6 scope)

| Item | Deferred to |
|------|------------|
| Connect `SemanticEmbedder` to `InferenceEngine` | Phase 7+ |
| Add `/semantic` or `/explain` API routes | Phase 7+ |
| Implement `explanation_model="llm"` back-end | Phase 7+ |
| Modify alert schema with semantic fields | Phase 7+ |
| Add `SemanticConfig` fields to `src/api/settings.py` | Phase 7+ |
| Hugging Face model hub integration | Phase 7+ |
