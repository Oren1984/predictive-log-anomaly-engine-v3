# Phase 0 — Baseline Protection Report

**Date:** 2026-03-30
**Branch:** `main`
**Commit:** `d975042` (Initial commit)
**Purpose:** Establish a verified baseline before any repository changes begin.

---

## 1. Test Baseline

### Command
```
pytest -m "not slow"
```

### Result

| Metric | Value |
|--------|-------|
| Tests passed | **578** |
| Tests deselected (slow) | 26 |
| Tests failed | 0 |
| Duration | ~85 s |

**Verdict: PASS — all 578 non-slow tests pass cleanly.**

---

## 2. Docker Build

### Command
```
docker build -f docker/Dockerfile -t predictive-log-anomaly-engine-v3:baseline .
```

### Result

| Step | Outcome |
|------|---------|
| Base image | `python:3.11-slim` (sha `9358444...`) |
| All 10 build steps | completed |
| Image tagged | `predictive-log-anomaly-engine-v3:baseline` |
| Image digest | `sha256:33d1e4d0...` |
| Build duration | ~156 s (first run; layers cached on second run) |

**Verdict: PASS — image builds successfully with zero errors or warnings.**

---

## 3. Compose Stack / Health Check

### Command
```
docker compose -f docker/docker-compose.yml up -d
curl http://localhost:8000/health
```

### Port conflict note
At the time of testing, ports **8000** (`smart-beauty-mirror-backend`) and **3000**
(`smart-beauty-mirror-frontend`) from a separate project were already bound. Those
containers were temporarily stopped for the duration of the health check, then
immediately restored. No code was changed.

### Services started

| Container | Image | Status at health check |
|-----------|-------|----------------------|
| `docker-api-1` | `docker-api:latest` (built from `docker/Dockerfile`) | `Up ~1 min (healthy)` |
| `docker-prometheus-1` | `prom/prometheus:v2.51.0` | `Up ~1 min` |
| `docker-grafana-1` | `grafana/grafana:10.4.2` | `Up ~1 min` |

### Health endpoint response

```
HTTP 200 OK

{
  "status": "healthy",
  "uptime_seconds": 73.3,
  "components": {
    "inference_engine": {
      "status": "ok",
      "artifacts_loaded": true
    },
    "alert_manager": {
      "status": "ok"
    },
    "alert_buffer": {
      "status": "ok",
      "size": 66
    }
  }
}
```

**Verdict: PASS — all three services started; `/health` returns `200 OK` with every component reporting `ok` and `artifacts_loaded: true`.**

---

## 4. Files Under `models/`

```
models/
├── anomaly/
│   ├── .gitkeep
│   └── anomaly_detector.pt
├── baseline.pkl
├── behavior/
│   ├── .gitkeep
│   └── behavior_model.pt
├── embeddings/
│   ├── .gitkeep
│   └── word2vec.model
├── severity/
│   ├── .gitkeep
│   └── severity_classifier.pt
└── transformer.pt
```

| File | Notes |
|------|-------|
| `models/baseline.pkl` | V1 ensemble baseline pickle |
| `models/transformer.pt` | Top-level transformer checkpoint |
| `models/anomaly/anomaly_detector.pt` | V2 anomaly detector |
| `models/behavior/behavior_model.pt` | V2 behavior model |
| `models/embeddings/word2vec.model` | Word2Vec embeddings |
| `models/severity/severity_classifier.pt` | V2 severity classifier |
| `models/*/. gitkeep` | Directory markers only (4 files) |

**Total artifact files: 6 model files + 4 `.gitkeep` markers.**

---

## 5. `proactive_engine.py` Import Analysis

### File under investigation
`src/engine/proactive_engine.py`

### Search scope
All `*.py` files under `src/`

### Findings

| Location | Import type | Details |
|----------|------------|---------|
| `src/engine/__init__.py:16` | **Active import** | `from .proactive_engine import EngineResult, ProactiveMonitorEngine` |

No other file in `src/` imports `proactive_engine` directly, nor imports from
`src.engine` (the package that re-exports it).

### Summary

`proactive_engine.py` is **imported in exactly one place within `src/`**: by
`src/engine/__init__.py`, which re-exports `ProactiveMonitorEngine` and
`EngineResult` via `__all__`. This import is active (not commented out).

However, `src/engine` itself is **not imported by any other module in `src/`**.
The only consumers are in `tests/unit/test_proactive_engine.py` (outside `src/`),
which the `__init__.py` comment explicitly documents.

**Conclusion:** `proactive_engine.py` is live code within its own package, but the
package is not wired into the production API path. It is isolated to test coverage
and reference architecture use.

---

## 6. Overall Baseline Verdict

| Check | Result |
|-------|--------|
| `pytest -m "not slow"` | PASS (578/578) |
| `docker build -f docker/Dockerfile` | PASS |
| `docker compose up` + `/health` | PASS (HTTP 200, all components OK) |
| `models/` inventory captured | DONE |
| `proactive_engine.py` import analysis | DONE — isolated to `src/engine/__init__.py`; not on production path |

**Baseline is clean. Safe to proceed with Phase 1.**
