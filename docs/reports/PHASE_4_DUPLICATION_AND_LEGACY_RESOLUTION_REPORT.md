# Phase 4 — Duplication and Legacy Resolution Report

**Date:** 2026-03-30
**Branch:** `main`
**Commit baseline:** `d975042`
**Purpose:** Resolve verified duplication and isolate verified legacy components, based strictly on Phase 1 evidence.

---

## Decision Framework

Every action in this phase required a prior proof from Phase 1:
- **No guessing** — only items classified in Phase 1 as *wrapper*, *duplicate*, or *legacy* were touched.
- Each sub-step was followed by a test run before proceeding.

---

## 1. Modeling Sub-Package Wrappers Removed

### Context (from Phase 1)
All three modeling sub-packages were classified as **wrappers**: pure re-export adapters with no logic of their own, pointing back to canonical top-level files.

### Evidence that removal was safe
A codebase-wide search for imports from the wrapper paths returned **zero matches**:
```
grep "from src.modeling.behavior"   → 0 results
grep "from src.modeling.anomaly"    → 0 results
grep "from src.modeling.severity"   → 0 results
```
No module in `src/`, `tests/`, or `scripts/` imported from the wrapper paths.

### Files removed

| Removed file | Was a wrapper for |
|-------------|------------------|
| `src/modeling/behavior/__init__.py` | `src/modeling/behavior_model.py` |
| `src/modeling/behavior/lstm_model.py` | `src/modeling/behavior_model.py` |
| `src/modeling/anomaly/__init__.py` | `src/modeling/anomaly_detector.py` |
| `src/modeling/anomaly/autoencoder.py` | `src/modeling/anomaly_detector.py` |
| `src/modeling/severity/__init__.py` | `src/modeling/severity_classifier.py` |
| `src/modeling/severity/severity_classifier.py` | `src/modeling/severity_classifier.py` |

### Canonical files retained (unchanged)

| File | Classification |
|------|---------------|
| `src/modeling/behavior_model.py` | active canonical — kept |
| `src/modeling/anomaly_detector.py` | active canonical — kept |
| `src/modeling/severity_classifier.py` | active canonical — kept |

### Imports changed
None. No callers of the wrapper paths existed.

### Test result after this sub-step
```
578 passed, 26 deselected   ✓  (no change from baseline)
```

---

## 2. `src/data/` Wrapper Files Removed

### Context (from Phase 1)
Three files in `src/data/` were classified as **wrappers**: they contained only `from src.synthetic.X import Y` re-exports with no original logic.
`src/data/log_event.py` was classified **active canonical** and was not touched.

### Evidence that removal was safe
A codebase-wide search for callers of the three wrapper files returned **zero active matches**:
```
grep "from src.data import SyntheticLogGenerator"  → 0 results (only comments inside src/data/ itself)
grep "from src.data import ScenarioBuilder"         → 0 results
grep "from src.data import FailurePattern"          → 0 results
```
The only external import from `src.data` was:
```python
# tests/unit/test_synth_generation.py:48
from src.data.log_event import LogEvent
```
This imports `LogEvent` — the canonical class — which was retained.

### Files removed

| Removed file | Was a wrapper for |
|-------------|------------------|
| `src/data/synth_generator.py` | `src/synthetic/generator.SyntheticLogGenerator` |
| `src/data/synth_patterns.py` | `src/synthetic/patterns.*Pattern` |
| `src/data/scenario_builder.py` | `src/synthetic/scenario_builder.ScenarioBuilder` |

### `src/data/__init__.py` updated

Removed the three dead re-export imports; retained only `LogEvent`:

```diff
-"""src.data — Data models and synthetic generation for the pipeline."""
+"""src.data — Core data models for the pipeline."""
 from .log_event import LogEvent
-from .scenario_builder import ScenarioBuilder
-from .synth_generator import SyntheticLogGenerator
-from .synth_patterns import (
-    AuthBruteForcePattern,
-    DiskFullPattern,
-    FailurePattern,
-    MemoryLeakPattern,
-    NetworkFlapPattern,
-)

 __all__ = [
     "LogEvent",
-    "FailurePattern",
-    "MemoryLeakPattern",
-    "DiskFullPattern",
-    "AuthBruteForcePattern",
-    "NetworkFlapPattern",
-    "SyntheticLogGenerator",
-    "ScenarioBuilder",
 ]
```

### Canonical files retained (unchanged)

| File | Status |
|------|--------|
| `src/data/log_event.py` | active canonical — kept |
| `src/synthetic/` (all files) | active canonical — kept |

### Test result after this sub-step
```
578 passed, 26 deselected   ✓  (no change from prior sub-step)
```

---

## 3. `src/engine/` Archived

### Context (from Phase 1)
`src/engine/proactive_engine.py` and `src/engine/__init__.py` were classified **legacy**: `ProactiveMonitorEngine` is intentionally disconnected from the production API path and was explicitly flagged in its own header comment as "retained for test coverage and reference architecture."

### Evidence that archival was safe
The only caller outside `src/engine/` itself:
```
tests/unit/test_proactive_engine.py:59   from src.engine.proactive_engine import (...)
tests/unit/test_proactive_engine.py:734  from src.engine import EngineResult, ProactiveMonitorEngine
```
No production module (`src/api/`, `src/runtime/`, `src/alerts/`, etc.) imported from `src/engine/`.

### Files moved to archive

| Source | Destination |
|--------|-------------|
| `src/engine/proactive_engine.py` | `archive/src/engine/proactive_engine.py` |
| `src/engine/__init__.py` | `archive/src/engine/__init__.py` |

`src/engine/` directory was removed from the live source tree.

### Imports changed
None in production code. The archived test file (`test_proactive_engine.py`) was the sole importer.

---

## 4. Tests Handled

### `tests/unit/test_proactive_engine.py` — **archived**

**Justification:** This test was the sole caller of `src.engine`. With `src/engine/` archived, the test would fail with `ModuleNotFoundError`. Since the test exclusively covers the archived legacy engine — not any production path — it was moved alongside the engine code it tests.

| Source | Destination |
|--------|-------------|
| `tests/unit/test_proactive_engine.py` | `archive/tests/unit/test_proactive_engine.py` |

**Tests removed from suite:** 65 (the proactive engine test file contained 65 test methods, more than the 47 listed in its header comment).

### `tests/unit/test_explain_decode.py` — **kept, unchanged**

**Justification:** This test covers `InferenceEngine.explain()` decode logic — a live production method. It was classified **active canonical** in Phase 1. No action taken.

---

## 5. Final Test Results

```
pytest -m "not slow"

513 passed, 26 deselected in 19.74s   ✓
```

| Metric | Phase 0 baseline | After Phase 4 |
|--------|-----------------|---------------|
| Tests passing | 578 | 513 |
| Tests deselected (slow) | 26 | 26 |
| Tests failing | 0 | **0** |
| Delta | — | −65 (archived proactive engine tests) |

The 65-test reduction is fully accounted for by `test_proactive_engine.py` being moved to `archive/`. Zero test failures introduced.

---

## 6. Summary of All Changes

### Files removed from live source tree

| File | Reason |
|------|--------|
| `src/modeling/behavior/__init__.py` | wrapper, zero callers |
| `src/modeling/behavior/lstm_model.py` | wrapper, zero callers |
| `src/modeling/anomaly/__init__.py` | wrapper, zero callers |
| `src/modeling/anomaly/autoencoder.py` | wrapper, zero callers |
| `src/modeling/severity/__init__.py` | wrapper, zero callers |
| `src/modeling/severity/severity_classifier.py` | wrapper, zero callers |
| `src/data/synth_generator.py` | wrapper, zero callers |
| `src/data/synth_patterns.py` | wrapper, zero callers |
| `src/data/scenario_builder.py` | wrapper, zero callers |
| `src/engine/proactive_engine.py` | legacy, archived |
| `src/engine/__init__.py` | legacy, archived |

### Files modified

| File | Change |
|------|--------|
| `src/data/__init__.py` | Removed three dead re-export imports; retained `LogEvent` only |

### Files archived

| File | Archive destination |
|------|-------------------|
| `src/engine/proactive_engine.py` | `archive/src/engine/proactive_engine.py` |
| `src/engine/__init__.py` | `archive/src/engine/__init__.py` |
| `tests/unit/test_proactive_engine.py` | `archive/tests/unit/test_proactive_engine.py` |

### Not touched (confirmed active canonical)

- `src/modeling/behavior_model.py`
- `src/modeling/anomaly_detector.py`
- `src/modeling/severity_classifier.py`
- `src/data/log_event.py`
- `src/synthetic/` (all files)
- `tests/unit/test_explain_decode.py`
- All production API, runtime, and alert modules
