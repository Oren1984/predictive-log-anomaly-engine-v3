# Phase 1 — Verification and Disambiguation Report

**Date:** 2026-03-30
**Branch:** `main`
**Commit:** `d975042`
**Purpose:** Classify ambiguous or potentially duplicated areas before any deletion, archive, or refactor.

---

## 1. `src/modeling/behavior_model.py` vs `src/modeling/behavior/lstm_model.py`

### Classification
| File | Classification |
|------|---------------|
| `src/modeling/behavior_model.py` | **active canonical** |
| `src/modeling/behavior/lstm_model.py` | **wrapper** |

### Evidence

`src/modeling/behavior/lstm_model.py` is a 13-line file that contains only:
```python
from ..behavior_model import BehaviorModelConfig, SystemBehaviorModel
__all__ = ["BehaviorModelConfig", "SystemBehaviorModel"]
```
Its own header comment states explicitly:
> "Re-exports SystemBehaviorModel and BehaviorModelConfig from their canonical
> location (src/modeling/behavior_model.py) under the v2 module path so
> training scripts and the v2 pipeline import from a stable, plan-aligned
> location without duplicating code."

All model logic — LSTM construction, `forward()`, `save()`, `load()` — lives exclusively in `behavior_model.py`. The sub-package file introduces no new code.

`test_proactive_engine.py` imports directly from `src.modeling.behavior_model`, confirming that is the active import path for tests.

### What should happen next
The canonical file (`behavior_model.py`) is the one to keep and maintain. The wrapper (`behavior/lstm_model.py`) exists so that any future V2 scripts can import from `src.modeling.behavior.lstm_model` without the canonical file moving. This is a valid indirection pattern — no action is required now; it can be collapsed in a later cleanup phase once the definitive import path is decided for V3.

---

## 2. `src/modeling/anomaly_detector.py` vs `src/modeling/anomaly/autoencoder.py`

### Classification
| File | Classification |
|------|---------------|
| `src/modeling/anomaly_detector.py` | **active canonical** |
| `src/modeling/anomaly/autoencoder.py` | **wrapper** |

### Evidence

`src/modeling/anomaly/autoencoder.py` is a 12-line file that contains only:
```python
from ..anomaly_detector import AEOutput, AnomalyDetector, AnomalyDetectorConfig
__all__ = ["AnomalyDetectorConfig", "AnomalyDetector", "AEOutput"]
```
Its header comment states:
> "Re-exports AnomalyDetector, AnomalyDetectorConfig, and AEOutput from their
> canonical location (src/modeling/anomaly_detector.py) under the v2 module
> path … without duplicating code."

All denoising autoencoder logic — encoder/decoder construction, `forward()`, `score()`, `fit_threshold()`, `save()`, `load()` — lives exclusively in `anomaly_detector.py`.

`test_proactive_engine.py` imports directly from `src.modeling.anomaly_detector`, confirming the canonical import path.

### What should happen next
Same pattern as item 1: canonical is the file to maintain. The wrapper is a stable re-export alias. No action required yet; can be collapsed in a later cleanup phase.

---

## 3. `src/modeling/severity_classifier.py` vs `src/modeling/severity/severity_classifier.py`

### Classification
| File | Classification |
|------|---------------|
| `src/modeling/severity_classifier.py` | **active canonical** |
| `src/modeling/severity/severity_classifier.py` | **wrapper** |

### Evidence

`src/modeling/severity/severity_classifier.py` is a 24-line file that contains only:
```python
from ..severity_classifier import (
    SEVERITY_LABELS, SeverityClassifier, SeverityClassifierConfig, SeverityOutput,
)
__all__ = ["SEVERITY_LABELS", "SeverityClassifier", "SeverityClassifierConfig", "SeverityOutput"]
```
Its header comment states:
> "Re-exports … from their canonical location (src/modeling/severity_classifier.py)
> under the v2 module path … without duplicating code."

All MLP logic, `predict()`, `predict_batch()`, `build_input()`, `save()`, `load()` — lives exclusively in `severity_classifier.py`.

`test_proactive_engine.py` imports directly from `src.modeling.severity_classifier`.

### What should happen next
Same pattern as items 1 and 2. The three modeling sub-packages (`behavior/`, `anomaly/`, `severity/`) all follow the identical wrapper-over-canonical pattern. A single future task can collapse all three consistently.

---

## 4. `src/data/` vs `src/synthetic/`

### Classification
| Package | Classification |
|---------|---------------|
| `src/synthetic/` | **active canonical** |
| `src/data/` | **wrapper** |

### Evidence

`src/synthetic/` contains the real implementations:
- `generator.py` — 169-line `SyntheticLogGenerator` implementation
- `patterns.py` — concrete `FailurePattern` subclasses
- `scenario_builder.py` — `ScenarioBuilder` implementation

`src/data/` contains three wrapper files that re-export directly from `src/synthetic/`:

| `src/data/` file | Contents |
|-----------------|---------|
| `synth_generator.py` | `from src.synthetic.generator import SyntheticLogGenerator` |
| `synth_patterns.py` | `from src.synthetic.patterns import (FailurePattern, ...)` |
| `scenario_builder.py` | `from src.synthetic.scenario_builder import ScenarioBuilder` |

The fourth file in `src/data/` — `log_event.py` — is **not** a wrapper. It is the canonical `LogEvent` dataclass definition, which is not present in `src/synthetic/`. This makes `src/data/` a mixed package: three wrappers plus one canonical class.

**Import usage confirmed:**
- Active tests (`test_pipeline_smoke.py`, `test_synth_generation.py`) import from `src.synthetic` directly.
- No production `src/` module imports from `src.data`'s wrapper files (only `src/parsing/parsers.py` imports `LogEvent` from `src.data_layer.models`, a different package).

### What should happen next
`src/synthetic/` is the canonical home for synthetic data generation. `src/data/synth_generator.py`, `src/data/synth_patterns.py`, and `src/data/scenario_builder.py` are wrapper aliases with no callers in `src/` outside their own package — they are safe to remove in a future cleanup, after verifying that no external callers rely on the `src.data` import path.

`src/data/log_event.py` is a canonical class used internally by `src/data/` itself (via `src/data/__init__.py`) and should be retained independently of the wrapper cleanup.

---

## 5. `static-demo/`

### Classification
**legacy** (non-code presentation artifact)

### Evidence

From `static-demo/README.md`:
> "This folder contains a lightweight static showcase website built for the Predictive Log Anomaly Engine V2 project."
> "The static demo was created as a presentation and demonstration layer for the existing system."
> "It does not replace the real project, model pipeline, notebook, or GPU-based execution flow."

Contents:
```
static-demo/
├── index.html
├── css/styles.css
├── js/main.js
├── assets/ASSETS_NEEDED.txt     ← placeholder, not actual assets
└── README.md
```

The `assets/ASSETS_NEEDED.txt` file indicates the demo was never fully populated with real screenshots. The site references V2; the current repo is V3. No Python files. No imports. Not referenced by any test, script, or CI configuration.

### What should happen next
`static-demo/` is a V2 presentation artifact with no production connection. It can be archived or deleted in a later cleanup phase. It carries no risk of breaking tests or runtime behavior.

---

## 6. `ai_workspace/`

### Classification
**legacy** (experiment / pipeline-execution workspace, now superseded by `src/`)

### Evidence

From `ai_workspace/README.md`:
> "This directory contains the core implementation of the Predictive Log Anomaly Engine."
> "Relation to the Overall Project: This directory represents the **core AI engine** of the project."

However, the workspace is organized as numbered execution stages (`stage_21` through `stage_26`), each containing standalone run scripts (`run_*.py`) that:
- Import from standard libraries, pandas, sklearn — **not from `src/`**
- Produce outputs to `ai_workspace/logs/`, `ai_workspace/reports/`, and `data/`
- Contain `report.md` files documenting completed experiment results

The `ai_workspace/logs/` directory contains historical execution logs (e.g., `stage_21_sampling.log`, `stage_26_hdfs_supervised_v2.log`), and `ai_workspace/stage_25_evaluation/` contains already-generated PNG evaluation plots. No stage script appears to be wired into the current CI pipeline or any `src/` import.

This directory was the **predecessor experimental environment** from which `src/` was derived. It is now frozen history, not active code.

### What should happen next
`ai_workspace/` should be archived (moved to `archive/` or documented and removed). It is not referenced by any test or runtime module. Its reports and evaluation plots may be worth preserving as documentation. It carries no risk of breaking the test suite or runtime if removed.

---

## 7. `tests/unit/test_explain_decode.py`

### Classification
**active canonical**

### Evidence

This test file covers the `token_id → template string` decoding logic used by `InferenceEngine.explain()`. It:
- Tests `artifacts/vocab.json` and `artifacts/templates.json` structure
- Tests the `token_id = template_id + 2` offset convention
- Tests `InferenceEngine.explain()` end-to-end (conditional on artifact presence)
- Imports from `src.runtime.inference_engine` and `src.sequencing.models` — both active production modules

The tests use `pytest.mark.skipif` guards that gracefully skip when artifacts are absent, so they run cleanly in CI. Three of the `TestEngineExplainDecode` tests ran in the Phase 0 baseline run (included in the 578 passing).

This file tests real, active production behavior. It is not a legacy or orphan test.

### What should happen next
No action required. Keep as-is. Should continue to be part of the standard test run.

---

## 8. `tests/unit/test_proactive_engine.py`

### Classification
**legacy** (tests a legacy module that is not on the production path)

### Evidence

This test file imports directly from `src.engine.proactive_engine`:
```python
from src.engine.proactive_engine import (
    EngineResult,
    ProactiveMonitorEngine,
    ...
)
```

As established in Phase 0:
- `ProactiveMonitorEngine` is not wired into the production API path (`src/runtime/inference_engine.py` / `src/api/pipeline.py`)
- `src/engine/__init__.py` itself documents: *"This module is retained for test coverage and as a reference architecture"*
- The only caller of `src.engine` outside `src/engine/` itself is this test file

The test coverage is thorough (47 test cases covering `EngineResult`, `_EmbeddingBuffer`, construction, `initialize_models`, `process_log`, `process_batch`, `score_sequence`, `generate_alert`, `process_event`, `recent_alerts`, `metrics_snapshot`, and LRU eviction). It passes cleanly in CI. However, it exists solely to maintain coverage over a component that is intentionally disconnected from production.

### What should happen next
This test file is **paired with** `src/engine/proactive_engine.py`. Any decision about that module (archive, delete, migrate) must be made jointly with this test. Do not delete the test without first deciding the fate of `src/engine/`. For now, keep as-is. In a future cleanup phase, both should move together to an `archive/` structure or be removed as a unit.

---

## Summary Table

| Item | Classification | Recommended Action |
|------|---------------|--------------------|
| `src/modeling/behavior_model.py` | active canonical | keep; maintain here |
| `src/modeling/behavior/lstm_model.py` | wrapper | keep for now; collapse later when V3 import paths are finalized |
| `src/modeling/anomaly_detector.py` | active canonical | keep; maintain here |
| `src/modeling/anomaly/autoencoder.py` | wrapper | keep for now; collapse later |
| `src/modeling/severity_classifier.py` | active canonical | keep; maintain here |
| `src/modeling/severity/severity_classifier.py` | wrapper | keep for now; collapse later |
| `src/synthetic/` | active canonical | keep; this is the real implementation |
| `src/data/` (synth_generator, synth_patterns, scenario_builder) | wrapper | remove wrappers in cleanup phase; verify no external callers first |
| `src/data/log_event.py` | active canonical | keep independently |
| `static-demo/` | legacy | archive or delete in future cleanup phase |
| `ai_workspace/` | legacy | archive in future cleanup phase; preserve reports/plots as documentation |
| `tests/unit/test_explain_decode.py` | active canonical | keep; covers production InferenceEngine.explain() |
| `tests/unit/test_proactive_engine.py` | legacy | keep paired with `src/engine/`; decide fate jointly in a future cleanup phase |
