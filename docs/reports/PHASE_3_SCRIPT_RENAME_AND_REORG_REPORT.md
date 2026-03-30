# Phase 3 — Script Rename and Reorganisation Report

**Date:** 2026-03-30
**Branch:** `main`
**Commit baseline:** `d975042`
**Purpose:** Clean up script naming and layout without changing runtime architecture.

---

## 1. `stage_07_run_api.py` → `run_api.py`

### File rename

| | Path |
|-|------|
| Before | `scripts/stage_07_run_api.py` |
| After | `scripts/run_api.py` |

### References updated

| File | Change |
|------|--------|
| `scripts/run_api.py` | Updated internal header comment (`# scripts/stage_07_run_api.py` → `# scripts/run_api.py`) and docstring usage examples |
| `scripts/run_api.py` | `ArgumentParser(description="Stage 07 — Run API server")` → `ArgumentParser(description="Run API server")` |
| `main.py` | Comment `scripts/stage_07_run_api.py` → `scripts/run_api.py` |
| `main.py` | Import `from scripts.stage_07_run_api import main` → `from scripts.run_api import main` |

### Docker assessment

The Dockerfile CMD (`src.api.app:create_app` via uvicorn) does not reference `stage_07_run_api.py` — **no Dockerfile change needed**.

---

## 2. `scripts/data_pipeline/` — created and populated

### Files moved into `scripts/data_pipeline/`

| Before | After |
|--------|-------|
| `scripts/stage_01_data.py` | `scripts/data_pipeline/stage_01_data.py` |
| `scripts/stage_01_synth_generate.py` | `scripts/data_pipeline/stage_01_synth_generate.py` |
| `scripts/stage_01_synth_to_processed.py` | `scripts/data_pipeline/stage_01_synth_to_processed.py` |
| `scripts/stage_01_synth_validate.py` | `scripts/data_pipeline/stage_01_synth_validate.py` |
| `scripts/stage_02_templates.py` | `scripts/data_pipeline/stage_02_templates.py` |
| `scripts/stage_03_sequences.py` | `scripts/data_pipeline/stage_03_sequences.py` |
| `scripts/stage_04_baseline.py` | `scripts/data_pipeline/stage_04_baseline.py` |
| `scripts/stage_04_transformer.py` | `scripts/data_pipeline/stage_04_transformer.py` |
| `scripts/stage_05_run.py` | `scripts/data_pipeline/stage_05_run.py` |
| `scripts/stage_06_demo_alerts.py` | `scripts/data_pipeline/stage_06_demo_alerts.py` |

### String reference updated in `src/`

`src/runtime/pipeline_v2.py:317` contained a user-facing error message with the old path:

```diff
- "Required for v2 inference: run scripts/stage_02_templates.py "
+ "Required for v2 inference: run scripts/data_pipeline/stage_02_templates.py "
```

No import statement was changed — this was a string literal in an error message.

---

## 3. `scripts/archive/` → `archive/scripts/`

### Files moved

| Before | After |
|--------|-------|
| `scripts/archive/10_download_data.py` | `archive/scripts/10_download_data.py` |
| `scripts/archive/20_prepare_events.py` | `archive/scripts/20_prepare_events.py` |
| `scripts/archive/30_build_sequences.py` | `archive/scripts/30_build_sequences.py` |
| `scripts/archive/40_train_baseline.py` | `archive/scripts/40_train_baseline.py` |
| `scripts/archive/90_run_api.py` | `archive/scripts/90_run_api.py` |
| `scripts/archive/run_0_4.py` | `archive/scripts/run_0_4.py` |
| `scripts/archive/stage_05_runtime_benchmark.py` | `archive/scripts/stage_05_runtime_benchmark.py` |
| `scripts/archive/stage_05_runtime_calibrate.py` | `archive/scripts/stage_05_runtime_calibrate.py` |
| `scripts/archive/stage_05_runtime_demo.py` | `archive/scripts/stage_05_runtime_demo.py` |
| `scripts/archive/validation/run_memory_validation.py` | `archive/scripts/validation/run_memory_validation.py` |
| `scripts/archive/validation/run_performance_validation.py` | `archive/scripts/validation/run_performance_validation.py` |

`scripts/archive/` was removed after all contents were moved.

### Note on `test_runtime_calibration.py`

`tests/unit/test_runtime_calibration.py` references `scripts/stage_05_runtime_calibrate.py` (line 66). This path never existed in `scripts/` directly — the file was always under `scripts/archive/`. The test is double-guarded by `@pytest.mark.slow` and `@needs_artifacts`, which requires `data/processed/events_tokenized.parquet` + multiple artifact files that are absent in the repo. It was therefore always in the "26 deselected" bucket and is unaffected by this move.

---

## 4. Final `scripts/` Structure

```
scripts/
├── 00_check_env.ps1
├── data_pipeline/
│   ├── stage_01_data.py
│   ├── stage_01_synth_generate.py
│   ├── stage_01_synth_to_processed.py
│   ├── stage_01_synth_validate.py
│   ├── stage_02_templates.py
│   ├── stage_03_sequences.py
│   ├── stage_04_baseline.py
│   ├── stage_04_transformer.py
│   ├── stage_05_run.py
│   └── stage_06_demo_alerts.py
├── demo_run.py
├── evaluate_v2.py
├── run_api.py           ← renamed from stage_07_run_api.py
└── smoke_test.sh
```

---

## 5. Test Results

```
pytest -m "not slow"

578 passed, 26 deselected in 32.78s
```

**Result: PASS — identical to Phase 0 baseline (578/578). No regressions.**

---

## 6. Confirmation: No Runtime Imports Changed

| Scope | Status |
|-------|--------|
| `src/` Python source files | Unchanged (only one string literal updated in `pipeline_v2.py`) |
| `tests/` test files | Unchanged |
| `docker/Dockerfile` | Unchanged |
| `docker/docker-compose.yml` | Unchanged |
| Runtime import path (`from scripts.run_api import main`) | Updated in `main.py` to match renamed file — functionally identical |

No model loading, API routing, inference, or alert logic was modified.
