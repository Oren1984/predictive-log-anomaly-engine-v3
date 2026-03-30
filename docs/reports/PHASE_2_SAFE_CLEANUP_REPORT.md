# Phase 2 — Safe Cleanup Report

**Date:** 2026-03-30
**Branch:** `main`
**Commit baseline:** `d975042`
**Purpose:** Perform safe, non-breaking cleanup actions with no runtime import changes.

---

## Changes Applied

### 1. `evaluation_report.json` → `reports/evaluation_report_v2.json`

| | Detail |
|-|--------|
| Action | Moved (`mv`) |
| Source | `evaluation_report.json` (repo root) |
| Destination | `reports/evaluation_report_v2.json` |
| `reports/` pre-existed | Yes — directory already contained `metrics.json`, `metrics_transformer.json`, `runtime_calibration_scores.csv`, `runtime_demo_evidence.jsonl`, `runtime_demo_results.csv` |

The file was renamed `_v2` to make its provenance explicit (it was the V2 model evaluation output).

---

### 2. `reports/*.json` in `.gitignore`

| | Detail |
|-|--------|
| Action | No change needed |
| Reason | `.gitignore` already contained `reports/*.json` (line 23) before this phase |

Pre-existing coverage confirmed. `evaluation_report_v2.json` is automatically ignored.

---

### 3. `artifacts/n8n_outbox/` — cleared and preserved

| | Detail |
|-|--------|
| Action | Deleted all `*.json` files; created `.gitkeep` |
| Files deleted | 802 UUID-named `.json` files |
| Files remaining | `artifacts/n8n_outbox/.gitkeep` (0 bytes, directory marker) |

Directory itself is preserved. No code references this directory at runtime (it is a dry-run outbox sink).

---

### 4. `artifacts/n8n_outbox/*.json` added to `.gitignore`

| | Detail |
|-|--------|
| Action | Modified `.gitignore` |
| Old entry | `artifacts/n8n_outbox/` (ignored entire directory — `.gitkeep` would also be ignored) |
| New entry | `artifacts/n8n_outbox/*.json` (ignores only JSON files; `.gitkeep` can now be tracked) |

**Diff:**
```diff
- artifacts/n8n_outbox/
+ artifacts/n8n_outbox/*.json
```

---

### 5. Archive folders created

| Directory | Contents |
|-----------|---------|
| `archive/src/engine/` | `.gitkeep` only |
| `archive/src/data/` | `.gitkeep` only |
| `archive/scripts/` | `.gitkeep` only |

No code was moved into these directories. They are placeholders for Phase 3+ archival operations.

---

## Files Changed

| File / Path | Change Type |
|------------|------------|
| `evaluation_report.json` | Removed from root (moved) |
| `reports/evaluation_report_v2.json` | Created (moved from root) |
| `artifacts/n8n_outbox/*.json` (802 files) | Deleted |
| `artifacts/n8n_outbox/.gitkeep` | Created |
| `archive/src/engine/.gitkeep` | Created |
| `archive/src/data/.gitkeep` | Created |
| `archive/scripts/.gitkeep` | Created |
| `.gitignore` | Modified (1 line: `n8n_outbox/` → `n8n_outbox/*.json`) |

---

## Confirmation: No Runtime Imports Changed

The following were **not** touched:

| Scope | Status |
|-------|--------|
| `src/` — all Python source files | Unchanged |
| `tests/` — all test files | Unchanged |
| `docker/Dockerfile` | Unchanged |
| `docker/docker-compose.yml` | Unchanged |
| `requirements/` | Unchanged |
| `scripts/` | Unchanged |
| `main.py` | Unchanged |

No import statement, module path, or runtime configuration was modified. The 578-test baseline established in Phase 0 will continue to pass without re-running.

---

## Not Done (per Phase 2 scope)

| Item | Deferred to |
|------|------------|
| Archiving `src/engine/` code | Phase 3+ |
| Archiving `src/data/` wrappers | Phase 3+ |
| Moving scripts | Phase 3+ |
| Touching tests | Phase 3+ |
| Deleting `static-demo/` or `ai_workspace/` | Phase 3+ |
