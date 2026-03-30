# Phase 5 — Repository Hygiene Report

**Date:** 2026-03-30
**Branch:** `main`
**Commit baseline:** `d975042`
**Purpose:** Finalize `.gitignore` coverage and remove confirmed legacy workspace before V3 feature work begins.

---

## 1. `.gitignore` Audit Results

All checks performed via `git check-ignore -v` against real files on disk.

### 1.1 Model files

| Pattern checked | Covered by | Line | Result |
|-----------------|-----------|------|--------|
| `models/*.pkl` (`baseline.pkl`) | `models/` | 19 | ✓ ignored |
| `models/*.pt` (`transformer.pt`) | `models/` | 19 | ✓ ignored |
| `models/behavior/behavior_model.pt` | `models/` | 19 | ✓ ignored |
| `models/anomaly/anomaly_detector.pt` | `models/` | 19 | ✓ ignored |
| `models/severity/severity_classifier.pt` | `models/` | 19 | ✓ ignored |
| `models/embeddings/word2vec.model` | `models/` | 19 | ✓ ignored |

**Assessment:** The blanket `models/` entry on line 19 covers the entire directory including all `.pkl`, `.pt`, and `.model` extensions at any nesting depth. No gaps.

---

### 1.2 Data paths

| Path | Covered by | Line | Result |
|------|-----------|------|--------|
| `data/raw/` | `data/raw/` | 13 | ✓ ignored |
| `data/raw/BGL.log` | `data/raw/` | 13 | ✓ ignored |
| `data/raw/HDFS_1`, `HDFS_2`, `HDFS.npz` | `data/raw/` | 13 | ✓ ignored |
| `data/intermediate/` | `data/intermediate/` | 14 | ✓ ignored |
| `data/processed/*.parquet` | `data/processed/*.parquet` | 16 | ✓ ignored |
| `data/processed/*.csv` | `data/processed/*.csv` | 15 | ✓ ignored |
| `data/synth/` | `data/synth/` | 37 | ✓ ignored |
| `data/processed/schema.md` | *(not ignored)* | — | intentional — doc file |

**Assessment:** All large binary and generated data paths are excluded. The only unignored file in `data/processed/` is `schema.md` — a text documentation file that is appropriate to track.

---

### 1.3 Generated report outputs

| Path | Covered by | Line | Result |
|------|-----------|------|--------|
| `reports/evaluation_report_v2.json` | `reports/*.json` | 23 | ✓ ignored |
| `reports/metrics.json` | `reports/*.json` | 23 | ✓ ignored |
| `reports/metrics_transformer.json` | `reports/*.json` | 23 | ✓ ignored |
| `reports/runtime_calibration_scores.csv` | `reports/*.csv` | 22 | ✓ ignored |
| `reports/runtime_demo_results.csv` | `reports/*.csv` | 22 | ✓ ignored |
| `reports/runtime_demo_evidence.jsonl` | `reports/*.jsonl` | 24 | ✓ ignored |

**Assessment:** All generated report file types (`.json`, `.jsonl`, `.csv`) are excluded. Markdown reports (`.md`) are intentionally tracked. No gaps.

---

### 1.4 Generated alert JSON files

| Path | Covered by | Line | Result |
|------|-----------|------|--------|
| `artifacts/vocab.json` | `artifacts/*.json` | 25 | ✓ ignored |
| `artifacts/templates.json` | `artifacts/*.json` | 25 | ✓ ignored |
| `artifacts/threshold.json` | `artifacts/*.json` | 25 | ✓ ignored |
| `artifacts/threshold_runtime.json` | `artifacts/*.json` | 25 | ✓ ignored |
| `artifacts/threshold_transformer.json` | `artifacts/*.json` | 25 | ✓ ignored |
| `artifacts/n8n_outbox/*.json` | `artifacts/n8n_outbox/*.json` | 28 | ✓ ignored |
| `artifacts/n8n_outbox/.gitkeep` | *(not ignored)* | — | ✓ correct — .gitkeep must be tracked |
| `artifacts/README.md` | *(not ignored)* | — | ✓ correct — doc file |

**Assessment:** All generated JSON alert files are excluded. The `.gitkeep` directory marker is correctly trackable (the `*.json` pattern does not match it). No gaps.

---

## 2. `ai_workspace/` — Confirmed Legacy, Removed

### Decision
Phase 1 classified `ai_workspace/` as **legacy**: it was the predecessor experiment environment (stages 21–26) from which `src/` was derived. It contained standalone run scripts that do not import from `src/`, historical execution logs, and evaluation plots. It was never wired into CI, tests, or the runtime.

Its git status was `??` (untracked) — it had never been committed.

### Actions taken

**1. `.gitignore` updated:**

```diff
-# Logs
-ai_workspace/logs/
+# AI workspace (legacy experiment directory — not part of the live source tree)
+ai_workspace/
```

The previous entry `ai_workspace/logs/` was narrower than needed. The new entry ignores the entire directory, ensuring it can never be accidentally committed.

**2. Directory removed:**

`ai_workspace/` was deleted from the working tree. Since it was untracked, this had no effect on git history. Nothing in `src/`, `tests/`, or the Docker build depended on it.

---

## 3. Final `.gitignore` State

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual env
.venv/
.env

# Data (ALL heavy outputs)
data/raw/
data/intermediate/
data/processed/*.csv
data/processed/*.parquet

# Models
models/

# Reports (keep only markdown)
reports/*.csv
reports/*.json
reports/*.jsonl
artifacts/*.json

# N8N outbox (test artifacts — accumulate on every dry-run)
artifacts/n8n_outbox/*.json

# AI workspace (legacy experiment directory — not part of the live source tree)
ai_workspace/

# Claude temp cwd markers
tmpclaude-*-cwd

# Synthetic data
data/synth/

# Windows junk
nul
reports/*.tmp
```

---

## 4. Summary of Changes

| Item | Action | Result |
|------|--------|--------|
| `models/*.pkl`, `*.pt`, `**/*.model` | Verified — covered by `models/` (line 19) | No change needed |
| `data/raw/`, `data/processed/`, `data/intermediate/` | Verified — all covered | No change needed |
| `reports/*.json/csv/jsonl` | Verified — all covered | No change needed |
| `artifacts/*.json`, `artifacts/n8n_outbox/*.json` | Verified — all covered | No change needed |
| `ai_workspace/logs/` → `ai_workspace/` | Updated `.gitignore` | Broadened coverage |
| `ai_workspace/` directory | Removed (untracked legacy workspace) | Directory deleted |

**Files changed: 1** (`.gitignore`)
**Directories removed: 1** (`ai_workspace/`)
**No source code modified. No imports changed. No tests affected.**

---

## 5. Repository Hygiene Status

| Area | Status |
|------|--------|
| Model binaries | ✓ fully excluded |
| Large data files | ✓ fully excluded |
| Generated reports | ✓ fully excluded (markdown kept) |
| Generated artifacts / alerts | ✓ fully excluded (README + .gitkeep kept) |
| Legacy workspace | ✓ excluded and removed |
| V3 feature code | Not started — hygiene phase complete |
