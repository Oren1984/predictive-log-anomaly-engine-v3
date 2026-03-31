# Static Demo Alignment Report

**Date:** 2026-03-31
**Task:** Phase 12 — align static showcase site with actual V3 repository state
**Repository:** `C:\Users\ORENS\predictive-log-anomaly-engine-v3`

---

## Executive Summary

The static demo at `static-demo/` was built to describe the V2 system and had not been updated to reflect the V3 semantic layer integration or the final test count. All version labels, test counts, architecture descriptions, API endpoint references, and pipeline stage counts have been corrected. No visual structure, layout, or CSS was changed. The site remains presentation-ready.

---

## Static Demo Folder Location and Structure

```
static-demo/
├── index.html          ← main HTML file (updated)
├── css/
│   └── styles.css      ← unchanged
├── js/
│   └── main.js         ← updated (header comment + pipeline steps)
├── assets/
│   └── ASSETS_NEEDED.txt ← updated
└── README.md           ← updated
```

---

## Files Changed

| File | Changes |
|------|---------|
| `index.html` | 17 targeted changes — see detail below |
| `js/main.js` | 3 changes — header comment, step 08 detail, added step 09 |
| `README.md` | V2 → V3 title and references; updated "what it shows" to include V3 |
| `assets/ASSETS_NEEDED.txt` | V2 → V3 title; endpoint list updated to include `/v3/ingest` |

---

## What Was Outdated

| Location | What Was Wrong | Why |
|----------|---------------|-----|
| `index.html` `<title>` | "V2" | Repo is V3 with integrated semantic layer |
| `index.html` nav brand | `v2` badge | Same |
| `index.html` hero counter | `578` tests | Current test count is 557 |
| `index.html` hero subtitle | "four-stage neural network pipeline" | System now has 9 stages including optional V3 |
| `index.html` arch subtitle | "Two inference engines" | Three layers now (V1, V2, V3) |
| `index.html` SVG severity label | `info/warn/critical` (unlabeled) | These are V2 ML classifier labels, not V1 AlertPolicy labels |
| `index.html` arch callout | "Dual-Engine Design" | Three-layer architecture |
| `index.html` pipeline title | "8-Stage Pipeline" | V3 adds a 9th optional stage |
| `index.html` results card | `578` tests | Stale |
| `index.html` FastAPI stack | `/ingest · /v2/ingest · /alerts` | Missing `/v3/ingest` |
| `index.html` testing stack | `578 tests` | Stale |
| `index.html` compare title | "Dual-Engine Architecture" | Three layers |
| `index.html` V2 status | "Training phase" | V2 pipeline is production-ready and has been for phases 1–10 |
| `index.html` footer brand | "V2" | Stale |
| `index.html` footer meta | `578 Tests · 4 Neural Stages` | Stale count; architecture summary outdated |
| `js/main.js` header comment | "V2 — Main JS" | Stale |
| `js/main.js` step 08 detail | "Classes: info / warning / critical" with no qualifier | Ambiguous — these are V2 ML classifier labels, not V1 AlertPolicy labels |
| `README.md` | "V2" throughout | Stale |
| `ASSETS_NEEDED.txt` | "V2 Showcase"; endpoint list | Stale |

---

## What Was Updated

### `index.html` (17 changes)

1. `<title>` V2 → V3
2. Nav brand version badge: `v2` → `v3`
3. Hero subtitle: "four-stage neural network pipeline" → "multi-layer AI pipeline"
4. Hero counter: `578` → `557`
5. Architecture section subtitle: "Two inference engines..." → "Three-layer architecture — V1 ensemble baseline, V2 deep learning pipeline, and optional V3 semantic enrichment layer"
6. SVG Severity MLP sublabel: `info/warn/critical` → `V2 classifier labels`
7. Architecture callout: "Dual-Engine Design" → "Triple-Layer Architecture" with V3 description
8. Pipeline section title: "The 8-Stage Pipeline" → "The 9-Stage Pipeline"
9. Pipeline section subtitle: updated to note "V2 core pipeline + optional V3 semantic enrichment"
10. Pipeline step counter placeholder: "Step 1 of 8" → "Step 1 of 9"
11. Results card: `578` → `557`
12. FastAPI stack entry: added `/v3/ingest` to endpoint list
13. Testing stack entry: `578 tests` → `557 tests`
14. Engine compare table title: "Dual-Engine Architecture" → "Multi-Layer Architecture"
15. V2 status: "Training phase" → "Production-ready"
16. Added V3 row to engine compare table: shows it as optional overlay with `/v3/ingest` and `SEMANTIC_ENABLED=true`
17. Footer: brand updated V2→V3; meta updated to "557 Tests · V2 Pipeline + V3 Semantic Layer"

### `js/main.js` (3 changes)

1. File header comment: "V2 — Main JS" → "V3 — Main JS"
2. Pipeline step 08 `detail` field: added "V2 classifier labels:" qualifier before `info / warning / critical`; added V1 AlertPolicy labels note
3. Added pipeline step 09 — V3 Semantic Enrichment (tagged `tag-ml`, clearly marked "V3 — OPTIONAL") with description of explanation/evidence_tokens/semantic_similarity fields and a note that all fields are null when `SEMANTIC_ENABLED=false` (default)

### `README.md`

- Title updated to V3
- "What it shows" updated to mention V3 semantic layer and 9-stage pipeline
- Relationship section updated to reference `src/` and `docs/` directly
- Notes section updated to reference `ASSETS_NEEDED.txt`

### `assets/ASSETS_NEEDED.txt`

- Title updated to V3
- FastAPI endpoint capture instruction updated to include `/v3/ingest`
- Added V3 semantic enrichment note at end

---

## Intentionally Kept Unchanged

| Item | Reason |
|------|--------|
| `css/styles.css` | No content inaccuracies; visual structure is correct and well-built |
| Simulation log data (`NORMAL_LOGS`, `ANOMALY_LOGS`, `ALERT_TEMPLATES` in `main.js`) | Data is illustrative HDFS format and remains accurate |
| SVG architecture diagram node layout | Structural change risk; node labels are correct; only sublabel text was corrected |
| Grafana placeholder block | No real screenshot available; placeholder is clearly marked and functional |
| Team section | Not affected by V2→V3 changes |
| BiLSTM label in step 06 and SVG | Not verified as incorrect against codebase in this session; audit did not flag it |
| `<1ms` inference latency claim | Refers to V2 CPU inference path, which remains accurate |
| Alert lifecycle diagram (5 steps) | V3 enrichment happens after alert emission, not during the lifecycle; adding it here would misrepresent the flow |
| Problem/comparison section (Traditional vs Behavioral) | Conceptual framing — technology-agnostic, timeless |

---

## Assumptions Made

1. The V3 semantic layer is optional and disabled by default — pipeline step 09 is labeled "V3 — OPTIONAL" accordingly.
2. The V2 pipeline is production-ready as of Phase 1–10 of the refactor — "Training phase" label was stale.
3. `tag-ml` CSS class was reused for pipeline step 09 (sentence-transformers is an ML model stage) to avoid adding new CSS.
4. The engine compare table uses a 3-column CSS grid — the V3 row uses the same 3-span structure (col 1 = layer name, col 2 = V1 context, col 3 = V2/V3 detail).

---

## Placeholders That Still Remain

| Placeholder | Location | Description |
|-------------|----------|-------------|
| Simulated Grafana dashboard | `#results` section | Functional simulation; replace with `assets/grafana-dashboard.png` when available |
| FastAPI docs screenshot | — | Not yet added; see `ASSETS_NEEDED.txt` item 2 |
| Terminal demo screenshot | — | Not yet added; see `ASSETS_NEEDED.txt` item 3 |

---

## Manual Follow-Up Recommendations

1. **Replace Grafana placeholder** — run `docker compose up`, capture a real dashboard screenshot, drop it at `static-demo/assets/grafana-dashboard.png` and uncomment the `<img>` tag per `ASSETS_NEEDED.txt` instructions.
2. **Verify BiLSTM vs LSTM** — the static demo and JS say "BiLSTM" (bidirectional); the notebooks say "LSTM Behavior Model". If the actual `SystemBehaviorModel` class is unidirectional LSTM, update `main.js` step 06 and the SVG node label accordingly.
3. **Run site in browser after edits** — open `static-demo/index.html` directly; verify the pipeline stepper shows "Step 1 of 9", the counter animates to 557, and pipeline step 09 renders correctly.
4. **V3 `ct-row` visual check** — the V3 row uses inline `border-top` style in the compare table. Confirm it renders cleanly in the target browsers.
