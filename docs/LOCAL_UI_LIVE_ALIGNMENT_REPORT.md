# Local UI Live Alignment Report

**Date:** 2026-03-31
**Task:** Phase 13 — upgrade local runtime UI to match static showcase design language
**Repository:** `C:\Users\ORENS\predictive-log-anomaly-engine-v3`

---

## Executive Summary

The local runtime UI (`templates/index.html`) has been upgraded to match the visual design language of the static showcase site (`static-demo/`). The changes are CSS-only plus three minor HTML element updates. All JavaScript logic, backend endpoint routes, data rendering functions, and HTML structure are completely unchanged. The UI remains fully live-data-driven and connected to the real FastAPI backend.

**Before:** Light theme, system-ui font, blue (`#2563eb`) accent, white cards, light gray borders.
**After:** Dark theme (`#0a0e1a`/`#111827`), Inter + JetBrains Mono fonts, cyan (`#00d4ff`) accent, dark cards with subtle borders — matching the static showcase's design language while remaining the real operational dashboard.

---

## Local UI Files Inspected

| File | Purpose |
|------|---------|
| `templates/index.html` | Single-file dashboard: embedded CSS (600 lines) + HTML structure + embedded JS (600 lines) |
| `src/api/ui.py` | FastAPI endpoint serving `GET /` → renders `templates/index.html` |

No separate CSS/JS asset files exist for the local UI — everything is self-contained in `templates/index.html`.

---

## Static Reference Files Inspected

| File | Used For |
|------|---------|
| `static-demo/css/styles.css` | Color palette, CSS variable values, spacing and typography decisions |
| `static-demo/index.html` | SVG logo icon path, badge patterns, layout reference |

---

## Files Changed

| File | Change type |
|------|-------------|
| `templates/index.html` | CSS block rewritten (dark theme); 3 HTML element changes; Google Fonts link added |

---

## What Was Visually Outdated

| Element | Issue |
|---------|-------|
| Color theme | Light (`#f5f7fb` background, `#ffffff` cards) — mismatched from static showcase dark theme |
| Accent color | Blue (`#2563eb`) — mismatched from static showcase cyan (`#00d4ff`) |
| Font family | `system-ui` only — no Inter or JetBrains Mono |
| Logo icon | Colored `<div>` box — no visual icon |
| Badge fills | Light pastel fills (e.g. `#fee2e2`) — not readable on dark background |
| Table row hover | Blue tint `rgba(37,99,235,0.04)` — wrong accent |
| Filter button active | `#dbeafe` fill — light mode only |
| Spinner accent | `#2563eb` top border — wrong accent |
| Input focus ring | `#2563eb` border — wrong accent |
| Tab active indicator | `#2563eb` bottom border — wrong accent |
| Component/raw boxes | White/light gray background — not dark-mode |
| Footer text | "Phase 8 Observability Dashboard" — stale internal reference |
| Footer/header color | Off-white text on white background — lost on dark |

---

## What Was Improved

### CSS Block (full replacement — same structure, dark values)

| Property | Before | After |
|----------|--------|-------|
| Body background | `#f5f7fb` | `#0a0e1a` + subtle 60px grid pattern |
| Card background | `#ffffff` | `#111827` |
| Card border | `#e5e7eb` | `#1e2d40` |
| Card shadow | `0 2px 8px rgba(0,0,0,0.06)` | `0 4px 16px rgba(0,0,0,0.3)` |
| Text primary | `#111827` | `#e6edf3` |
| Text secondary/muted | `#6b7280` | `#8b949e` |
| Text dim | `#9ca3af` | `#4b5563` |
| Accent | `#2563eb` | `#00d4ff` |
| Active tab indicator | `#2563eb` | `#00d4ff` |
| Active filter btn | `#dbeafe`/`#2563eb` | `rgba(0,212,255,0.12)`/`#00d4ff` |
| Primary button | `#2563eb` bg | `#00d4ff` bg, `#0a0e1a` text, glow on hover |
| Ghost button hover | `#2563eb` | `#00d4ff` |
| Input background | `#f9fafb` | `#0d1117` |
| Input focus ring | `#2563eb` border | `#00d4ff` + `rgba(0,212,255,0.1)` shadow |
| Severity badges | Light pastel fills | Dark semi-transparent fills readable on dark bg |
| Table header bg | `#f9fafb` | `#0d1117` |
| Table row hover | `rgba(37,99,235,0.04)` | `rgba(0,212,255,0.04)` |
| Sev bar track bg | `#e5e7eb` | `#1e2d40` |
| Raw/answer boxes | `#f9fafb` | `#0d1117` |
| Component cards | `#f9fafb` | `#0d1117` |
| Spinner accent | `#2563eb` | `#00d4ff` |
| Dot colors (header) | Green/amber/red | Shifted to `#30d158`/`#ff9500`/`#ff3b30` (iOS-style, matches static) |
| Header/nav background | `#ffffff` | `rgba(10,14,26,0.97)` + `backdrop-filter: blur(12px)` |
| Footer text | `#1e3a5a` | `#4b5563` |
| Collapsible toggle | Missing `font-family` | Added `font-family: inherit` |

### HTML changes (3)

1. **Google Fonts link** added in `<head>`: Inter (400/500/600/700) + JetBrains Mono (400/600)
2. **Logo icon** — `<div class="logo-icon"></div>` replaced with `<svg>` lightning bolt (same path as static showcase nav icon `M13 2L3 14h9l-1 8 10-12h-9l1-8z`)
3. **Footer label** — "Phase 8 Observability Dashboard" → "Observability Dashboard"

---

## What Was Intentionally Left Unchanged

| Item | Reason |
|------|--------|
| All JavaScript | Zero behavior changes — all data fetching, rendering, polling, and logic is identical |
| HTML structure | All section layouts, tab IDs, form elements, table structures unchanged |
| Backend routes | `/health`, `/alerts`, `/metrics`, `/query` — untouched |
| Section content | All labels, descriptions, placeholder texts, error messages unchanged |
| Collapsible toggle text | "Show / Hide raw JSON" — functional, keep as-is |
| Auto-refresh interval | 30 seconds — unchanged |
| Responsive breakpoints | `640px` and `768px` — unchanged |
| Semantic status colors | Green/amber/red for OK/warn/error — only shade adjusted for dark mode |

---

## Backend Constraints Affecting UI Polish

1. **No WebSocket support** — auto-refresh is poll-based (30s). A live streaming `<→ LIVE` indicator is not feasible without backend changes.
2. **`/query` endpoint** — RAG queries are synchronous; "Searching knowledge base..." spinner is the appropriate UX without a streaming response.
3. **Alert ring buffer** — max 200 alerts. Table pagination is not needed with current volume but would be warranted at higher throughput.
4. **No alert severity on V1 path** — some alerts may have `severity: null`; `severityClass()` already falls back to `'info'`.

---

## Remaining Minor UI Items (not blocking)

| Item | Notes |
|------|-------|
| Header subtitle on mobile | Hidden by `@media (max-width: 640px)` — intentional, still correct |
| Tab badge on Investigation and Health | No count shown (only Alerts tab has a badge) — correct behavior |
| Grafana/Prometheus links in Metrics tab | Still reference `localhost:3000` / `localhost:9090` — correct for local runtime |
| Empty state icons are emoji | Functional but could be replaced with SVG icons in a future polish pass |

---

## Validation Summary

| Check | Result |
|-------|--------|
| Google Fonts link present | ✓ |
| Dark body background (`#0a0e1a`) | ✓ |
| Dark card background (`#111827`) | ✓ |
| Cyan accent (`#00d4ff`) throughout | ✓ |
| Grid background pattern | ✓ |
| SVG logo icon (same as static showcase) | ✓ |
| No legacy light colors (`#f5f7fb`, `#ffffff`, `#2563eb`) | ✓ |
| All JS logic preserved (`refreshAll`, `renderDashboard`, etc.) | ✓ |
| All backend routes preserved (`/health`, `/alerts`, `/query`) | ✓ |
| Footer updated (no "Phase 8") | ✓ |
| File remains single self-contained HTML | ✓ |
