# Static Demo — Predictive Log Anomaly Engine V3

This folder contains a lightweight static showcase website built for the **Predictive Log Anomaly Engine V3** project.

---

## Purpose

The static demo is a **presentation and demonstration layer** for the existing system.

It does not replace the real project, model pipeline, notebooks, or runtime observability stack.

Instead, it helps present the project in a clear and visual way during:

- academic presentation
- live demo sessions
- high-level system walkthroughs

---

## What it shows

The static site visually summarizes the same system documented and implemented in the main project:

- project overview and value proposition
- problem framing: reactive monitoring vs. behavioral anomaly detection
- architecture overview (V1 + V2 + optional V3 semantic layer)
- 9-stage pipeline walkthrough (V2 core pipeline + V3 semantic enrichment step)
- live simulation of logs and alerts
- results / observability section
- technical stack summary

---

## Relationship to the main project

This site is aligned with the full project artifacts, including:

- project codebase (`src/`)
- design/specification documents (`docs/`)
- Jupyter notebook demos (`notebooks/`)
- GPU notebook execution flow
- observability and monitoring setup

It should be treated as a **visual companion demo** to the real implemented system.

---

## Run locally

Option 1 — open directly:

- Open `index.html` in a browser

Option 2 — run a simple local server:

```bash
cd static-demo
python -m http.server 8080
```

Then open:

http://localhost:8080

---

## Notes

To strengthen the credibility of the demo, replace placeholder visuals with real project screenshots where available, especially:

- Grafana dashboard (`assets/grafana-dashboard.png`)
- alert/output screenshots
- FastAPI Swagger UI screenshot

See `assets/ASSETS_NEEDED.txt` for instructions.

---
