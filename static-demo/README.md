# Static Demo — Predictive Log Anomaly Engine V2

This folder contains a lightweight static showcase website built for the **Predictive Log Anomaly Engine V2** project.

---

## Purpose

The static demo was created as a **presentation and demonstration layer** for the existing system.

It does not replace the real project, model pipeline, notebook, or GPU-based execution flow.

Instead, it helps present the project in a clear and visual way during:

- academic presentation
- live demo sessions
- high-level system walkthroughs

---

## What it shows

The static site visually summarizes the same system documented and implemented in the main project:

- project overview and value proposition
- problem framing: reactive monitoring vs. behavioral anomaly detection
- architecture overview
- 8-stage pipeline walkthrough
- live simulation of logs and alerts
- results / observability section
- technical stack summary

---

## Relationship to the main project

This site is aligned with the full project artifacts, including:

- project codebase
- design/specification documents
- Jupyter notebook demos
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

- Grafana dashboard
- alert/output screenshots
- architecture export if needed

---
