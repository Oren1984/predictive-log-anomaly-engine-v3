# Predictive Log Anomaly Engine v3

An AI-driven log anomaly detection platform that identifies system instability before failures occur — by analyzing behavioral patterns in log streams rather than static thresholds.

---

## What This Project Does

Traditional monitoring catches failures after they happen. This system works earlier.

It analyzes behavioral patterns in log streams — not just error counts or infrastructure metrics — and flags deviations before they escalate. Logs are parsed, sequenced, and scored through a multi-stage ML pipeline. When anomaly thresholds are crossed, alerts are generated, deduplicated, and forwarded in real time.

A modular V3 semantic layer adds optional explanation and similarity enrichment on top of the core runtime — without replacing it.

---

## Key Capabilities

- Real-time log anomaly detection via behavioral sequence analysis
- Multi-stage ML pipeline for behavioral anomaly detection and severity scoring
- Three versioned API paths (V1, V2, V3) with backward compatibility
- Alert pipeline with deduplication, cooldown, and n8n webhook forwarding
- Prometheus metrics + Grafana dashboards out of the box
- Optional semantic explanation layer (V3) powered by Sentence Transformers
- Fully Dockerized — single command to run the complete stack

---

## System Flow

Raw Logs → Parsing → Sequence → Model → Anomaly → Alert → Metrics

(Detailed pipeline includes embedding, autoencoder, and severity classification.)

V3 adds an optional semantic enrichment step after anomaly confirmation: explanation text, evidence tokens, and similarity scoring against prior alerts.

---

## Architecture Snapshot

| Layer             | Technology                                    |
|-------------------|-----------------------------------------------|
| API backend       | FastAPI (versioned: V1 / V2 / V3)             |
| Inference engines | V1 rolling-window + V2 four-stage ML pipeline |
| Semantic layer    | Sentence Transformers (`all-MiniLM-L6-v2`)    |
| Observability     | Prometheus + Grafana                          |
| Runtime           | Docker Compose                                |

**Core Technologies:** Python · FastAPI · PyTorch · Sentence Transformers · Prometheus · Grafana · Docker · Pytest

---

## Quick Start

```bash
docker compose -f docker/docker-compose.yml up -d
```

---

## Service URLs

//localhost:8000

Health check	http://localhost:8000/health

Metrics	http://localhost:8000/metrics

Prometheus	http://localhost:9090

Grafana	http://localhost:3000

To enable V3 semantic enrichment:

```bash
SEMANTIC_ENABLED=true docker compose -f docker/docker-compose.yml up -d
```

---

## Project Structure

src/         Core system — API, runtime, alerts, observability, semantic layer
docker/      Runtime stack — Compose files, Dockerfile
tests/       Automated test suite (unit + integration)
training/    Model training scripts
artifacts/   Runtime assets — vocab, thresholds, templates
scripts/     Data pipeline and utility scripts
notebooks/   Walkthroughs and demos
docs/        Architecture docs and reports
archive/     Legacy scripts and generated history

---

## Documentation & Notebooks

Documentation	           |    Path
---------------------------|----------------------------------------------
Repository Structure	   |docs/repository_structure.md

Technical Walkthrough	   |notebooks/technical_walkthrough.ipynb

Repository Structure	   |docs/repository_structure.md

Interactive Structure	   | notebooks/repository_structure.ipynb

V3 Semantic Demo	       | notebooks/v3_semantic_demo.ipynb

V3 Design Plan	           | docs/V3_REFACTOR_AND_HF_INTEGRATION_PLAN.md

---

## Use Cases

- Early detection of system instability in production environments
- Behavioral monitoring of distributed systems
- Log-driven incident prediction and alerting
- Observability augmentation beyond traditional metrics

---

## Project Status

- Docker build and runtime: validated
- Health and metrics endpoints: confirmed
- Alert pipeline, deduplication, and n8n forwarding: complete
- V3 semantic layer: integrated and tested
- Observability stack: Prometheus + Grafana configured
- Test suite: validated (full execution confirmed in containerized environment)
- Documentation: comprehensive and up to date
- Notebooks: technical walkthrough and V3 demo completed

---

## Project Team

Developed as part of an Applied AI Engineering project.

**Oren Salami · Dan Kalfon · Nahshon Raizman · Jonathan Finkelstein**

The team collaborated across multiple domains including system architecture, backend development, observability, and AI pipeline design.

**Core Areas:** DevOps · Backend · Frontend · UI · QA · Architecture · Technical Design

---

