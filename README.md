# Predictive Log Anomaly Engine v3

AI-powered observability, anomaly detection, and semantic enrichment system for software logs.

The **Predictive Log Anomaly Engine** analyzes behavioral patterns in log streams,
detects abnormal activity before full service failure occurs,
and is evolving into a **hybrid AI observability platform** with semantic and explanation-aware capabilities.

---

## Tech Stack

Python • FastAPI • Docker • Prometheus • Grafana • PyTorch • Sentence Transformers • Pytest • GitHub Actions

---

## Overview

Traditional monitoring systems usually detect failures only after they already impact the system, for example through CPU spikes, crashes, or error bursts.

The Predictive Log Anomaly Engine takes a more proactive approach by analyzing behavioral log patterns rather than relying only on infrastructure metrics.

The platform currently combines:

- parsing and normalization of raw logs
- template extraction and event sequencing
- anomaly detection through behavioral modeling
- severity scoring and alert generation
- FastAPI runtime services
- Prometheus and Grafana observability
- a growing V3 semantic layer for future explanation-aware enrichment

The project is evolving from a pure anomaly detection engine into a **hybrid AI observability system** that preserves the existing V1/V2 runtime backbone while adding semantic capabilities in a modular way.

---

## Key Capabilities

- Real-time log anomaly detection using behavioral sequence analysis
- Multi-stage AI inference pipeline
- Severity scoring and alert generation
- FastAPI service exposing ingestion and investigation endpoints
- Prometheus metrics and Grafana dashboards
- Docker-based deployment
- CI/CD validation with automated tests
- Interactive demonstration notebooks
- V3 semantic scaffold prepared for Hugging Face-based enrichment and explanation workflows

---

## System Architecture

The system currently follows a full AI runtime pipeline:

Logs
↓
Parsing & Template Mining
↓
Sequence Builder
↓
Embedding Generation (Word2Vec / behavioral features)
↓
Behavior Modeling
↓
Autoencoder Anomaly Detection
↓
Severity Classification
↓
Alert Manager
↓
FastAPI Service
↓
Prometheus + Grafana
↓
Investigation UI

V3 direction adds an optional semantic enrichment layer on top of confirmed anomaly results,
enabling future explanation-aware workflows without replacing the existing runtime backbone.

---

## Main Components

### Parsing & Template Mining

Transforms raw logs into structured templates that reduce noise while preserving behavioral patterns.

---

### Sequence Builder

Builds event windows and behavioral sequences that represent operational system activity.

---

### Embedding Engine

Uses **Word2Vec embeddings** to convert log templates into semantic vector representations.

---

### Behavior Model

Learns normal operational sequences using a deep learning behavioral model.

---

### Autoencoder Anomaly Detector

Detects abnormal patterns by measuring reconstruction error from learned normal behavior.

---

### Severity Classifier

Assigns severity levels to detected anomalies based on learned patterns and rule-based signals.

---

### Alert Manager

Generates alerts with severity levels and handles alert lifecycle management.

---

### FastAPI Runtime Service

Provides:

- ingestion endpoints
- system health signals
- alert queries
- investigation endpoints
- UI serving

---

### Observability Layer

The project integrates a full observability stack:

- Prometheus metrics collection
- Grafana dashboards
- system health visibility
- ingestion monitoring
- alert activity monitoring

---

### Demo Notebooks

The repository includes interactive demonstration notebooks for system explanation, visualization, and walkthrough purposes.

These notebooks help demonstrate:

- log ingestion simulation
- anomaly scoring visualization
- runtime behavior analysis
- system evolution from V2 anomaly detection toward V3 semantic enrichment

---

### Demo Scripts

The repository also includes runnable demo scripts:

demo/
predictive_log_anomaly_engine_demo.py
predictive_log_anomaly_engine_gpu_demo.py

These scripts generate synthetic events and visualize anomaly detection behavior.

---

## Quick Start

Build and run the system:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## Open the Services

### API / UI

```web
http://localhost:8000
```

### Prometheus

```web
http://localhost:9090
```

### Grafana

```web
http://localhost:3000
```

---

## Demo Walkthrough

| Step  | Action                | Expected Result                            |
|-------|-----------------------|--------------------------------------------|
| 1     | Ingest events         | Synthetic logs enter the runtime pipeline  |
| 2     | Open alerts view      | Alerts appear with severity and score      |
| 3     | Open dashboards       | Prometheus / Grafana show runtime metrics  |
| 4     | Query investigation   | System returns investigation information   |

---

## Evaluation (V1 vs V2)

The repository includes evaluation tooling for comparing the **V1 and V2 anomaly pipelines** on labeled datasets.

## Prerequisites

All four model artifacts must exist:
models/embeddings/word2vec.model
models/behavior/behavior_model.pt
models/anomaly/anomaly_detector.pt
models/severity/severity_classifier.pt

Train them if missing:

python -m training.train_embeddings
python -m training.train_behavior_model
python -m training.train_autoencoder
python -m training.train_severity_model

---

## Run Evaluation

python scripts/evaluate_v2.py

Evaluates labeled HDFS sessions and generates performance comparison.

Output is written to:

evaluation_report.json

---

## Testing

### Fast tests

pytest -m "not slow"

### Integration tests

pytest -m integration

---

## Documentation Structure

Additional documentation is available under:

```text
docs/

docs/current_system
docs/api
docs/operations
docs/system_validation
```

---

## Project Team

Developed as part of an **Applied AI Engineering project**.
DevOps, QA, Backend Engineering, Frontend Development, UI Design, Architecture Design, Technical Specification.

**Oren Salami**
**Dan Kalfon**
**Nahshon Raizman**
**Jonathan Finkelstein**

---

## Project Status

Current repository status:

- runtime inference pipeline implemented
- anomaly detection models integrated
- alert pipeline implemented
- observability stack integrated
- Grafana and Prometheus configured
- Docker deployment implemented
- evaluation framework implemented
- investigation UI implemented
- V3 semantic scaffold added as the foundation for future semantic enrichment and explanation-aware workflows

The project currently preserves the working V1/V2 backbone while preparing the next V3 integration layer in a modular and backward-compatible way.

**All tests passing (578 tests)** and full containerized runtime included.

---

## Career / Portfolio Value

This project demonstrates practical work across:

- AI runtime systems
- anomaly detection pipelines
- FastAPI backend engineering
- observability architecture
- Docker-based deployment
- CI/CD validation
- investigation-oriented system design

Built as part of an **Applied AI Engineering project**.

---
