# Predictive Log Anomaly Engine V3
### Full Project Presentation

---

## 1. Project Overview

### What Is the System?

The **Predictive Log Anomaly Engine V3** is a production-grade, AI-driven platform that detects anomalies in system log streams **before** failures occur. It analyzes behavioral patterns across sequences of log events using a layered machine-learning pipeline, fires structured alerts, and exposes everything to a real observability stack (Prometheus + Grafana).

### The Problem It Solves

| Traditional Monitoring | This System |
|---|---|
| Threshold on individual metrics | Pattern analysis across sequences |
| Detects failures **after** they happen | Detects behavioral drift **before** failure |
| Static rules (e.g., CPU > 90%) | Learned models that adapt to system behavior |
| High false-positive alert noise | Deduplication + cooldown policies |

**Core insight:** A server rarely fails instantly. It shows a behavioral signature — unusual log patterns, increasing error density, novel templates — before it crashes. This system captures that signature.

### Target Use Case

- Large-scale distributed systems (HDFS, microservices, cloud infra)
- DevOps teams who need early warning, not post-mortem alerts
- Any organization ingesting structured or semi-structured logs at scale

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT / LOG AGENT                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ POST /ingest  (V1 / V2 / V3)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI API LAYER                            │
│  routes.py (V1) │ routes_v2.py (V2) │ routes_v3.py (V3)           │
│                     pipeline.py (Orchestrator)                      │
└───────┬─────────────────────────────────────────────────┬───────────┘
        │                                                 │
        ▼                                                 ▼
┌────────────────────┐                        ┌──────────────────────┐
│   INFERENCE LAYER  │                        │   ALERT MANAGER      │
│                    │                        │                      │
│  V1: IsolationFor. │   ──→ RiskResult ──→  │  Policy enforcement  │
│  V2: LSTM + AE     │                        │  Cooldown / dedup    │
│  Transformer: GPT  │                        │  Severity buckets    │
└────────────────────┘                        └──────────┬───────────┘
        ▲                                                 │
        │                                                 ▼
┌────────────────────┐                        ┌──────────────────────┐
│  SEQUENCE BUFFER   │                        │  SEMANTIC LAYER (V3) │
│  Per-stream deque  │                        │  Rule-based explain  │
│  Rolling window    │                        │  Embedding similarity│
└────────────────────┘                        └──────────┬───────────┘
        ▲                                                 │
        │                                                 ▼
┌────────────────────┐                        ┌──────────────────────┐
│  LOG PARSING       │                        │  N8N WEBHOOK CLIENT  │
│  RegexLogParser    │                        │  Dry-run outbox OR   │
│  JsonLogParser     │                        │  Live webhook POST   │
│  TemplateMiner     │                        └──────────────────────┘
└────────────────────┘

        OBSERVABILITY STACK
        ┌─────────────────────────────────┐
        │  /metrics → Prometheus → Grafana │
        │  /health  → Load balancer probe  │
        └─────────────────────────────────┘
```

### Components Summary

| Component | File(s) | Role |
|---|---|---|
| API Layer | `src/api/` | Receives requests, orchestrates pipeline |
| Inference Engine V1 | `src/runtime/inference_engine.py` | IsolationForest baseline scoring |
| Inference Engine V2 | `src/runtime/pipeline_v2.py` | LSTM → Anomaly Detector → Severity |
| Sequence Buffer | `src/runtime/sequence_buffer.py` | Per-stream rolling window |
| Alert Manager | `src/alerts/` | Policy, cooldown, dedup, severity |
| Semantic Layer | `src/semantic/` | Explainability + similarity (V3) |
| Observability | `src/observability/` | Prometheus metrics, health checks |
| Security | `src/security/` | API key auth middleware |
| Parsing | `src/parsing/` | Log → structured event |

### Data Flow (Step-by-Step)

```
1. Raw log string arrives at POST /v2/ingest
2. RegexLogParser / JsonLogParser → structured LogEvent
3. _V2LogTokenizer → template matched → token_id via TemplateMiner
4. Word2Vec lookup → float32 embedding vector
5. SequenceBuffer (per stream_key) → append + emit window when full
6. InferenceEngineV2:
     SystemBehaviorModel (LSTM) → context vector
     AnomalyDetector → reconstruction error → is_anomaly flag
     SeverityClassifier → Info / Warning / Critical
7. RiskResult created (score, is_anomaly, threshold, evidence_window)
8. AlertManager.emit(risk_result):
     - Is anomaly? → score above threshold? → cooldown elapsed?
     - Yes to all → fire Alert (UUID, severity, score, evidence)
9. Alert dispatched to N8nWebhookClient (or dry-run outbox)
10. V3 optional: semantic explanation + similarity lookup attached
11. Prometheus counters/histograms updated
12. HTTP response returned to caller
```

---

## 3. Key Technologies

### Why FastAPI?
- **Async-first:** All route handlers are `async def`, enabling high-throughput ingest without blocking
- **Pydantic v2:** Automatic request/response validation with zero boilerplate
- **OpenAPI docs:** Auto-generated `/docs` endpoint for free
- Alternative considered: Flask — rejected because synchronous and missing native async support

### Why PyTorch (LSTM + Transformer)?
- **LSTM** is the natural choice for sequential, time-ordered log data — it maintains hidden state across the window
- **Autoregressive Transformer** learned next-token distribution; anomalies are low-probability sequences
- PyTorch gives fine-grained control over custom training loops vs. Keras abstraction overhead

### Why IsolationForest (V1 Baseline)?
- Unsupervised — no labeled anomaly data required to bootstrap
- O(n log n) training, O(log n) inference — fast and lightweight
- Serves as a **reproducible baseline** to compare V2 improvements against

### Why Word2Vec (not one-hot)?
- Dense embeddings capture semantic relationships between log templates
- Trained on log-specific sequences — "template 42 near template 17 = common pattern"
- More robust than raw token IDs fed to the LSTM

### Why Docker + Docker Compose?
- Reproducibility: dev, staging, prod run identical stacks
- Orchestrates three services (API, Prometheus, Grafana) with one command
- Volume mounts separate model artifacts from application code

### Why Prometheus + Grafana?
- Industry standard observability stack
- Prometheus scrapes `/metrics` every 15s — time-series persistence
- Grafana: live dashboards for alert rates, latency histograms, health status

### Why Sentence Transformers (V3)?
- `all-MiniLM-L6-v2`: lightweight, fast, runs on CPU
- Converts log evidence text → dense embedding for similarity comparison
- Enables "find alerts that look like this one" without an external LLM

---

## 4. Implementation Walkthrough

### Phase 1 — Log Ingestion
A log agent POSTs to `/ingest` (V1) or `/v2/ingest` (V2). The API validates the request via Pydantic schema. For V2, raw strings are accepted and parsed internally.

### Phase 2 — Tokenization & Embedding
```
Raw: "2024-01-15 08:32:11 ERROR dfs.DataNode$PacketResponder: PacketResponder 1 Exception"
 ↓ RegexLogParser
LogEvent(timestamp=..., level="ERROR", message="dfs.DataNode$PacketResponder: ...")
 ↓ TemplateMiner (9-step normalization: strip BLK IDs, IPs, hex, numbers)
Template: "dfs.DataNode$PacketResponder: PacketResponder <NUM> Exception"
template_id = 47
 ↓ token_id = template_id + 2 = 49   (OFFSET convention)
 ↓ Word2Vec["49"] → float32[128]
```

### Phase 3 — Window Buffering
Each `stream_key = service:session_id` has its own `deque(maxlen=window_size)`. When the buffer reaches `window_size`, a window is emitted. Stride controls how many new events must arrive before the next emit.

### Phase 4 — Inference
**V1:** Hand-crafted features (token entropy, frequency, temporal spacing) → IsolationForest → anomaly score.

**V2 (four stages):**
1. `SystemBehaviorModel` (LSTM): 128-dim context vector from token sequence
2. `AnomalyDetector`: Reconstruction error — if the sequence is "unusual", reconstruction loss is high
3. `SeverityClassifier`: Multi-class softmax → Info / Warning / Critical

### Phase 5 — Alert Decision
```python
# AlertManager.emit()
if not risk_result.is_anomaly:          return []   # below threshold
if score < policy.threshold:            return []   # policy filter
if stream_key in recent_alerts         
   and elapsed < cooldown_seconds:      return []   # cooldown suppression
# → Fire alert
```

### Phase 6 — Observability
Every ingest call: counters increment, latency histograms record. Prometheus scrapes every 15s. Grafana renders live.

---

## 5. AI Logic

### V1 — Baseline (IsolationForest)
- **Features:** Token frequency distribution, Shannon entropy over window, inter-arrival timing
- **Model:** Unsupervised, trained on "normal" log sequences from HDFS dataset
- **Decision:** Anomaly score → compare to threshold → binary flag
- **Limitation:** Fixed feature set, no temporal context across windows

### V2 — Four-Stage Pipeline
- **LSTM Behavior Model:** Encodes the sequence into a 128-dim context vector capturing temporal order and co-occurrence
- **Anomaly Detector:** Reconstructs the sequence from the context vector; high reconstruction error = unfamiliar pattern
- **Severity Classifier:** Trained on labeled examples of Info/Warning/Critical sequences
- **Limitation:** Requires pre-trained models; LSTM context resets between sessions; no online learning

### V3 — Semantic Enrichment (Rule-Based)
- **Not an LLM** — pure heuristic rules applied to evidence_window:
  1. Error keyword scan (error, exception, timeout, fail, traceback)
  2. Template diversity check (< 3 unique templates = monotonous pattern)
  3. Density check (> 40 tokens = high-volume burst)
- **Similarity:** Cosine distance between current alert embedding and ring buffer of last 200 alerts
- **Limitation:** Rules may miss domain-specific anomaly types; ring buffer is in-memory only

### Why Not an LLM?
- LLMs add latency (100ms–2s per call), require API keys, add cost per alert
- Rule-based explanation is deterministic, auditable, zero-cost
- Future: can swap `RuleBasedExplainer` for an LLM explainer without changing the interface

---

## 6. Demo Explanation

### Pre-Demo Checklist
```bash
# Option A — Docker (recommended for demo)
docker compose -f docker/docker-compose.yml up -d

# Option B — Local
pip install -r requirements/requirements.txt
python main.py

# Verify health
curl http://localhost:8000/health
```

### Demo Flow (6 Steps)

**Step 1 — Show the architecture slide** and explain the problem being solved.

**Step 2 — Open `/docs`** (Swagger UI at `http://localhost:8000/docs`)
- Show the three endpoint groups: V1, V2, V3
- Explain the versioning strategy

**Step 3 — Ingest a normal event (V1)**
```json
POST /ingest
{
  "token_id": 5,
  "service": "hdfs",
  "session_id": "session-demo-1",
  "timestamp": 1700000001,
  "label": 0
}
```
- Show that `is_anomaly: false` — normal behavior, no alert

**Step 4 — Ingest an anomalous V2 log**
```json
POST /v2/ingest
{
  "log_line": "2024-01-15 ERROR dfs.DataNode: PacketResponder Exception timeout",
  "service": "hdfs",
  "session_id": "session-demo-2"
}
```
- Show tokenization output, RiskResult, alert fired

**Step 5 — Check `/alerts`**
- Show the alert object: UUID, severity, score, evidence_window, timestamps

**Step 6 — Show Grafana** (`http://localhost:3000`)
- Point to: alert rate graph, ingest latency histogram, health gauge

### What to Say at Each Step
*(see speaking_script.md for full scripts)*

### Demo Safety Notes
- `DEMO_MODE=true` ensures scoring works even without pre-trained models
- Warmup events run on startup — you will see alerts immediately
- `ALERT_COOLDOWN_SECONDS=0` ensures every anomaly fires (good for demo)

---

## 7. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Token ID consistency across V1 and V2 | Standardized `token_id = template_id + 2` convention across all encoders |
| Alert storms (same anomaly fires repeatedly) | Per-stream cooldown policy; configurable window |
| Models not available in CI / demo | `DEMO_MODE=true` flag + synthetic scoring fallback |
| Semantic model download at startup | Lazy loading + optional `SEMANTIC_ENABLED` flag |
| Cross-contamination between log streams | Per-stream-key `deque` buffer (not global) |
| Testing time-based cooldowns | `clock_fn` dependency injection in `AlertManager` |
| Backward compatibility as API evolved | Optional fields with `None` defaults in V3 schema |

---

## 8. Limitations

### What Is NOT Implemented
- **Online/incremental learning:** Models are trained offline; no real-time model updates as new data arrives
- **Distributed mode:** Single-instance only; no horizontal scaling or distributed buffering
- **Cross-stream correlation:** Each stream is analyzed independently; no cross-service causation detection
- **Persistent alert storage:** Alert buffer is in-memory (configurable size, default 200); no database persistence
- **Full LLM integration:** V3 uses rule-based explanation, not an actual language model
- **Streaming ingestion protocol:** HTTP REST only; no Kafka/gRPC connector built-in

### What Is Simulated in Demo Mode
- When `DEMO_MODE=true`, model scoring is replaced by synthetic scores
- `DEMO_SCORE=100.0` forces anomaly detection regardless of model state
- Warmup events are synthetic (generated by `src/synthetic/`) not real log data

### Honest Assessment
The system demonstrates the full architectural pattern and production-quality code structure. The ML models work with real HDFS log data. Demo mode exists for presentation safety, not because the real models don't function.

---

## 9. Future Improvements

### Realistic Next Steps

1. **Online learning adapter** — Allow the LSTM to fine-tune on the most recent window (incremental update, not full retrain)

2. **Kafka connector** — Replace REST ingest with a Kafka consumer for true stream processing at scale

3. **Cross-stream correlation** — When anomalies co-occur in multiple services within a short window, generate a compound "cascade" alert

4. **Persistent alert store** — Replace in-memory buffer with PostgreSQL or Redis for durability and querying

5. **LLM explanation mode** — Swap `RuleBasedExplainer` for a Claude/GPT-4 call when deeper explanation is needed

6. **Active feedback loop** — Operator marks alerts as True Positive / False Positive → use labels to retrain severity classifier

7. **Grafana alerting rules** — Move from Prometheus alert rules to Grafana unified alerting with PagerDuty integration

8. **Model registry** — Version and track model artifacts with MLflow or DVC instead of manual file management

---

## 10. Conclusion

### Why This Project Is Strong

- **Real ML pipeline:** Not a tutorial wrapper. V2 has four trained components (tokenizer, embeddings, LSTM, classifier) working in sequence
- **Production architecture:** FastAPI + Docker + Prometheus + Grafana is the real-world observability stack
- **Three API versions:** Shows ability to evolve a system without breaking existing clients
- **578 tests:** Unit, integration, and system-level coverage demonstrates engineering discipline
- **Clean abstractions:** Parser, SequenceBuilder, Explainer are interfaces — new implementations can be plugged in
- **Honest demo mode:** The system transparently supports demo/CI mode while the real ML path is fully implemented

### What It Demonstrates

- Full-stack ML system design (data pipeline → model → API → monitoring)
- Software engineering maturity (versioning, testing, containerization, security)
- Ability to work with real-world, messy data (log parsing, template mining)
- End-to-end thinking from log line to Grafana dashboard
