# Q&A Preparation — Predictive Log Anomaly Engine V3
### Examiner Mode: Expected Questions + Strong Answers

---

## CATEGORY 1 — Problem & Motivation

### Q: Why is log anomaly detection better than simple threshold-based alerting?

**Answer:**
Threshold-based alerting monitors individual metrics at a single point in time. The problem is that individual metrics are noisy — a momentary CPU spike is irrelevant, but ten seconds of unusual log patterns preceding a failure is a signal. Our system analyzes sequences of events, capturing temporal context that single-metric monitors cannot. Additionally, thresholds require manual tuning per metric per system. Our ML models learn what "normal" looks like automatically from historical data.

**Key point:** Thresholds catch *what* failed. Sequence anomaly detection catches *how* the failure is developing.

---

### Q: Why did you choose log data specifically, not CPU/memory metrics?

**Answer:**
Logs are the richest signal in a distributed system. Metrics tell you resource utilization. Logs tell you what the software is doing, what templates are being generated, what sequence of operations is occurring. The HDFS dataset we trained on has ground-truth labels — specific log sequences that preceded real block failures. This gives us a labeled training signal that raw metrics rarely provide.

---

## CATEGORY 2 — ML & Model Choices

### Q: Why IsolationForest for V1? Why not something simpler like Z-score or CUSUM?

**Answer:**
Z-score and CUSUM work well on univariate metrics. Our V1 feature vector is multi-dimensional — token frequency distribution, entropy, temporal spacing — and IsolationForest naturally handles high-dimensional feature spaces without distributional assumptions. It's unsupervised, so we don't need labeled anomaly examples to train it. It also gives a continuous anomaly score (not just binary), which we use for severity bucketing.

---

### Q: Why LSTM for the behavior model? Why not a simple feedforward network?

**Answer:**
A feedforward network treats each token in the window independently — it loses the sequential ordering information. LSTMs maintain a hidden state that accumulates context across the sequence. For log data, order matters critically: "token A then B then C" has different meaning than "C then A then B". The LSTM captures that ordering. We considered Transformers (self-attention), but for short sequences (window_size=10), the LSTM was faster to train and comparable in accuracy.

---

### Q: How did you train the models? What data did you use?

**Answer:**
We used the HDFS (Hadoop Distributed File System) log dataset — a publicly available benchmark with ~11 million log events and ground-truth labels for anomalous sessions. The data was preprocessed through our 9-step template normalization pipeline to extract stable templates. The LSTM and AnomalyDetector were trained on normal sessions only — unsupervised — so that reconstruction error spikes on anomalous patterns. The SeverityClassifier was trained with labeled examples.

---

### Q: What is the reconstruction error, exactly? How does it work?

**Answer:**
The AnomalyDetector is an autoencoder structure: it takes the LSTM context vector and tries to reconstruct the original embedding sequence. During training, it only sees normal sequences, so it learns to reconstruct them accurately. When an anomalous sequence is fed in — one it has never seen patterns like before — the reconstruction is poor. We measure Mean Squared Error between the original embeddings and the reconstruction. If MSE exceeds the trained threshold, we flag is_anomaly=True. The threshold is calibrated on a held-out validation set of normal sequences.

---

### Q: What is the anomaly score? Is it a probability?

**Answer:**
No — it is not a calibrated probability. For V1, it's the IsolationForest anomaly score (negative mean depth). For V2, it's the reconstruction MSE from the AnomalyDetector. Both are monotonic with anomalousness — higher means more anomalous — but they are not probability estimates. We apply a threshold to convert to binary is_anomaly. We're explicit about this in the API schema and do not claim false precision.

---

### Q: How do you handle the cold-start problem — a stream with fewer events than the window size?

**Answer:**
The `SequenceBuffer` doesn't emit a window until it has accumulated `window_size` events. So there is no scoring until the buffer is full. This means the first `window_size - 1` events of any new stream are in a "warm-up" period. This is an honest limitation: very short sessions may never fill the buffer and will produce no score. For V2, window_size defaults to 10, so the warm-up is short.

---

### Q: What if the system encounters a completely new log template it has never seen?

**Answer:**
The TemplateMiner assigns new templates a new template_id, which gets a corresponding token_id. The Word2Vec model will not have an embedding for this token — it returns a zero vector by default. This is a known limitation: out-of-vocabulary templates are represented as all-zeros, losing their semantic signal. The LSTM will process the zero vector, likely producing a high reconstruction error — which is actually the correct behavior, since genuinely novel templates are a signal of anomalous behavior.

---

## CATEGORY 3 — Architecture & Design

### Q: Why three separate API versions (V1, V2, V3)? Why not just one endpoint?

**Answer:**
API versioning is a production engineering practice, not academic choice. V1 existed first — existing clients depend on it. Replacing it with V2 would break those clients. Adding V2 as a separate route allows us to evolve without disruption. V3 adds optional fields to the response schema — all Optional with None defaults — so V3 clients and V1 clients can coexist. This is how APIs evolve in practice.

---

### Q: Why per-stream buffering instead of a global buffer?

**Answer:**
A global buffer would mix events from service A and service B. Service A's unusual log pattern could be contaminated by service B's normal events. The anomaly score for the mixed stream would be meaningless. Per-stream-key buffering — keyed on `service:session_id` — ensures each stream is analyzed in isolation. This is a fundamental correctness requirement, not an optimization.

---

### Q: What happens if the per-stream buffer grows unbounded? (thousands of unique streams)

**Answer:**
The buffer manager caps the number of active stream buffers at 5,000 by default (LRU eviction). If the 5,001st stream arrives, the least-recently-used stream's buffer is evicted. This is a pragmatic limit. A production system would externalize the buffer to Redis with TTL-based expiry. This is called out in our future improvements.

---

### Q: Why use a dry-run outbox for n8n by default, instead of live webhook?

**Answer:**
Safety by default. During development and demo, accidentally firing live webhooks to an n8n workflow could trigger real downstream actions — PagerDuty pages, Slack messages, emails. The dry-run mode writes JSON files to `artifacts/n8n_outbox/` for inspection without external side effects. Enabling live webhook requires explicit `N8N_WEBHOOK_URL` environment variable — an intentional activation step.

---

### Q: How does the authentication middleware work?

**Answer:**
`AuthMiddleware` inspects the `X-API-Key` header on every request. If the key matches the configured `API_KEY` environment variable, the request proceeds. A configurable list of public paths (default: `/health`, `/metrics`, `/`) bypasses authentication — these are needed for load-balancer probes and Prometheus scraping without credentials. `DISABLE_AUTH=true` disables the middleware entirely, which is used in demo and CI environments.

---

## CATEGORY 4 — "What Happens If..." Edge Cases

### Q: What happens if the ML models fail to load at startup?

**Answer:**
The `Pipeline` class validates artifact paths on startup. If a required artifact is missing, the service logs the error and transitions to `DEMO_MODE` — which uses synthetic scoring. The `/health` endpoint reflects a `"degraded"` status (not healthy, not down). Prometheus's `service_health` gauge drops from 1.0 to 0.5. The service continues accepting requests; it won't silently fail.

---

### Q: What happens if the same anomaly fires continuously for 60 seconds?

**Answer:**
The first anomaly fires an alert. The `AlertManager` records the timestamp for that `stream_key`. Every subsequent anomaly from the same stream within `cooldown_seconds` is suppressed — it does not fire a new alert. The anomaly score continues to be computed and returned in the response (is_anomaly=True), but no new Alert object is created. Once the cooldown expires, the next anomaly will fire again. This prevents alert storms while still computing scores continuously.

---

### Q: What if two services generate the same session_id?

**Answer:**
The stream_key is `f"{service}:{session_id}"` — both fields combined. So `hdfs:session-1` and `nginx:session-1` are different stream keys with independent buffers and cooldown states. Cross-service collision is impossible by design.

---

### Q: What if an attacker sends malformed JSON to the API?

**Answer:**
Pydantic v2 validates every request body against the schema. A malformed JSON or missing required field returns HTTP 422 Unprocessable Entity before reaching any business logic. SQL injection is not possible — we have no SQL database. Command injection is not possible — we don't shell out on ingest. XSS is not possible on JSON APIs. The main risk surface is denial-of-service via large payloads, which would be handled at the load-balancer layer in production.

---

### Q: What happens under high load? Is the API thread-safe?

**Answer:**
FastAPI with Uvicorn runs in an async event loop. All route handlers are `async def`. The per-stream `deque` is thread-safe in CPython due to the GIL. For multiple uvicorn workers (via `--workers N`), buffer state would be split across processes — each worker would have an independent buffer, which would cause incomplete windows. The current implementation is single-worker by design; multi-worker is listed as a future improvement requiring external state.

---

## CATEGORY 5 — Semantic Layer (V3)

### Q: Why rule-based explanation instead of an LLM?

**Answer:**
Three reasons: latency, cost, determinism. An LLM call adds 100ms–2000ms per alert. At high alert rates, this becomes a bottleneck. LLM API calls cost money per token — alerts could fire thousands of times per day. And LLM outputs are non-deterministic — the same alert might get a different explanation each time, making debugging and auditing harder. Rule-based explanation is deterministic, free, and sub-millisecond. The interface is designed so swapping in an LLM requires changing one class, not the architecture.

---

### Q: How is the semantic similarity computed?

**Answer:**
When an alert fires, we generate a text summary of the evidence window (template previews, severity, service). We pass this through `all-MiniLM-L6-v2` (Sentence Transformers) to get a 384-dimensional embedding vector. We then compute cosine similarity between this embedding and every embedding in the ring buffer (last 200 historical alerts). The top-3 closest alerts are returned as `top_similar_events`. Cosine similarity measures the angle between vectors — a score of 1.0 means identical, 0.0 means orthogonal.

---

## CATEGORY 6 — Testing & Quality

### Q: You have 578 tests — what do they cover?

**Answer:**
Three tiers. Unit tests verify individual components in isolation: does the TemplateMiner produce the correct template? Does the AlertManager suppress within cooldown? Does the IsolationForest return sensible scores? Integration tests hit the actual FastAPI endpoints via TestClient: does POST /ingest return 200 with the correct schema? Does auth middleware reject invalid keys? System tests run end-to-end: simulate a realistic log stream, verify alerts fire at the right times, verify Prometheus counters increment correctly.

---

### Q: How do you test time-based logic like cooldown without waiting real seconds?

**Answer:**
`AlertManager` accepts a `clock_fn` parameter — a callable that returns the current time. In tests, we inject a mock clock: `clock_fn = lambda: current_time`. We advance `current_time` in test code to simulate elapsed time without sleeping. This is a standard dependency injection pattern for testing time-sensitive logic.

---

## CATEGORY 7 — Deployment & Operations

### Q: How do you deploy this in production?

**Answer:**
```bash
docker compose -f docker/docker-compose.yml up -d
```
This starts three containers: the FastAPI API on port 8000, Prometheus on 9090, and Grafana on 3000. Models and artifacts are mounted as volumes from the host. The Dockerfile uses a multi-stage build — builder stage installs dependencies, runtime stage is lean. In production, you'd add a reverse proxy (nginx), TLS termination, and a secrets manager for the API key.

---

### Q: How do you know the system is healthy in production?

**Answer:**
Three signals. First, `/health` returns 200 with `{"status": "healthy"}` — load balancers probe this. Second, `service_health` Prometheus gauge is 1.0 (healthy), 0.5 (degraded), 0.0 (unhealthy). Third, Grafana dashboards show alert rates, latency histograms, and error rates. If `ingest_errors_total` starts climbing, or latency spikes above the P99 threshold, that's the signal to investigate.

---

### Q: How long does it take to start up?

**Answer:**
With models pre-loaded (artifacts present): approximately 2–5 seconds for FastAPI + model loading. If `SEMANTIC_ENABLED=true`, add 5–15 seconds for Sentence Transformers model download (first run only; cached on disk thereafter). The Dockerfile pre-downloads the HF model into the image at build time to avoid runtime download delays.
