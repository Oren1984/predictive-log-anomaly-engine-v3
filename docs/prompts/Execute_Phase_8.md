# Execute Phase 8 only.

Goal:
complete the V3 surface area around the semantic layer.

Tasks:

1. Add practical API endpoints:

   * `GET /alerts/{alert_id}/explanation`
   * `GET /models/info`
   * optional `POST /v3/ingest` only if consistent with current structure
2. Update health reporting to include semantic layer readiness
3. Add observability metrics for V3 semantic processing
4. Update Grafana dashboard with the new metrics
5. Update Docker build and dependency handling for Hugging Face models
6. Update README and create V3 architecture documentation
7. Add or update demo notebook for V3

Deliverables:

* create:

  * `docs/PHASE_8_API_OBSERVABILITY_AND_DOCS_REPORT.md`
* include:

  * routes added
  * metrics added
  * Docker impact summary
  * documentation files created or updated
  * final validation results

Important:

* keep everything additive
* preserve existing CI behavior
* prefer CPU-friendly defaults
