# Execute Phase 7 only.

Goal:
integrate the semantic enrichment layer into the pipeline in an additive, backward-compatible way.

Tasks:

1. Create:

   * `src/semantic/similarity.py`
   * `src/semantic/explainer.py`
2. Integrate semantic enrichment into the pipeline only after anomaly confirmation
3. Extend alert data model with optional fields only:

   * `explanation`
   * `semantic_similarity`
   * `top_similar_events`
   * `evidence_tokens`
4. Update API schemas accordingly with backward compatibility preserved
5. Run integration tests and full tests

Deliverables:

* create:

  * `docs/PHASE_7_PIPELINE_SEMANTIC_INTEGRATION_REPORT.md`
* include:

  * exact files modified
  * compatibility notes
  * test results
  * confirmation that existing flows still work with semantic disabled

Important:

* preserve current V1/V2 behavior
* semantic enrichment must be gated by config
* no breaking schema changes
