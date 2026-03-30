# Run one final targeted validation pass on the current V3 repository state.

Tasks:

1. run the relevant final tests
2. verify Docker build succeeds
3. verify the compose stack starts correctly
4. check:

   * `/health`
   * `/metrics`
   * `/models/info`
   * `/alerts/{alert_id}/explanation` if implemented
5. run one normal ingest flow
6. run one anomaly/enriched flow and verify the semantic/explanation fields appear correctly
7. return a short summary of results only

Important:

* validation only
* no refactor
* no cleanup
* no new features
* no docs updates
