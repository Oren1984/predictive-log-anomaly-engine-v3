# Execute Phase 0 only.

Goal:
establish a safe baseline before any repository changes.

Tasks:

1. Run the current validation baseline:

   * `pytest -m "not slow"`
   * Docker build using `docker/Dockerfile`
   * bring up the compose stack and verify `/health`
2. Document:

   * current test result summary
   * Docker build result
   * compose/health result
   * current files present under `models/`
3. Verify whether `src/engine/proactive_engine.py` has any active imports anywhere in `src/`
4. Do not modify code yet
5. Produce a structured report file:

   * `docs/PHASE_0_BASELINE_PROTECTION_REPORT.md`

Important:

* no cleanup yet
* no refactor yet
* no file moves yet
* report only
