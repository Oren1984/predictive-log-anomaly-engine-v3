# Execute Phase 4 only.

Goal:
resolve verified duplication and isolate verified legacy components, but only where Phase 1 provided evidence.

Rules:

* do not guess
* only act on items that were proven in Phase 1
* after every sub-step, run tests or the relevant validation

Tasks:

1. For each verified wrapper/duplicate among:

   * `src/modeling/behavior_model.py`
   * `src/modeling/anomaly_detector.py`
   * `src/modeling/severity_classifier.py`
     decide the canonical implementation and update imports accordingly
2. If `src/data/` is confirmed duplicate to `src/synthetic/`:

   * merge any unique logic into `src/synthetic/`
   * then remove or archive the obsolete duplicate files
3. Move verified legacy engine files:

   * `src/engine/proactive_engine.py`
   * `src/engine/__init__.py`
     into `archive/src/engine/`
4. Handle these tests based on verified dependency status:

   * `tests/unit/test_proactive_engine.py`
   * `tests/unit/test_explain_decode.py`
     Either update, archive, or remove them only if justified
5. Run full tests at the end

Deliverables:

* create:

  * `docs/PHASE_4_DUPLICATION_AND_LEGACY_RESOLUTION_REPORT.md`
* include:

  * each decision
  * why it was safe
  * what imports were changed
  * what tests were affected
  * full validation result

Important:

* do not introduce new V3 modules yet
* no Hugging Face yet
