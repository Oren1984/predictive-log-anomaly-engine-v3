# Execute Phase 1 only.

Goal:
verify ambiguous or potentially duplicated areas before any deletion, archive, or refactor.

Tasks:
Read and analyze these items and determine their exact relationship:

1. `src/modeling/behavior_model.py` vs `src/modeling/behavior/lstm_model.py`
2. `src/modeling/anomaly_detector.py` vs `src/modeling/anomaly/autoencoder.py`
3. `src/modeling/severity_classifier.py` vs `src/modeling/severity/severity_classifier.py`
4. `src/data/` vs `src/synthetic/`
5. `static-demo/`
6. `ai_workspace/`
7. `tests/unit/test_explain_decode.py`
8. `tests/unit/test_proactive_engine.py`

Deliverables:

* classify each item as one of:

  * active canonical
  * wrapper
  * duplicate
  * legacy
  * unclear
* explain what should happen next for each item
* create:

  * `docs/PHASE_1_VERIFICATION_AND_DISAMBIGUATION_REPORT.md`

Important:

* do not move files
* do not delete files
* do not refactor imports yet
* analysis and proof only
