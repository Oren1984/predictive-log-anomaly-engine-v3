# Execute Phase 2 only.

Goal:
perform safe cleanup actions that do not change runtime imports or architecture.

Approved tasks:

1. Move `evaluation_report.json` from repo root to:

   * `reports/evaluation_report_v2.json`
2. Create `reports/` if missing
3. Add `reports/*.json` to `.gitignore`
4. Delete generated JSON contents inside:

   * `artifacts/n8n_outbox/`
     while preserving the directory itself with `.gitkeep`
5. Add `artifacts/n8n_outbox/*.json` to `.gitignore`
6. Create archive folders only:

   * `archive/src/engine/`
   * `archive/src/data/`
   * `archive/scripts/`

Deliverables:

* apply only these changes
* create:

  * `docs/PHASE_2_SAFE_CLEANUP_REPORT.md`
* include exact files changed
* include confirmation that no runtime imports were changed

Important:

* do not archive code yet
* do not delete `src/data/`
* do not move scripts yet
* do not touch tests yet
