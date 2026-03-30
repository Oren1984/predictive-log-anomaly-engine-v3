# Execute Phase 5 only.

Goal:
finalize repository hygiene before V3 feature work begins.

Tasks:

1. Verify `.gitignore` coverage for:

   * `models/*.pkl`
   * `models/*.pt`
   * `models/**/*.model`
   * generated report outputs
   * generated alert JSON files
2. If `ai_workspace/` is confirmed to be IDE or assistant workspace noise:

   * add it to `.gitignore`
   * optionally remove it from the repo if safe
3. Verify large data paths are excluded appropriately, including:

   * `data/raw/`
   * `data/processed/`
4. Create:

   * `docs/PHASE_5_REPOSITORY_HYGIENE_REPORT.md`

Important:

* no semantic/V3 code yet
* no API changes yet
* hygiene only
