# Execute Phase 3 only.

Goal:
clean up script naming and layout without changing core runtime architecture.

Approved tasks:

1. Rename:

   * `scripts/stage_07_run_api.py` → `scripts/run_api.py`
2. Update all references accordingly, including:

   * `main.py`
   * Docker entry/CMD if needed
   * any internal references
3. Create:

   * `scripts/data_pipeline/`
4. Move only offline pipeline scripts into that folder:

   * all `stage_01*.py`
   * all `stage_02*.py`
   * all `stage_03*.py`
   * all `stage_04*.py`
   * all `stage_05*.py`
   * all `stage_06*.py`
5. Move contents of `scripts/archive/` into:

   * `archive/scripts/`
6. Run tests after changes

Deliverables:

* apply only this phase
* create:

  * `docs/PHASE_3_SCRIPT_RENAME_AND_REORG_REPORT.md`
* include before/after mapping of file moves
* include test results after the changes

Important:

* do not modify runtime logic
* do not add Hugging Face yet
* do not touch `src/data/` yet
