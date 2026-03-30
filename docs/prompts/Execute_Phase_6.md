# Execute Phase 6 only.

Goal:
create the initial V3 semantic scaffolding without integrating it into the runtime yet.

Tasks:

1. Create new package:

   * `src/semantic/`
2. Add:

   * `src/semantic/__init__.py`
   * `src/semantic/config.py`
   * `src/semantic/loader.py`
   * `src/semantic/embeddings.py`
3. Add configuration support for:

   * `SEMANTIC_ENABLED=false`
   * `SEMANTIC_MODEL=all-MiniLM-L6-v2`
   * `EXPLANATION_ENABLED=false`
   * `EXPLANATION_MODEL=rule-based`
   * `SEMANTIC_CACHE_SIZE=1000`
4. Add dependency:

   * `sentence-transformers>=2.7.0`
5. Update `.env.example`
6. Add mocked unit tests for semantic embeddings
7. Ensure the entire semantic layer is inert when `SEMANTIC_ENABLED=false`
8. Run tests after implementation

Deliverables:

* create:

  * `docs/PHASE_6_V3_SEMANTIC_SCAFFOLD_REPORT.md`
* include:

  * files created
  * config variables added
  * proof that the layer is disabled by default
  * test results

Important:

* do not connect this layer to pipeline/API yet
* no route changes yet
* no alert schema changes yet
