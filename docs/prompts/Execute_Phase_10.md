# You are working inside the repository:

`C:\Users\ORENS\predictive-log-anomaly-engine-v3`

Target notebook directory:

`C:\Users\ORENS\predictive-log-anomaly-engine-v3\notebooks`

Target notebooks:

1. `predictive_log_anomaly_engine_demo.ipynb`
2. `predictive_log_anomaly_engine_gpu_demo.ipynb`

Your task right now is to perform a deep audit of both notebooks against the actual current repository state and prepare a structured update plan.

Important:

* Do NOT update the notebooks yet
* Do NOT rewrite cells yet
* Do NOT fabricate files, paths, endpoints, models, or features
* Base everything only on the real repository state

What to do:

1. Scan the repository as needed to understand the current system architecture, runtime flow, endpoints, configs, artifacts, observability stack, and model flow
2. Read both notebooks completely:

   * markdown cells
   * code cells
   * imports
   * paths
   * outputs if relevant
   * endpoint examples
   * model references
   * artifact names
   * metrics references
   * Docker/runtime assumptions
3. Compare each notebook against the current repository reality
4. Identify all mismatches, including:

   * broken or outdated imports
   * wrong paths
   * obsolete file names
   * outdated architecture descriptions
   * old pipeline explanations
   * stale metric names
   * outdated model references
   * stale endpoint usage
   * inconsistent demo flow
   * misleading markdown
   * duplicated or unnecessary cells
   * sections that are still valid and should remain unchanged

Create this report file:
`docs/NOTEBOOKS_V3_AUDIT_REPORT.md`

The report must include:

* Executive summary
* Notebook-by-notebook analysis
* Cell-level issue categories
* What is outdated vs what is still valid
* Must-fix changes
* Nice-to-have improvements
* Manual verification items
* Risk notes
* Recommended update order
* Which notebook should be updated first and why

Also include a proposed phased notebook update plan aligned to:

* Phase 0
* Phase 1
* Phase 2
* Phase 3
* STOP for manual review
* Phase 4
* Phase 5
* STOP for manual review
* Phase 6
* Phase 7
* Phase 8

Do not start editing yet.
First complete the audit and write the report.
