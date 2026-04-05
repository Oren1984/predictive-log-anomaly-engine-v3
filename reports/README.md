# Reports

This folder stores generated evaluation and runtime evidence outputs.

Current policy:
- Generated report files may be rotated out of this folder during cleanup.
- Historical snapshots are archived under `archive/generated/reports/<date>/`.
- Runtime code should not rely on pre-existing files here.

Common generated files include:
- `evaluation_report_v2.json`
- `runtime_calibration_scores.csv`
- `runtime_demo_results.csv`
- `runtime_demo_evidence.jsonl`
