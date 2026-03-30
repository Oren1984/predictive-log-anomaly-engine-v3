# events_unified.csv — Column Schema

**File:** `data/processed/events_unified.csv`
**Rows:** 15,923,592
**Source datasets:** HDFS_1, HDFS_2, BGL

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float64 (nullable) | Unix timestamp (BGL only); NaN for HDFS rows |
| `dataset` | str | Source dataset: `"hdfs"` or `"bgl"` |
| `session_id` | str | HDFS: block ID (e.g. `blk_-123…`); BGL: window ID (e.g. `window_0`) |
| `message` | str | Raw log message text |
| `label` | int8 | Anomaly label — `0` = normal, `1` = anomalous |

## Label Distribution

| label | dataset | rows | pct |
|-------|---------|-----:|----:|
| 0 | hdfs | 10,887,379 | 68.37% |
| 1 | bgl  | 4,399,503  | 27.63% |
| 0 | bgl  | 348,460    |  2.19% |
| 1 | hdfs | 288,250    |  1.81% |

## Session Definition

- **HDFS** — sessions are grouped by block ID; each `blk_*` string may span many log lines across distributed nodes.
- **BGL** — sessions use a fixed sliding window of 20 lines (stride 1); each window is treated as one session.
- Labels are session-level: one anomalous event labels the entire session/window anomalous (`label = max(event_labels)`).

## Downstream Artifacts

| File | Description |
|------|-------------|
| `data/intermediate/events_with_templates.csv` | events_unified + `template_id`, `template_text` columns (1M sample) |
| `data/intermediate/templates.csv` | 7,833 unique templates with count and anomaly_rate |
| `data/intermediate/session_sequences_v2.csv` | 495,405 sessions with ordered template sequence string |
| `data/intermediate/session_features_v2.csv` | 495,405 sessions × 407 numeric features |
