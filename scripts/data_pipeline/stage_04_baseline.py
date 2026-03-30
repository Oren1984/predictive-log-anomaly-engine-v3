# scripts/stage_04_baseline.py

# Purpose: Train a baseline IsolationForest anomaly detection model on the sequence data, 
# calibrate a threshold, and evaluate performance.

# Input: Reads data/processed/sequences_train.parquet, sequences_val.parquet, 
# and sequences_test.parquet, which contain sequence_id, tokens (JSON string), and label columns.

# Output: Writes models/baseline.pkl (the trained IsolationForest model), 
# artifacts/threshold.json (the calibrated threshold and related info), 
# and reports/stage_04_baseline.md (a markdown report summarizing the results).

# Used by: This script is used by the main pipeline to train and evaluate a baseline anomaly detection model. 
# The generated model and threshold are used for inference in later stages, 
# and the report provides a summary of the baseline performance. 
# It can be run independently to retrain the baseline model if the sequence data is updated 
# or if you want to switch between demo and full modes.

"""
Stage 04A — Baseline: train IsolationForest anomaly detector.

Reads  : data/processed/sequences_train/val/test.parquet
Writes : models/baseline.pkl
         artifacts/threshold.json
         reports/stage_04_baseline.md

Usage:
    python scripts/stage_04_baseline.py
    python scripts/stage_04_baseline.py --mode demo
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sequencing import Sequence
from src.modeling.baseline import (
    BaselineFeatureExtractor, BaselineAnomalyModel, ThresholdCalibrator
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = ROOT / "data" / "processed"
MODEL_OUT = ROOT / "models" / "baseline.pkl"
THRESH_OUT = ROOT / "artifacts" / "threshold.json"
REPORT_OUT = ROOT / "reports" / "stage_04_baseline.md"


def load_sequences(path: Path) -> list[Sequence]:
    df = pd.read_parquet(path)
    seqs = []
    for row in df.itertuples(index=False):
        tokens = json.loads(row.tokens)
        seqs.append(Sequence(
            sequence_id=str(row.sequence_id),
            tokens=tokens,
            label=int(row.label) if row.label is not None else None,
        ))
    return seqs


def main(mode: str) -> None:
    log.info("Stage 04A (Baseline) | mode=%s", mode)
    t0 = time.time()

    train_seqs = load_sequences(DATA_DIR / "sequences_train.parquet")
    val_seqs   = load_sequences(DATA_DIR / "sequences_val.parquet")
    test_seqs  = load_sequences(DATA_DIR / "sequences_test.parquet")
    log.info("Sequences: train=%d  val=%d  test=%d",
             len(train_seqs), len(val_seqs), len(test_seqs))

    # Feature extraction
    extractor = BaselineFeatureExtractor(top_k=100)
    X_train = extractor.fit_transform(train_seqs)
    X_val   = extractor.transform(val_seqs)
    X_test  = extractor.transform(test_seqs)
    log.info("Features: %d dims", extractor.n_features)

    # Train model
    model = BaselineAnomalyModel(n_estimators=300, random_state=42)
    model.fit(X_train)
    log.info("IsolationForest fitted")

    # Calibrate threshold on val
    val_scores = model.score(X_val)
    val_labels = np.array([s.label for s in val_seqs], dtype=np.int8)
    cal = ThresholdCalibrator(n_thresholds=300)
    cal.fit(val_scores, val_labels)
    log.info("Threshold=%.5f  val_F1=%.4f", cal.threshold_, cal.best_f1_)

    # Evaluate on test
    test_scores = model.score(X_test)
    test_labels = np.array([s.label for s in test_seqs], dtype=np.int8)
    test_preds  = cal.predict(test_scores)

    from sklearn.metrics import (classification_report, roc_auc_score,
                                  average_precision_score)
    roc  = roc_auc_score(test_labels, test_scores)
    prauc = average_precision_score(test_labels, test_scores)
    report_str = classification_report(test_labels, test_preds,
                                       target_names=["normal", "anomaly"])
    log.info("Test  ROC-AUC=%.4f  PR-AUC=%.4f", roc, prauc)
    log.info("\n%s", report_str)

    # Save artifacts
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    cal.save(THRESH_OUT)

    elapsed = time.time() - t0
    md = f"""# Stage 04A — Baseline Report

**Mode**: {mode}
**Elapsed**: {elapsed:.1f}s

## Dataset
| Split | N |
|-------|---|
| train | {len(train_seqs)} |
| val   | {len(val_seqs)} |
| test  | {len(test_seqs)} |

## Features
- Dimensions: {extractor.n_features}
- Top-K: 100 template frequency features

## Threshold Calibration (val set)
- Threshold: {cal.threshold_:.5f}
- Val F1: {cal.best_f1_:.4f}

## Test Metrics
- ROC-AUC: {roc:.4f}
- PR-AUC: {prauc:.4f}

```
{report_str}
```

## Artifacts
- `models/baseline.pkl`
- `artifacts/threshold.json`
"""
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(md)
    log.info("Report written to %s", REPORT_OUT)
    log.info("Stage 04A complete in %.1fs", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "demo"])
    args = parser.parse_args()
    main(args.mode)
