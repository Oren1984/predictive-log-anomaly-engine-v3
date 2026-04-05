# scripts/data_pipeline/stage_04_transformer.py

# Purpose: Train a next-token GPT-style transformer model on the sequence data 
# and evaluate its performance for anomaly detection.

# Input: Reads data/processed/sequences_train.parquet, sequences_val.parquet, 
# and sequences_test.parquet, which contain sequence_id, 
# tokens (JSON string), and label columns.

# Output: Writes models/transformer.pt (the trained transformer model), 
# artifacts/threshold_transformer.json (the calibrated threshold and related info), 
# reports/stage_04_transformer.md (a markdown report summarizing the results), 
# and reports/metrics_transformer.json (a JSON file with detailed metrics).

# Used by: This script is used by the main pipeline to train and evaluate a transformer-based anomaly detection model. 
# The generated model and threshold are used for inference in later stages, 
# and the report provides a summary of the transformer's performance.

"""
Stage 04B — Transformer: train next-token GPT-style model and score sequences.

Reads  : data/processed/sequences_train/val/test.parquet
Writes : models/transformer.pt
         artifacts/threshold_transformer.json
         reports/stage_04_transformer.md
         reports/metrics_transformer.json

Usage:
    python scripts/data_pipeline/stage_04_transformer.py
    python scripts/data_pipeline/stage_04_transformer.py --mode demo   (2 epochs, small cfg)
    python scripts/data_pipeline/stage_04_transformer.py --device cuda
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.sequencing import Sequence
from src.modeling.transformer import (
    TransformerConfig, NextTokenTransformerModel, Trainer, AnomalyScorer
)
from src.modeling.baseline import ThresholdCalibrator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = ROOT / "data" / "processed"
MODEL_OUT  = ROOT / "models" / "transformer.pt"
THRESH_OUT = ROOT / "artifacts" / "threshold_transformer.json"
REPORT_OUT = ROOT / "reports" / "stage_04_transformer.md"
METRICS_OUT = ROOT / "reports" / "metrics_transformer.json"

TEMPLATES_CSV = ROOT / "data" / "intermediate" / "templates.csv"


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


def get_vocab_size() -> int:
    import pandas as pd
    df = pd.read_csv(TEMPLATES_CSV, usecols=["template_id"])
    return int(df["template_id"].max()) + 3   # PAD + UNK + offset


def main(mode: str, device: str) -> None:
    log.info("Stage 04B (Transformer) | mode=%s | device=%s", mode, device)
    t0 = time.time()

    train_seqs = load_sequences(DATA_DIR / "sequences_train.parquet")
    val_seqs   = load_sequences(DATA_DIR / "sequences_val.parquet")
    test_seqs  = load_sequences(DATA_DIR / "sequences_test.parquet")
    log.info("Sequences: train=%d  val=%d  test=%d",
             len(train_seqs), len(val_seqs), len(test_seqs))

    vocab_size = get_vocab_size()

    if mode == "demo":
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=128, batch_size=32, max_epochs=2,
        )
        train_seqs = train_seqs[:500]
        val_seqs   = val_seqs[:100]
        test_seqs  = test_seqs[:100]
    else:
        cfg = TransformerConfig(vocab_size=vocab_size)

    # Train
    trainer = Trainer(cfg=cfg, device=device)
    log.info("Params: %d", sum(p.numel() for p in trainer.model.parameters()))
    history = trainer.train(train_seqs, val_seqs, save_path=MODEL_OUT)
    log.info("Training complete. Best val loss: %.4f",
             min(history["val_loss"]))

    # Score sequences (use best checkpoint if saved, else current weights)
    model = (NextTokenTransformerModel.load(MODEL_OUT, map_location=device)
             if MODEL_OUT.exists() else trainer.model)
    scorer = AnomalyScorer(model=model, cfg=cfg, device=device)

    val_scores  = scorer.score(val_seqs)
    test_scores = scorer.score(test_seqs)
    val_labels  = np.array([s.label for s in val_seqs],  dtype=np.int8)
    test_labels = np.array([s.label for s in test_seqs], dtype=np.int8)

    # Calibrate threshold on val
    cal = ThresholdCalibrator(n_thresholds=300)
    cal.fit(val_scores, val_labels)
    scorer.set_threshold(cal.threshold_)
    log.info("Threshold=%.5f  val_F1=%.4f", cal.threshold_, cal.best_f1_)

    # Evaluate on test
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  average_precision_score)
    roc   = roc_auc_score(test_labels, test_scores)
    prauc = average_precision_score(test_labels, test_scores)
    test_preds  = scorer.predict(test_scores)
    report_str  = classification_report(test_labels, test_preds,
                                        target_names=["normal", "anomaly"])
    log.info("Test  ROC-AUC=%.4f  PR-AUC=%.4f", roc, prauc)
    log.info("\n%s", report_str)

    # Save artifacts
    scorer.save_threshold(THRESH_OUT)
    metrics = {
        "mode": mode,
        "roc_auc": round(roc, 4),
        "pr_auc":  round(prauc, 4),
        "threshold": cal.threshold_,
        "val_f1": round(cal.best_f1_, 4),
        "history": history,
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))

    elapsed = time.time() - t0
    md = f"""# Stage 04B — Transformer Report

**Mode**: {mode}  **Device**: {device}
**Elapsed**: {elapsed:.1f}s

## Architecture
| Param | Value |
|-------|-------|
| vocab_size | {cfg.vocab_size} |
| d_model | {cfg.d_model} |
| n_heads | {cfg.n_heads} |
| n_layers | {cfg.n_layers} |
| max_seq_len | {cfg.max_seq_len} |
| batch_size | {cfg.batch_size} |
| max_epochs | {cfg.max_epochs} |

## Training History
| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
{"".join(f"| {i+1} | {tl} | {vl} |" + chr(10) for i, (tl, vl) in enumerate(zip(history['train_loss'], history['val_loss'])))}

## Threshold Calibration
- Threshold: {cal.threshold_:.5f}
- Val F1: {cal.best_f1_:.4f}

## Test Metrics
- ROC-AUC: {roc:.4f}
- PR-AUC:  {prauc:.4f}

```
{report_str}
```

## Artifacts
- `models/transformer.pt`
- `artifacts/threshold_transformer.json`
- `reports/metrics_transformer.json`
"""
    REPORT_OUT.write_text(md)
    log.info("Report written to %s", REPORT_OUT)
    log.info("Stage 04B complete in %.1fs", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "demo"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.mode, args.device)
