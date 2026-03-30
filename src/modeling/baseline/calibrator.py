# src/modeling/baseline/calibrator.py

# Purpose: This module defines the `ThresholdCalibrator` class, 
# which is responsible for finding the optimal decision threshold that maximizes 
# the F1 score on a validation set of anomaly scores and corresponding labels. 
# The class provides methods for fitting the model to the data, 
# making predictions based on the calibrated threshold, 
# and saving/loading the calibrator state to/from a file.

# Input: - `scores`: A 1-D numpy array of anomaly scores, where higher values indicate more anomalous instances.
#        - `labels`: A 1-D numpy array of integer labels, where 0 represents normal instances and 1 represents anomalies.

# Output: - The `fit` method returns the `ThresholdCalibrator` instance with the optimal threshold and best F1 score stored as attributes.
#         - The `predict` method returns a 1-D numpy array of integer predictions (0 for normal, 1 for anomaly) based on the calibrated threshold.
#         - The `save` method writes the calibrator's state (threshold, best F1 score, and number of thresholds) to a JSON file.
#         - The `load` class method reads the calibrator's state from a JSON file and returns a `ThresholdCalibrator` instance with the loaded state.

# Used by: Other modules in the project that require threshold calibration for anomaly detection, 
# such as training scripts that need to calibrate the threshold on a validation set, 
# or evaluation scripts that need to apply the calibrated threshold to test data.

"""Stage 4A — Baseline: threshold calibrator (F1-optimal scan)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score


class ThresholdCalibrator:
    """
    Finds the decision threshold that maximises F1 on a validation set.

    Parameters
    ----------
    n_thresholds : number of candidate thresholds between p1 and p99
                   of the score distribution (default 300)
    """

    def __init__(self, n_thresholds: int = 300):
        self.n_thresholds = n_thresholds
        self.threshold_: Optional[float] = None
        self.best_f1_: Optional[float] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> "ThresholdCalibrator":
        """
        Scan thresholds on (scores, labels) and store the best.

        Parameters
        ----------
        scores : 1-D float array of anomaly scores (higher = more anomalous)
        labels : 1-D int array (0 = normal, 1 = anomaly)
        """
        lo = float(np.percentile(scores, 1))
        hi = float(np.percentile(scores, 99))
        candidates = np.linspace(lo, hi, self.n_thresholds)

        best_t, best_f1 = float(candidates[0]), -1.0
        for t in candidates:
            pred = (scores >= t).astype(np.int8)
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.threshold_ = best_t
        self.best_f1_ = best_f1
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call fit() before predict().")
        return (scores >= self.threshold_).astype(np.int8)

    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        payload = {
            "threshold": self.threshold_,
            "best_f1": self.best_f1_,
            "n_thresholds": self.n_thresholds,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "ThresholdCalibrator":
        payload = json.loads(Path(path).read_text())
        obj = cls(n_thresholds=payload["n_thresholds"])
        obj.threshold_ = payload["threshold"]
        obj.best_f1_ = payload["best_f1"]
        return obj
