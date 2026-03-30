# src/modeling/baseline/model.py

# Purpose: This module defines the `BaselineAnomalyModel` class, which is a thin wrapper around the 
# `IsolationForest` algorithm from scikit-learn for anomaly detection. 
# The class provides methods for fitting the model to a feature matrix,
# scoring new data points to obtain anomaly scores, and making predictions based on a threshold.

# Input: - The `fit` method takes a 2-D numpy array `X` of shape (n_samples, n_features) 
#          containing the feature matrix for training.
#        - The `score` method takes a 2-D numpy array `X` of shape (n_samples, n_features) 
#          and returns a 1-D numpy array of anomaly scores, 
#          where higher values indicate more anomalous instances.
#        - The `predict` method takes a 2-D numpy array `X` and a float `threshold`, 
#          and returns a 1-D numpy array of integer predictions (1 for anomaly, 0 for normal) 
#          based on whether the anomaly scores exceed the threshold.

# Output: - The `fit` method returns the `BaselineAnomalyModel` instance after fitting the model.
#         - The `score` method returns a 1-D numpy array of anomaly scores.

# Used by: Other modules in the project that require anomaly detection using the baseline model,
# such as training scripts that need to fit the model on training data, 
# or evaluation scripts that need to score or predict anomalies.

"""Stage 4A — Baseline: IsolationForest anomaly model wrapper."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


class BaselineAnomalyModel:
    """
    Thin wrapper around sklearn IsolationForest for anomaly detection.

    Anomaly score convention: higher = more anomalous.
    (IsolationForest's score_samples() returns lower values for anomalies;
    we negate so the caller always uses higher-is-worse semantics.)

    Parameters
    ----------
    n_estimators : number of trees (default 300)
    random_state : reproducibility seed (default 42)
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self._model = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "BaselineAnomalyModel":
        """Fit the IsolationForest on feature matrix X."""
        self._model.fit(X)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores (float32 array, shape [n]).
        Higher = more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")
        return (-self._model.score_samples(X)).astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return int8 predictions (1=anomaly) using given threshold."""
        scores = self.score(X)
        return (scores >= threshold).astype(np.int8)

    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, path: Path | str) -> "BaselineAnomalyModel":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj._model = pickle.load(f)
        obj._fitted = True
        return obj
