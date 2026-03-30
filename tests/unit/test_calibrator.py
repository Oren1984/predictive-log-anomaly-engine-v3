# test/unit/test_calibrator.py

# Purpose: Unit tests for the ThresholdCalibrator class to verify fitting, prediction, and persistence.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.
 
# Used by: N/A (these are unit tests for the ThresholdCalibrator class)

"""Unit tests for ThresholdCalibrator."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.modeling.baseline import ThresholdCalibrator


@pytest.fixture()
def perfect_scores():
    """Scores that perfectly separate 0/1 labels."""
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    return scores, labels


@pytest.fixture()
def noisy_scores():
    """Scores with some overlap between classes."""
    rng = np.random.default_rng(42)
    normal  = rng.normal(loc=0.3, scale=0.1, size=100)
    anomaly = rng.normal(loc=0.7, scale=0.15, size=20)
    scores  = np.concatenate([normal, anomaly]).astype(np.float32)
    labels  = np.concatenate([np.zeros(100), np.ones(20)]).astype(np.int8)
    return scores, labels


class TestFit:
    def test_threshold_set_after_fit(self, perfect_scores):
        scores, labels = perfect_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)
        assert cal.threshold_ is not None

    def test_best_f1_set_after_fit(self, perfect_scores):
        scores, labels = perfect_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)
        assert cal.best_f1_ is not None

    def test_perfect_f1_on_perfect_scores(self, perfect_scores):
        scores, labels = perfect_scores
        cal = ThresholdCalibrator(n_thresholds=500)
        cal.fit(scores, labels)
        assert cal.best_f1_ == pytest.approx(1.0, abs=0.01)

    def test_threshold_in_score_range(self, noisy_scores):
        scores, labels = noisy_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)
        lo = float(np.percentile(scores, 1))
        hi = float(np.percentile(scores, 99))
        assert lo <= cal.threshold_ <= hi

    def test_noisy_f1_positive(self, noisy_scores):
        scores, labels = noisy_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)
        assert cal.best_f1_ > 0.0


class TestPredict:
    def test_predict_before_fit_raises(self):
        cal = ThresholdCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict(np.array([0.5]))

    def test_predict_shape(self, perfect_scores):
        scores, labels = perfect_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)
        preds = cal.predict(scores)
        assert preds.shape == scores.shape

    def test_predict_dtype(self, perfect_scores):
        scores, labels = perfect_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)
        preds = cal.predict(scores)
        assert preds.dtype == np.int8

    def test_predict_values_binary(self, noisy_scores):
        scores, labels = noisy_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)
        preds = cal.predict(scores)
        assert set(preds.tolist()).issubset({0, 1})


class TestPersistence:
    def test_save_and_load(self, perfect_scores):
        scores, labels = perfect_scores
        cal = ThresholdCalibrator(n_thresholds=100)
        cal.fit(scores, labels)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            cal.save(tmp_path)
            cal2 = ThresholdCalibrator.load(tmp_path)
            assert cal2.threshold_ == pytest.approx(cal.threshold_)
            assert cal2.best_f1_ == pytest.approx(cal.best_f1_)
            assert cal2.n_thresholds == cal.n_thresholds
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_predict_after_load(self, perfect_scores):
        scores, labels = perfect_scores
        cal = ThresholdCalibrator()
        cal.fit(scores, labels)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            cal.save(tmp_path)
            cal2 = ThresholdCalibrator.load(tmp_path)
            preds = cal2.predict(scores)
            assert preds.shape == scores.shape
        finally:
            Path(tmp_path).unlink(missing_ok=True)
