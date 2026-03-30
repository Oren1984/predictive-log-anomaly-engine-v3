# test/unit/test_runtime_calibration.py

# Purpose: Unit tests for the runtime calibration script to verify that it produces expected artifacts and outputs.

# Input: None (test code only)

# Output: Test results (pass/fail) when run with pytest.

# Used by: N/A (these are unit tests for the runtime calibration script, indirectly used by the script itself and its outputs)


"""Unit tests for Stage 31 runtime calibration script."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

_PARQUET = ROOT / "data" / "processed" / "events_tokenized.parquet"
_BASELINE_PKL = ROOT / "models" / "baseline.pkl"
_TRANSFORMER_PT = ROOT / "models" / "transformer.pt"
_ARTIFACTS_OK = (
    _PARQUET.exists()
    and _BASELINE_PKL.exists()
    and (ROOT / "artifacts" / "vocab.json").exists()
    and (ROOT / "artifacts" / "threshold.json").exists()
)
_TRANSFORMER_OK = _ARTIFACTS_OK and _TRANSFORMER_PT.exists()

needs_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_OK,
    reason="Required artifacts/data not found; skipping calibration tests",
)
needs_transformer = pytest.mark.skipif(
    not _TRANSFORMER_OK,
    reason="Transformer model not found; skipping full calibration test",
)

# ---------------------------------------------------------------------------
# Expected output paths
# ---------------------------------------------------------------------------

THRESHOLD_RUNTIME_PATH = ROOT / "artifacts" / "threshold_runtime.json"
SCORES_CSV_PATH        = ROOT / "reports"   / "runtime_calibration_scores.csv"
REPORT_MD_PATH         = ROOT / "reports"   / "stage_31_runtime_calibration_report.md"
LOG_PATH               = ROOT / "ai_workspace" / "logs" / "stage_05_runtime_calibrate.log"


# ---------------------------------------------------------------------------
# Helper: import run_calibration lazily
# ---------------------------------------------------------------------------

def _import_calibrate():
    import importlib
    spec = importlib.util.spec_from_file_location(
        "stage_05_runtime_calibrate",
        ROOT / "scripts" / "stage_05_runtime_calibrate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_artifacts
@needs_transformer
class TestRuntimeCalibration:
    """Run calibration on a tiny sample and verify outputs."""

    N_EVENTS = 2000

    def test_calibration_creates_threshold_artifact(self, tmp_path):
        """threshold_runtime.json must be created and contain expected keys."""
        mod = _import_calibrate()

        result = mod.run_calibration(
            mode="demo",
            model="ensemble",
            n_events=self.N_EVENTS,
            key_by="service",
            window_size=50,
            stride=10,
            target_alert_rate=0.005,
        )

        # Artifact file must exist
        assert THRESHOLD_RUNTIME_PATH.exists(), (
            f"threshold_runtime.json not found at {THRESHOLD_RUNTIME_PATH}"
        )

        # Load and validate schema
        with open(THRESHOLD_RUNTIME_PATH, encoding="utf-8") as fh:
            data = json.load(fh)

        assert "thresholds" in data, "Missing 'thresholds' key"
        assert "score_stats" in data, "Missing 'score_stats' key"
        assert "method" in data, "Missing 'method' key"
        assert data["method"] in ("f1", "percentile"), (
            f"Invalid method: {data['method']!r}"
        )

    def test_thresholds_are_numeric(self):
        """All three model thresholds must be finite floats."""
        assert THRESHOLD_RUNTIME_PATH.exists(), \
            "Run test_calibration_creates_threshold_artifact first"

        with open(THRESHOLD_RUNTIME_PATH, encoding="utf-8") as fh:
            data = json.load(fh)

        thresholds = data["thresholds"]
        for model in ("baseline", "transformer", "ensemble"):
            assert model in thresholds, f"Missing threshold for model={model!r}"
            val = thresholds[model]
            assert isinstance(val, (int, float)), (
                f"threshold[{model!r}] is not numeric: {val!r}"
            )
            assert val == val, f"threshold[{model!r}] is NaN"  # NaN != NaN
            assert float("-inf") < val < float("inf"), (
                f"threshold[{model!r}] is not finite: {val}"
            )

    def test_score_stats_present(self):
        """score_stats must contain min, p50, p95, p99, max for all models."""
        assert THRESHOLD_RUNTIME_PATH.exists(), \
            "Run test_calibration_creates_threshold_artifact first"

        with open(THRESHOLD_RUNTIME_PATH, encoding="utf-8") as fh:
            data = json.load(fh)

        stats = data["score_stats"]
        expected_keys = {"min", "p50", "p95", "p99", "max"}
        for model in ("baseline", "transformer", "ensemble"):
            assert model in stats, f"score_stats missing model={model!r}"
            model_stats = stats[model]
            missing = expected_keys - set(model_stats.keys())
            assert not missing, (
                f"score_stats[{model!r}] missing keys: {missing}"
            )
            # All stat values should be numeric
            for k in expected_keys:
                v = model_stats[k]
                assert isinstance(v, (int, float)), (
                    f"score_stats[{model!r}][{k!r}] not numeric: {v!r}"
                )

    def test_scores_csv_created(self):
        """reports/runtime_calibration_scores.csv must be created with data."""
        assert SCORES_CSV_PATH.exists(), (
            f"Scores CSV not found at {SCORES_CSV_PATH}"
        )
        import pandas as pd
        df = pd.read_csv(SCORES_CSV_PATH)
        assert len(df) > 0, "Scores CSV is empty"
        required_cols = {"timestamp", "stream_key", "model", "risk_score"}
        missing = required_cols - set(df.columns)
        assert not missing, f"Scores CSV missing columns: {missing}"

    def test_report_md_created(self):
        """reports/stage_31_runtime_calibration_report.md must be created."""
        assert REPORT_MD_PATH.exists(), (
            f"Report not found at {REPORT_MD_PATH}"
        )
        content = REPORT_MD_PATH.read_text(encoding="utf-8")
        assert "demo-calibrated threshold" in content.lower() or \
               "demo-calibrated" in content, \
               "Report must contain demo-calibrated disclaimer"
        assert "threshold" in content.lower(), "Report must mention threshold"

    def test_log_file_created(self):
        """ai_workspace/logs/stage_05_runtime_calibrate.log must be created."""
        assert LOG_PATH.exists(), f"Log file not found at {LOG_PATH}"
        assert LOG_PATH.stat().st_size > 0, "Log file is empty"

    def test_n_windows_positive(self):
        """n_windows in the artifact must be > 0 (stream produced windows)."""
        assert THRESHOLD_RUNTIME_PATH.exists()
        with open(THRESHOLD_RUNTIME_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data.get("n_windows", 0) > 0, "No windows were emitted during calibration"
