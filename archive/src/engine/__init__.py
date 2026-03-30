# src/engine/__init__.py
#
# LEGACY — Phase 7 development orchestrator
#
# ProactiveMonitorEngine was developed as a standalone AI pipeline integrating
# LogPreprocessor -> BehaviorModel -> AnomalyDetector -> SeverityClassifier.
# It is NOT connected to the production API path.
#
# Production inference path:
#   src/runtime/inference_engine.py  (InferenceEngine)
#   src/api/pipeline.py              (Pipeline container)
#
# This module is retained for test coverage and as a reference architecture.
# See tests/unit/test_proactive_engine.py for usage.

from .proactive_engine import EngineResult, ProactiveMonitorEngine

__all__ = ["ProactiveMonitorEngine", "EngineResult"]
