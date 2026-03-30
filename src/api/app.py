# src/api/app.py

# Purpose: Implement the main application factory for the API using FastAPI. 
# This includes setting up shared state (like the inference pipeline and metrics registry), 
# configuring middleware (for authentication and metrics), 
# and including the API routes.

# Input: The create_app function is the main application factory for the API.
# It initializes the FastAPI application, sets up shared state 
# (like the inference pipeline and metrics registry), 
# and configures middleware and routes.

# Output: The create_app function returns a configured FastAPI application instance, 
# ready to be run by an ASGI server (e.g., Uvicorn).

# Used by: The create_app function is used in the main entry point of the application 
# (e.g., run.py) to create the FastAPI app instance.

"""Stage 7 — API: FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ..health.checks import HealthChecker
from ..observability.metrics import MetricsMiddleware, MetricsRegistry
from ..security.auth import AuthMiddleware
from .pipeline import Pipeline
from .routes import router
from .routes_v2 import router_v2
from .settings import Settings
from .ui import ui_router

logger = logging.getLogger(__name__)


async def _warmup_task(pipeline, n_events: int, interval_seconds: float) -> None:
    """
    Background task: ingest a small synthetic batch through the pipeline.

    Runs once (interval_seconds <= 0) or periodically.  Each call ingests
    n_events events, then logs a single summary line.
    """
    await asyncio.sleep(2)  # let uvicorn finish binding before we start
    while True:
        total = 0
        alerts = 0
        for i in range(n_events):
            event = {
                "service": "demo",
                "token_id": (abs(hash(f"warmup-{i}")) % 7833) + 2,
                "session_id": "warmup-session",
                "timestamp": 0.0,
                "label": 0,
            }
            try:
                result = pipeline.process_event(event)
                total += 1
                if result.get("alert"):
                    alerts += 1
            except Exception as exc:
                logger.warning("DEMO_WARMUP: event error: %s", exc)
        logger.info("DEMO_WARMUP: ingested %d events, alerts=%d", total, alerts)
        if interval_seconds <= 0:
            break
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load ML models on startup; optionally run warmup traffic; clean up on shutdown."""
    pipeline: Pipeline = app.state.pipeline
    cfg: Settings = app.state.settings
    logger.info("API startup: loading inference pipeline ...")
    try:
        pipeline.load_models()
        logger.info("API startup: pipeline ready")
    except Exception as exc:
        logger.error("API startup: pipeline load failed: %s", exc)
        # Continue anyway so /health can report unhealthy

    # ------------------------------------------------------------------
    # v2 engine — loaded when MODEL_MODE contains "v2" (e.g. "v2" or "both")
    # ------------------------------------------------------------------
    app.state.engine_v2 = None
    if "v2" in cfg.model_mode.lower():
        logger.info("API startup: MODEL_MODE=%r — loading v2 pipeline ...", cfg.model_mode)
        try:
            from ..runtime.inference_engine_v2 import InferenceEngineV2
            from ..runtime.pipeline_v2 import V2PipelineConfig
            v2_cfg = V2PipelineConfig(window_size=cfg.window_size)
            engine_v2 = InferenceEngineV2(
                cfg=v2_cfg,
                alert_buffer_size=cfg.alert_buffer_size,
                alert_cooldown_seconds=cfg.alert_cooldown_seconds,
            )
            engine_v2.load_models()
            app.state.engine_v2 = engine_v2
            logger.info("API startup: v2 pipeline ready")
        except Exception as exc:
            logger.error("API startup: v2 pipeline load failed: %s", exc)
            # Continue — /v2/ingest will return 503 until models are trained

    warmup = None
    if cfg.demo_warmup_enabled:
        logger.info(
            "API startup: scheduling demo warmup (%d events, interval=%.0fs)",
            cfg.demo_warmup_events,
            cfg.demo_warmup_interval_seconds,
        )
        warmup = asyncio.create_task(
            _warmup_task(
                pipeline,
                cfg.demo_warmup_events,
                cfg.demo_warmup_interval_seconds,
            )
        )

    yield

    if warmup and not warmup.done():
        warmup.cancel()
    logger.info("API shutdown")


def create_app(
    settings: Optional[Settings] = None,
    pipeline: Optional[Pipeline] = None,
) -> FastAPI:
    """
    Application factory.

    Parameters
    ----------
    settings    : Settings instance (env-driven when None)
    pipeline    : pre-built Pipeline (skip model loading when provided)
                  Useful in tests — pass a mock Pipeline with a no-op load_models.
    """
    cfg = settings or Settings()

    # ------------------------------------------------------------------
    # Metrics (created before pipeline so pipeline can reference it)
    # ------------------------------------------------------------------
    metrics: Optional[MetricsRegistry] = None
    if cfg.metrics_enabled:
        metrics = MetricsRegistry()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    if pipeline is None:
        pipeline = Pipeline(settings=cfg, metrics=metrics)

    # ------------------------------------------------------------------
    # Health checker
    # ------------------------------------------------------------------
    health_checker = HealthChecker(pipeline=pipeline)

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    app = FastAPI(
        title="Predictive Log Anomaly Engine",
        description="Stage 7 — REST API for real-time log anomaly detection",
        version="0.7.0",
        lifespan=_lifespan,
    )

    # Attach shared state
    app.state.pipeline = pipeline
    app.state.metrics = metrics
    app.state.health_checker = health_checker
    app.state.settings = cfg

    # ------------------------------------------------------------------
    # Middleware (outermost first → innermost last)
    # ------------------------------------------------------------------
    if cfg.metrics_enabled:
        app.add_middleware(MetricsMiddleware)

    app.add_middleware(
        AuthMiddleware,
        api_key=cfg.api_key,
        disable_auth=cfg.disable_auth,
        public_paths=cfg.public_endpoints,
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    app.include_router(router)
    app.include_router(router_v2)
    app.include_router(ui_router)

    return app
