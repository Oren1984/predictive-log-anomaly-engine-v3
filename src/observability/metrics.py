# src/observability/metrics.py

# Purpose: Define the MetricsRegistry and MetricsMiddleware classes for 
# collecting and exposing Prometheus metrics related to the API service.

# Input: The MetricsRegistry class takes an optional CollectorRegistry instance.
#        The MetricsMiddleware class takes no input but relies on 
#        a MetricsRegistry instance being available in the app state.

# Output: The MetricsRegistry class provides methods to generate Prometheus metrics in text format.
#         The MetricsMiddleware class records HTTP request counts and latency for every route.


# Used by: This module is used by the API service to collect and expose Prometheus metrics.
# By integrating MetricsRegistry and MetricsMiddleware, the API service can monitor
# key performance indicators such as request counts, latency, and error rates.

"""Stage 7 — Observability: Prometheus MetricsRegistry + MetricsMiddleware."""
from __future__ import annotations

import time
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class MetricsRegistry:
    """
    Wraps a private ``CollectorRegistry`` so each instance registers its own
    collectors — this prevents ``ValueError: Duplicated timeseries`` when
    multiple instances are created during tests.

    Metrics exposed
    ---------------
    ingest_events_total          Counter  — events POSTed to /ingest
    ingest_windows_total         Counter  — windows emitted by InferenceEngine
    alerts_total                 Counter  — alerts fired (labelled by severity)
    ingest_errors_total          Counter  — unhandled errors in /ingest
    ingest_latency_seconds       Histogram — end-to-end /ingest handler latency
    scoring_latency_seconds      Histogram — model scoring latency (per window)
    service_health               Gauge    — application health: 1=healthy, 0.5=degraded, 0=unhealthy
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.ingest_events_total = Counter(
            "ingest_events_total",
            "Total events received by POST /ingest",
            registry=self.registry,
        )
        self.ingest_windows_total = Counter(
            "ingest_windows_total",
            "Total scoring windows emitted by InferenceEngine",
            registry=self.registry,
        )
        self.alerts_total = Counter(
            "alerts_total",
            "Total alerts fired",
            ["severity"],
            registry=self.registry,
        )
        self.ingest_errors_total = Counter(
            "ingest_errors_total",
            "Total unhandled errors in /ingest",
            registry=self.registry,
        )
        self.ingest_latency_seconds = Histogram(
            "ingest_latency_seconds",
            "End-to-end /ingest handler latency in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        self.scoring_latency_seconds = Histogram(
            "scoring_latency_seconds",
            "Model scoring latency per window in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        # Initialise optimistically; the /health handler updates this on each poll.
        # Values: 1.0=healthy  0.5=degraded  0.0=unhealthy
        self.service_health = Gauge(
            "service_health",
            "Application health state: 1=healthy, 0.5=degraded, 0=unhealthy",
            registry=self.registry,
        )
        self.service_health.set(1.0)

    def generate_text(self) -> tuple[str, str]:
        """Return (body, content_type) for the /metrics endpoint."""
        return (
            generate_latest(self.registry).decode("utf-8"),
            CONTENT_TYPE_LATEST,
        )


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Records POST /ingest latency via the ingest_latency_seconds histogram.
    Other routes are passed through without instrumentation.
    Only active when a MetricsRegistry is wired into app state.
    """

    async def dispatch(self, request: Request, call_next):
        metrics: Optional[MetricsRegistry] = getattr(
            request.app.state, "metrics", None
        )
        if metrics is None:
            return await call_next(request)

        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0

        # Only track /ingest latency with the dedicated histogram
        if request.url.path == "/ingest" and request.method == "POST":
            metrics.ingest_latency_seconds.observe(elapsed)

        return response
