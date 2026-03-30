# src/api/settings.py

# Purpose: Define the Settings dataclass for the API, 
# which encapsulates all configuration parameters for the service.
# The settings are designed to be easily configurable via environment variables, 
# allowing for flexible deployment and runtime configuration.

# Input: The Settings class is defined as a dataclass, 
# with each field corresponding to a specific configuration parameter for the API.
# Each field has a default value that can be overridden by an environment variable.

# Output: The Settings class provides a structured way to access configuration parameters throughout the API implementation.
# It is used in the main application factory (src.api.app.py) to configure the API service
# and in the Pipeline class (src.api.pipeline.py) to configure the inference engine and alert manager.

# Used by: The Settings class defined in this file is used 
# throughout the API implementation to access configuration settings.

"""Stage 7 — API: application settings (env-driven)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Helper to parse boolean environment variables with a default fallback.
def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default

# Note: We could use Pydantic's BaseSettings here for more features (e.g., validation, .env files),
# but a simple dataclass with env var fallbacks is sufficient for this stage.
@dataclass
class Settings:
    """
    All tuneable parameters for the Stage 7 API service.

    Each field falls back to an environment variable with the same name
    (upper-cased) when not supplied directly.
    """

    # -- Server
    api_host: str = field(
        default_factory=lambda: os.environ.get("API_HOST", "0.0.0.0")
    )
    api_port: int = field(
        default_factory=lambda: int(os.environ.get("API_PORT", "8000"))
    )

    # -- Security
    api_key: str = field(
        default_factory=lambda: os.environ.get("API_KEY", "")
    )
    disable_auth: bool = field(
        default_factory=lambda: _env_bool("DISABLE_AUTH", False)
    )
    public_endpoints: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip()
            for p in os.environ.get(
                "PUBLIC_ENDPOINTS", "/health,/metrics,/,/query"
            ).split(",")
            if p.strip()
        )
    )

    # -- Observability
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )

    # -- Inference pipeline
    model_mode: str = field(
        default_factory=lambda: os.environ.get("MODEL_MODE", "ensemble")
    )
    window_size: int = field(
        default_factory=lambda: int(os.environ.get("WINDOW_SIZE", "50"))
    )
    stride: int = field(
        default_factory=lambda: int(os.environ.get("STRIDE", "10"))
    )

    # -- Alert buffer
    alert_buffer_size: int = field(
        default_factory=lambda: int(os.environ.get("ALERT_BUFFER_SIZE", "200"))
    )

    # -- Alert cooldown
    alert_cooldown_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("ALERT_COOLDOWN_SECONDS", "60.0")
        )
    )

    # -- Demo / fallback mode
    # DEMO_MODE=true lowers the fallback scorer output to a value above the
    # model threshold so at least one alert fires even without trained models.
    # Never enable in production.
    demo_mode: bool = field(
        default_factory=lambda: _env_bool("DEMO_MODE", False)
    )
    demo_score: float = field(
        default_factory=lambda: float(os.environ.get("DEMO_SCORE", "2.0"))
    )

    # -- Demo warmup traffic
    # When enabled, the API ingests a small synthetic batch on startup so the
    # demo shows live data immediately.  Disabled by default; safe for prod.
    demo_warmup_enabled: bool = field(
        default_factory=lambda: _env_bool("DEMO_WARMUP_ENABLED", False)
    )
    demo_warmup_events: int = field(
        default_factory=lambda: int(os.environ.get("DEMO_WARMUP_EVENTS", "75"))
    )
    # If > 0, repeat warmup every N seconds; otherwise run once.
    demo_warmup_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("DEMO_WARMUP_INTERVAL_SECONDS", "0")
        )
    )
