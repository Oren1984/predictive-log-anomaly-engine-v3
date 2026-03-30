# src/security/auth.py

# Purpose: Defines the AuthMiddleware class, which implements API key authentication for incoming HTTP requests. 
# It checks the "X-API-Key" header against a configured value and allows or denies access accordingly. 
# It also supports disabling authentication and defining public paths that bypass the check.

# Input: - API key value (from environment variable or constructor argument)
#        - Disable auth flag (from environment variable or constructor argument)

# Output: - HTTP 401 response for unauthorized requests
#         - Pass-through for authorized requests or when auth is disabled

# Used by: - The main application to enforce authentication on incoming requests.

"""Stage 7 — Security: X-API-Key header authentication middleware."""
from __future__ import annotations

import logging
import os
from typing import Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Validates the ``X-API-Key`` request header for every non-public endpoint.

    Parameters
    ----------
    api_key         : expected key value; falls back to env ``API_KEY``
    disable_auth    : when True, all requests pass through (env ``DISABLE_AUTH=true``)
    public_paths    : set of path prefixes that skip auth (default: /health, /metrics)
    """

    def __init__(
        self,
        app,
        api_key: str | None = None,
        disable_auth: bool | None = None,
        public_paths: Sequence[str] | None = None,
    ) -> None:
        super().__init__(app)

        self.api_key: str = (
            api_key
            if api_key is not None
            else os.environ.get("API_KEY", "")
        )

        if disable_auth is not None:
            self.disable_auth = disable_auth
        else:
            self.disable_auth = os.environ.get("DISABLE_AUTH", "false").lower() in (
                "true", "1", "yes"
            )

        self.public_paths: tuple[str, ...] = tuple(
            public_paths if public_paths is not None else ("/health", "/metrics")
        )

    # ------------------------------------------------------------------
    def _is_public(self, path: str) -> bool:
        return any(path == p or path.startswith(p + "/") for p in self.public_paths)

    async def dispatch(self, request: Request, call_next):
        if self.disable_auth or self._is_public(request.url.path):
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        if not self.api_key:
            # No key configured — allow all traffic but log a warning once
            logger.warning("AuthMiddleware: API_KEY not set; bypassing auth check")
            return await call_next(request)

        if provided != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header"},
            )

        return await call_next(request)
