# src/security/__init__.py

"""Stage 7 — Security: API key authentication middleware."""
from .auth import AuthMiddleware

__all__ = ["AuthMiddleware"]
