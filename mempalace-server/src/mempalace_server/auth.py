"""Bearer-token authentication middleware for the remote MCP server.

Tokens are checked against the ``mp_api_keys`` table in the same Postgres
database used by the pgvector backend. The raw token is hashed with SHA-256
before lookup — only hashes are stored.

A small in-memory TTL cache (60 s) avoids hitting Postgres on every request.
The ``/health`` endpoint bypasses authentication so external monitors can
probe without credentials.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Any

import psycopg
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .rate_limit import RateLimiter

logger = logging.getLogger("mempalace_server.auth")

_CACHE_TTL_SECONDS = 60.0
_HEALTH_PATHS = {"/health", "/healthz"}


def _dsn() -> str:
    dsn = os.environ.get("MEMPALACE_PGVECTOR_DSN")
    if not dsn:
        raise RuntimeError("MEMPALACE_PGVECTOR_DSN env var is required for auth")
    return dsn


def verify_api_key(token: str, dsn: str | None = None) -> dict[str, Any] | None:
    """Look up an active key by SHA-256 hash. Returns dict or None."""
    if not token:
        return None
    key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    try:
        with psycopg.connect(dsn or _dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, rate_limit FROM mp_api_keys "
                    "WHERE key_hash = %s AND is_active = true",
                    (key_hash,),
                )
                row = cur.fetchone()
                if row:
                    return {"id": str(row[0]), "name": row[1], "rate_limit": int(row[2])}
    except psycopg.Error:
        logger.exception("DB error during API key verification")
        return None
    return None


class _TTLCache:
    """Tiny thread-safe TTL cache for verified tokens."""

    def __init__(self, ttl: float = _CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._data: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires < now:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        expires = time.monotonic() + self._ttl
        with self._lock:
            # Light-weight eviction: drop one expired entry if cache > 1024.
            if len(self._data) > 1024:
                now = time.monotonic()
                for k, (exp, _) in list(self._data.items()):
                    if exp < now:
                        self._data.pop(k, None)
                        break
            self._data[key] = (expires, value)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validates ``Authorization: Bearer <token>`` and enforces per-key rate limits."""

    def __init__(self, app, dsn: str | None = None, rate_limiter: RateLimiter | None = None):
        super().__init__(app)
        self._dsn = dsn
        self._cache = _TTLCache()
        self._rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next) -> Response:
        # Health endpoints bypass auth entirely.
        if request.url.path in _HEALTH_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header:
            return JSONResponse(
                {"error": "missing_authorization", "message": "Authorization header required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mempalace"'},
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return JSONResponse(
                {"error": "invalid_authorization", "message": "Expected 'Bearer <token>'"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mempalace"'},
            )
        token = token.strip()

        # Cache hit?
        info = self._cache.get(token)
        if info is None:
            info = verify_api_key(token, self._dsn)
            if info is None:
                return JSONResponse(
                    {"error": "invalid_api_key", "message": "API key not recognised or revoked"},
                    status_code=403,
                )
            self._cache.set(token, info)

        # Rate limit check.
        if not self._rate_limiter.check(info["name"], info.get("rate_limit", 120)):
            return JSONResponse(
                {
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded for key '{info['name']}'",
                },
                status_code=429,
                headers={"Retry-After": "60"},
            )

        # Stash for downstream tools/handlers if they want it.
        request.state.api_key = info
        return await call_next(request)
