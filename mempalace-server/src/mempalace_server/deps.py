"""Backend lifecycle helpers for the HTTP server.

Wraps the lazy singletons defined in ``mempalace.mcp_server`` so we can
initialize them once at startup (warming the connection pool, flushing any
schema migrations) and shut them down cleanly on exit.
"""

from __future__ import annotations

import logging
import os

import psycopg

logger = logging.getLogger("mempalace_server.deps")


CREATE_API_KEYS_SQL = """
CREATE TABLE IF NOT EXISTS mp_api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    key_hash    TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    rate_limit  INT DEFAULT 120,
    is_active   BOOLEAN DEFAULT true
);
CREATE INDEX IF NOT EXISTS mp_api_keys_key_hash_idx ON mp_api_keys (key_hash);
"""


def get_dsn() -> str:
    dsn = os.environ.get("MEMPALACE_PGVECTOR_DSN")
    if not dsn:
        raise RuntimeError("MEMPALACE_PGVECTOR_DSN env var is required")
    return dsn


def ensure_api_keys_table(dsn: str | None = None) -> None:
    """Create mp_api_keys table if it does not exist. Also enables pgcrypto."""
    target = dsn or get_dsn()
    with psycopg.connect(target) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            except psycopg.Error:
                # pgcrypto may not be available — gen_random_uuid() will fail
                # but we let the user discover that through CREATE TABLE.
                logger.warning("Could not enable pgcrypto extension")
            cur.execute(CREATE_API_KEYS_SQL)
        conn.commit()
    logger.info("mp_api_keys table ensured")


def warm_backend() -> None:
    """Force-initialize the mempalace backend + KG so the first request is fast."""
    # Imported lazily so importing this module doesn't drag in the world.
    from mempalace import mcp_server as core

    try:
        core._get_collection()
        logger.info("Backend collection initialized")
    except Exception:
        logger.exception("Failed to warm backend collection")
        raise
    try:
        core._get_kg()
        logger.info("Knowledge graph initialized")
    except Exception:
        logger.exception("Failed to warm knowledge graph")


def shutdown_backend() -> None:
    """Best-effort cleanup of backend resources."""
    try:
        from mempalace import mcp_server as core

        backend = getattr(core, "_backend", None)
        close = getattr(backend, "close", None)
        if callable(close):
            close()
            logger.info("Backend closed")
    except Exception:
        logger.exception("Error during backend shutdown")
