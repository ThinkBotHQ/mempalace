"""PgvectorBackend — BaseBackend implementation for PostgreSQL + pgvector."""

from __future__ import annotations

import logging
import os
from typing import ClassVar, Optional

import psycopg
from psycopg_pool import ConnectionPool

from mempalace.backends.base import (
    BaseBackend,
    BaseCollection,
    EmbedderIdentityMismatchError,
    HealthStatus,
    PalaceNotFoundError,
    PalaceRef,
)

from .collection import PgvectorCollection
from .embedder import Embedder, GeminiEmbedder

logger = logging.getLogger(__name__)


class PgvectorBackend(BaseBackend):
    name: ClassVar[str] = "pgvector"
    spec_version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"vector_search", "jsonb_filter"})

    def __init__(
        self,
        dsn: str | None = None,
        embedder: Embedder | None = None,
        pool_max: int | None = None,
    ) -> None:
        self._dsn = dsn or os.environ.get("MEMPALACE_PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise ValueError(
                "No DSN: set MEMPALACE_PGVECTOR_DSN or DATABASE_URL, "
                "or pass dsn= to PgvectorBackend"
            )

        self._pool_max = pool_max or int(os.environ.get("MEMPALACE_PGVECTOR_POOL_MAX", "4"))
        self._pool = ConnectionPool(self._dsn, min_size=1, max_size=self._pool_max, open=True)
        self._embedder = embedder or self._make_default_embedder()
        self._collections: dict[str, PgvectorCollection] = {}

    def _make_default_embedder(self) -> GeminiEmbedder:
        model = os.environ.get("MEMPALACE_EMBEDDER_MODEL", "gemini-embedding-2")
        dim = int(os.environ.get("MEMPALACE_EMBEDDER_DIM", "768"))
        return GeminiEmbedder(model=model, dimension=dim)

    def get_collection(
        self,
        *,
        palace: PalaceRef,
        collection_name: str,
        create: bool = False,
        options: Optional[dict] = None,
    ) -> BaseCollection:
        cache_key = f"{palace.id}:{collection_name}"
        if cache_key in self._collections:
            return self._collections[cache_key]

        conn = self._pool.getconn()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, embedder_name, embedder_dim FROM mp_collections WHERE palace_id = %s AND collection_name = %s",
                (palace.id, collection_name),
            )
            row = cur.fetchone()

        if row is None:
            if not create:
                self._pool.putconn(conn)
                raise PalaceNotFoundError(
                    f"Collection {collection_name!r} not found in palace {palace.id!r}"
                )
            import uuid

            coll_id = uuid.uuid4()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mp_collections (id, palace_id, namespace, collection_name, embedder_name, embedder_dim)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (str(coll_id), palace.id, palace.namespace, collection_name, self._embedder.name, self._embedder.dimension),
                )
            conn.commit()
        else:
            coll_id = row[0]
            stored_name = row[1]
            if stored_name != self._embedder.name:
                self._pool.putconn(conn)
                raise EmbedderIdentityMismatchError(
                    f"Collection {collection_name!r} was created with embedder {stored_name!r}, "
                    f"but current embedder is {self._embedder.name!r}. "
                    f"Re-embedding required to switch models."
                )

        coll = PgvectorCollection(conn=conn, collection_id=coll_id, embedder=self._embedder)
        self._collections[cache_key] = coll
        return coll

    def close_palace(self, palace: PalaceRef) -> None:
        prefix = f"{palace.id}:"
        to_remove = [k for k in self._collections if k.startswith(prefix)]
        for k in to_remove:
            del self._collections[k]

    def close(self) -> None:
        self._collections.clear()
        self._pool.close()

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        try:
            conn = self._pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return HealthStatus.healthy(f"pgvector pool: {self._pool.get_stats()}")
            finally:
                self._pool.putconn(conn)
        except Exception as e:
            return HealthStatus.unhealthy(str(e))
