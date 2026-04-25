"""PgvectorCollection — BaseCollection implementation backed by PostgreSQL + pgvector."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Optional

import psycopg
from psycopg.types.json import Jsonb

from mempalace.backends.base import (
    BaseCollection,
    GetResult,
    HealthStatus,
    QueryResult,
    _IncludeSpec,
)

from .embedder import Embedder
from .where_compiler import compile_where

logger = logging.getLogger(__name__)


def _positional_to_pyformat(sql: str, params: list, offset: int) -> tuple[str, list]:
    """Convert $N positional params from where_compiler to %s pyformat for psycopg."""
    if not params:
        return sql, params
    result = sql
    for i in range(len(params), 0, -1):
        result = result.replace(f"${i + offset}", "%s")
    return result, params


class PgvectorCollection(BaseCollection):
    def __init__(
        self,
        conn: psycopg.Connection,
        collection_id: uuid.UUID,
        embedder: Embedder,
    ) -> None:
        self._conn = conn
        self._collection_id = collection_id
        self._embedder = embedder

        from pgvector.psycopg import register_vector

        register_vector(conn)

    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            embeddings = self._embedder.embed(documents)
        if metadatas is None:
            metadatas = [{}] * len(documents)

        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO mp_documents (collection_id, item_id, document, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        str(self._collection_id),
                        ids[i],
                        documents[i],
                        Jsonb(metadatas[i]),
                        embeddings[i],
                    )
                    for i in range(len(ids))
                ],
            )
        self._conn.commit()

    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            embeddings = self._embedder.embed(documents)
        if metadatas is None:
            metadatas = [{}] * len(documents)

        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO mp_documents (collection_id, item_id, document, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (collection_id, item_id) DO UPDATE SET
                    document = EXCLUDED.document,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        str(self._collection_id),
                        ids[i],
                        documents[i],
                        Jsonb(metadatas[i]),
                        embeddings[i],
                    )
                    for i in range(len(ids))
                ],
            )
        self._conn.commit()

    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult:
        if query_texts is None and query_embeddings is None:
            return QueryResult.empty()

        if query_embeddings is None:
            query_embeddings = self._embedder.embed_query(query_texts)

        spec = _IncludeSpec.resolve(include)

        all_ids: list[list[str]] = []
        all_docs: list[list[str]] = []
        all_metas: list[list[dict]] = []
        all_dists: list[list[float]] = []
        all_embeds: list[list[list[float]]] | None = [] if spec.embeddings else None

        for qvec in query_embeddings:
            filter_sql, filter_params = compile_where(where, where_document, param_offset=2)
            filter_sql_py, filter_params = _positional_to_pyformat(filter_sql, filter_params, 2)

            where_clause = "WHERE d.collection_id = %s"
            params: list = [str(self._collection_id)]

            if filter_sql_py:
                where_clause += f" AND {filter_sql_py}"
                params.extend(filter_params)

            select_cols = "d.item_id, d.document, d.metadata, (1 - (d.embedding <=> %s::vector))::float AS similarity"
            if spec.embeddings:
                select_cols += ", d.embedding"

            sql = f"""
                SELECT {select_cols}
                FROM mp_documents d
                {where_clause}
                ORDER BY d.embedding <=> %s::vector
                LIMIT %s
            """
            params_full = [str(qvec)] + params + [str(qvec), n_results]

            with self._conn.cursor() as cur:
                cur.execute("SET LOCAL hnsw.ef_search = 100")
                try:
                    cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                except psycopg.errors.UndefinedObject:
                    pass
                cur.execute(sql, params_full)
                rows = cur.fetchall()

            hit_ids: list[str] = []
            hit_docs: list[str] = []
            hit_metas: list[dict] = []
            hit_dists: list[float] = []
            hit_embeds: list[list[float]] = []

            for row in rows:
                hit_ids.append(row[0])
                hit_docs.append(row[1] if spec.documents else "")
                hit_metas.append(row[2] if spec.metadatas else {})
                hit_dists.append(1.0 - row[3] if spec.distances else 0.0)
                if spec.embeddings and len(row) > 4:
                    hit_embeds.append(list(row[4]))

            all_ids.append(hit_ids)
            all_docs.append(hit_docs)
            all_metas.append(hit_metas)
            all_dists.append(hit_dists)
            if all_embeds is not None:
                all_embeds.append(hit_embeds)

        return QueryResult(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            distances=all_dists,
            embeddings=all_embeds,
        )

    def hybrid_query(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> QueryResult:
        """Hybrid vector + full-text search using Reciprocal Rank Fusion.

        Runs two candidate searches in parallel inside a single SQL CTE:
        - Vector: top 50 by cosine similarity (``embedding <=> query``)
        - FTS:    top 50 by ts_rank_cd against ``websearch_to_tsquery``

        The results are fused with Reciprocal Rank Fusion (k=60), the
        canonical formulation from Cormack et al. 2009::

            score(doc) = sum over searches of 1 / (k + rank)

        Filters from ``where`` / ``where_document`` are applied to BOTH
        candidate sets so RRF ranks within the filtered subset.

        Returns a standard ``QueryResult`` with ``ids[0]``, ``documents[0]``,
        ``metadatas[0]``, and ``distances[0]`` populated. Distances here
        are ``1 - rrf_score`` so smaller is "better" (matching the rest of
        the QueryResult convention).
        """
        # Filter SQL is reused inside both branches of the CTE. compile_where
        # yields a fragment that does not include the leading WHERE.
        # We use param_offset=0 here because we'll pass the filter params
        # twice (once for each branch) and re-bind via psycopg %s pyformat.
        filter_sql, filter_params = compile_where(where, where_document, param_offset=0)
        filter_sql_py, filter_params = _positional_to_pyformat(filter_sql, filter_params, 0)

        extra_filter = f" AND {filter_sql_py}" if filter_sql_py else ""

        sql = f"""
            WITH vec AS (
                SELECT d.item_id,
                       row_number() OVER (ORDER BY d.embedding <=> %s::vector) AS rnk
                FROM mp_documents d
                WHERE d.collection_id = %s{extra_filter}
                ORDER BY d.embedding <=> %s::vector
                LIMIT 50
            ),
            fts AS (
                SELECT d.item_id,
                       row_number() OVER (
                           ORDER BY ts_rank_cd(d.document_tsv,
                                               websearch_to_tsquery('english', %s)) DESC
                       ) AS rnk
                FROM mp_documents d
                WHERE d.collection_id = %s
                  AND d.document_tsv @@ websearch_to_tsquery('english', %s){extra_filter}
                ORDER BY ts_rank_cd(d.document_tsv,
                                    websearch_to_tsquery('english', %s)) DESC
                LIMIT 50
            ),
            fused AS (
                SELECT item_id,
                       SUM(1.0 / (60 + rnk)) AS score
                FROM (
                    SELECT item_id, rnk FROM vec
                    UNION ALL
                    SELECT item_id, rnk FROM fts
                ) u
                GROUP BY item_id
            )
            SELECT d.item_id, d.document, d.metadata, f.score
            FROM fused f
            JOIN mp_documents d
              ON d.collection_id = %s AND d.item_id = f.item_id
            ORDER BY f.score DESC
            LIMIT %s
        """

        params: list = []
        # vec branch
        params.append(query_embedding)  # ORDER BY in row_number
        params.append(str(self._collection_id))  # collection_id =
        params.extend(filter_params)  # filter (vec)
        params.append(query_embedding)  # outer ORDER BY for LIMIT
        # fts branch
        params.append(query_text)  # ts_rank_cd in row_number
        params.append(str(self._collection_id))  # collection_id =
        params.append(query_text)  # @@ websearch_to_tsquery
        params.extend(filter_params)  # filter (fts)
        params.append(query_text)  # outer ORDER BY ts_rank_cd
        # final join
        params.append(str(self._collection_id))  # JOIN d.collection_id
        params.append(n_results)

        with self._conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = 100")
            try:
                cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            except psycopg.errors.UndefinedObject:
                pass
            cur.execute(sql, params)
            rows = cur.fetchall()

        hit_ids: list[str] = []
        hit_docs: list[str] = []
        hit_metas: list[dict] = []
        hit_dists: list[float] = []

        # Maximum possible RRF score for 2 sources (vec + fts) with k=60:
        # each source contributes at most 1/(60+1), so max = 2/61 ~ 0.0328.
        max_rrf = 2.0 / 61.0

        for row in rows:
            hit_ids.append(row[0])
            hit_docs.append(row[1])
            hit_metas.append(row[2])
            # Normalize RRF score to [0, 1] then convert to distance (lower = better)
            normalized_score = min(float(row[3]) / max_rrf, 1.0)
            hit_dists.append(1.0 - normalized_score)

        return QueryResult(
            ids=[hit_ids],
            documents=[hit_docs],
            metadatas=[hit_metas],
            distances=[hit_dists],
            embeddings=None,
        )

    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        spec = _IncludeSpec.resolve(include, default_distances=False)

        where_clause = "WHERE d.collection_id = %s"
        params: list = [str(self._collection_id)]

        if ids is not None:
            where_clause += " AND d.item_id = ANY(%s)"
            params.append(ids)

        filter_sql, filter_params = compile_where(where, where_document, param_offset=len(params))
        filter_sql_py, filter_params = _positional_to_pyformat(
            filter_sql, filter_params, len(params)
        )
        if filter_sql_py:
            where_clause += f" AND {filter_sql_py}"
            params.extend(filter_params)

        select_cols = "d.item_id, d.document, d.metadata"
        if spec.embeddings:
            select_cols += ", d.embedding"

        sql = f"SELECT {select_cols} FROM mp_documents d {where_clause} ORDER BY d.created_at"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        if offset is not None:
            sql += " OFFSET %s"
            params.append(offset)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        result_ids: list[str] = []
        result_docs: list[str] = []
        result_metas: list[dict] = []
        result_embeds: list[list[float]] | None = [] if spec.embeddings else None

        for row in rows:
            result_ids.append(row[0])
            result_docs.append(row[1] if spec.documents else "")
            result_metas.append(row[2] if spec.metadatas else {})
            if result_embeds is not None and len(row) > 3:
                result_embeds.append(list(row[3]))

        return GetResult(
            ids=result_ids,
            documents=result_docs,
            metadatas=result_metas,
            embeddings=result_embeds,
        )

    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None:
        where_clause = "WHERE collection_id = %s"
        params: list = [str(self._collection_id)]

        if ids is not None:
            where_clause += " AND item_id = ANY(%s)"
            params.append(ids)

        filter_sql, filter_params = compile_where(where, param_offset=len(params))
        filter_sql_py, filter_params = _positional_to_pyformat(
            filter_sql, filter_params, len(params)
        )
        if filter_sql_py:
            where_clause += f" AND {filter_sql_py}"
            params.extend(filter_params)

        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM mp_documents {where_clause}", params)
        self._conn.commit()

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM mp_documents WHERE collection_id = %s",
                (str(self._collection_id),),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def health(self) -> HealthStatus:
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return HealthStatus.healthy("pgvector connection ok")
        except Exception as e:
            return HealthStatus.unhealthy(str(e))
