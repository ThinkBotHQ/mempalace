"""
knowledge_graph.py — PostgreSQL-backed temporal knowledge graph for MemPalace.

Implements the same interface as ``mempalace.knowledge_graph.KnowledgeGraph``
but stores data in PostgreSQL via psycopg v3. Uses jsonb for entity properties
and pg_trgm for fuzzy name search.

Usage:
    from mempalace_pgvector.knowledge_graph import PgKnowledgeGraph

    kg = PgKnowledgeGraph(dsn="postgresql://postgres:mempalace@localhost:5434/mempalace")
    kg.add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
    kg.query_entity("Max", as_of="2026-01-15")
    kg.close()
"""

from __future__ import annotations

import hashlib
import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DEFAULT_DSN_ENV = "MEMPALACE_PGVECTOR_DSN"
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class PgKnowledgeGraph:
    """PostgreSQL-backed temporal knowledge graph.

    Parameters
    ----------
    dsn:
        PostgreSQL DSN. If omitted, falls back to ``MEMPALACE_PGVECTOR_DSN`` or
        ``DATABASE_URL`` environment variables.
    conn:
        Optional existing psycopg connection. Takes precedence over ``dsn``.
        The caller retains ownership; ``close()`` will not close it.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> None:
        self._external_conn = conn is not None
        self._connection: Optional[psycopg.Connection] = conn
        self._dsn = dsn or os.environ.get(DEFAULT_DSN_ENV) or os.environ.get("DATABASE_URL")
        if self._connection is None and not self._dsn:
            raise ValueError(
                "No DSN: pass dsn= or set MEMPALACE_PGVECTOR_DSN / DATABASE_URL, "
                "or pass an existing conn="
            )
        self._lock = threading.Lock()
        self._init_db()

    # ── Connection management ─────────────────────────────────────────────

    def _conn(self) -> psycopg.Connection:
        if self._connection is None or self._connection.closed:
            assert self._dsn is not None
            self._connection = psycopg.connect(self._dsn, row_factory=dict_row)
        elif self._connection.row_factory is not dict_row:
            # Ensure dict rows for consistent row access even when caller
            # supplied a connection with a different default.
            self._connection.row_factory = dict_row
        return self._connection

    def close(self) -> None:
        """Close the database connection (no-op if caller provided it)."""
        with self._lock:
            if self._connection is not None and not self._external_conn:
                try:
                    self._connection.close()
                finally:
                    self._connection = None

    def rename_entity_references(self, old_id: str, new_id: str) -> int:
        """Update all triples referencing ``old_id`` to use ``new_id``.

        Mirror of :meth:`mempalace.knowledge_graph.KnowledgeGraph.rename_entity_references`
        for the PostgreSQL backend. Uses the ``mp_kg_*`` table names and
        ``%s`` placeholders. Returns the count of triples updated.

        Ensures ``new_id`` exists as an entity row first so the foreign-key
        constraint on ``mp_kg_triples`` doesn't trip when callers haven't
        pre-seeded the target. The upsert leaves an existing row untouched.
        """
        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mp_kg_entities (id, name, type)
                    VALUES (%s, %s, 'unknown')
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (new_id, new_id),
                )
                cur.execute(
                    "UPDATE mp_kg_triples SET subject = %s WHERE subject = %s",
                    (new_id, old_id),
                )
                c1 = cur.rowcount or 0
                cur.execute(
                    "UPDATE mp_kg_triples SET object = %s WHERE object = %s",
                    (new_id, old_id),
                )
                c2 = cur.rowcount or 0
                cur.execute(
                    "DELETE FROM mp_kg_entities WHERE id = %s",
                    (old_id,),
                )
            conn.commit()
            return c1 + c2

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Ensure KG tables + indexes exist.

        We execute only the KG-related subset of ``schema.sql`` so that callers
        using this class without the full pgvector backend don't need the
        ``vector`` extension installed.
        """
        ddl = """
            CREATE EXTENSION IF NOT EXISTS pg_trgm;

            CREATE TABLE IF NOT EXISTS mp_kg_entities (
                id              text PRIMARY KEY,
                name            text NOT NULL,
                type            text DEFAULT 'unknown',
                properties      jsonb DEFAULT '{}',
                created_at      timestamptz DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_kg_entities_name_trgm
                ON mp_kg_entities USING gin (name gin_trgm_ops);
            CREATE INDEX IF NOT EXISTS idx_kg_entities_type
                ON mp_kg_entities (type);

            CREATE TABLE IF NOT EXISTS mp_kg_triples (
                id              text PRIMARY KEY,
                subject         text NOT NULL REFERENCES mp_kg_entities(id),
                predicate       text NOT NULL,
                object          text NOT NULL REFERENCES mp_kg_entities(id),
                valid_from      text,
                valid_to        text,
                confidence      real DEFAULT 1.0,
                source_closet   text,
                source_file     text,
                source_drawer_id text,
                adapter_name    text,
                metadata        jsonb DEFAULT '{}',
                extracted_at    timestamptz DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_kg_triples_subject ON mp_kg_triples(subject);
            CREATE INDEX IF NOT EXISTS idx_kg_triples_object ON mp_kg_triples(object);
            CREATE INDEX IF NOT EXISTS idx_kg_triples_predicate ON mp_kg_triples(predicate);
            CREATE INDEX IF NOT EXISTS idx_kg_triples_valid ON mp_kg_triples(valid_from, valid_to);
            CREATE INDEX IF NOT EXISTS idx_kg_triples_source_drawer ON mp_kg_triples(source_drawer_id);
        """
        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _entity_id(name: str) -> str:
        return name.lower().replace(" ", "_").replace("'", "")

    @staticmethod
    def _predicate_id(predicate: str) -> str:
        return predicate.lower().replace(" ", "_")

    # ── Write operations ──────────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        properties: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add or update an entity node. Returns the entity id."""
        eid = self._entity_id(name)
        props = Jsonb(properties or {})
        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mp_kg_entities (id, name, type, properties)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        type = EXCLUDED.type,
                        properties = EXCLUDED.properties
                    """,
                    (eid, name, entity_type, props),
                )
            conn.commit()
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        confidence: float = 1.0,
        source_closet: Optional[str] = None,
        source_file: Optional[str] = None,
        source_drawer_id: Optional[str] = None,
        adapter_name: Optional[str] = None,
    ) -> str:
        """Add a relationship triple ``subject → predicate → object``.

        Auto-creates entity rows for the subject and object if missing.
        Returns the triple id. If an identical still-valid triple already
        exists, returns that existing id without inserting a duplicate.
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = self._predicate_id(predicate)

        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO mp_kg_entities (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (sub_id, subject),
                )
                cur.execute(
                    "INSERT INTO mp_kg_entities (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (obj_id, obj),
                )

                cur.execute(
                    """
                    SELECT id FROM mp_kg_triples
                    WHERE subject = %s AND predicate = %s AND object = %s AND valid_to IS NULL
                    LIMIT 1
                    """,
                    (sub_id, pred, obj_id),
                )
                existing = cur.fetchone()
                if existing:
                    conn.commit()
                    return existing["id"]

                triple_id = (
                    f"t_{sub_id}_{pred}_{obj_id}_"
                    f"{hashlib.sha256(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
                )

                cur.execute(
                    """
                    INSERT INTO mp_kg_triples (
                        id, subject, predicate, object,
                        valid_from, valid_to, confidence,
                        source_closet, source_file,
                        source_drawer_id, adapter_name
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        triple_id,
                        sub_id,
                        pred,
                        obj_id,
                        valid_from,
                        valid_to,
                        confidence,
                        source_closet,
                        source_file,
                        source_drawer_id,
                        adapter_name,
                    ),
                )
            conn.commit()
        return triple_id

    def invalidate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        ended: Optional[str] = None,
    ) -> None:
        """Mark a relationship as no longer valid (set ``valid_to``)."""
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = self._predicate_id(predicate)
        ended = ended or date.today().isoformat()

        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE mp_kg_triples
                    SET valid_to = %s
                    WHERE subject = %s AND predicate = %s AND object = %s
                      AND valid_to IS NULL
                    """,
                    (ended, sub_id, pred, obj_id),
                )
            conn.commit()

    # ── Query operations ──────────────────────────────────────────────────

    def query_entity(
        self,
        name: str,
        as_of: Optional[str] = None,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """Get all relationships for an entity.

        Parameters
        ----------
        direction: ``"outgoing"``, ``"incoming"``, or ``"both"``.
        as_of: ISO date/time string. Only return facts valid at that time.
        """
        eid = self._entity_id(name)
        results: list[dict[str, Any]] = []

        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                if direction in ("outgoing", "both"):
                    query = (
                        "SELECT t.*, e.name AS obj_name FROM mp_kg_triples t "
                        "JOIN mp_kg_entities e ON t.object = e.id WHERE t.subject = %s"
                    )
                    params: list[Any] = [eid]
                    if as_of:
                        query += (
                            " AND (t.valid_from IS NULL OR t.valid_from <= %s) "
                            "AND (t.valid_to IS NULL OR t.valid_to >= %s)"
                        )
                        params.extend([as_of, as_of])
                    cur.execute(query, params)
                    for row in cur.fetchall():
                        results.append(
                            {
                                "direction": "outgoing",
                                "subject": name,
                                "predicate": row["predicate"],
                                "object": row["obj_name"],
                                "valid_from": row["valid_from"],
                                "valid_to": row["valid_to"],
                                "confidence": row["confidence"],
                                "source_closet": row["source_closet"],
                                "current": row["valid_to"] is None,
                            }
                        )

                if direction in ("incoming", "both"):
                    query = (
                        "SELECT t.*, e.name AS sub_name FROM mp_kg_triples t "
                        "JOIN mp_kg_entities e ON t.subject = e.id WHERE t.object = %s"
                    )
                    params = [eid]
                    if as_of:
                        query += (
                            " AND (t.valid_from IS NULL OR t.valid_from <= %s) "
                            "AND (t.valid_to IS NULL OR t.valid_to >= %s)"
                        )
                        params.extend([as_of, as_of])
                    cur.execute(query, params)
                    for row in cur.fetchall():
                        results.append(
                            {
                                "direction": "incoming",
                                "subject": row["sub_name"],
                                "predicate": row["predicate"],
                                "object": name,
                                "valid_from": row["valid_from"],
                                "valid_to": row["valid_to"],
                                "confidence": row["confidence"],
                                "source_closet": row["source_closet"],
                                "current": row["valid_to"] is None,
                            }
                        )
        return results

    def query_relationship(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get triples matching any combination of subject/predicate/object.

        All three filters are optional; omitting a filter matches any value
        for that position. When only ``predicate`` is supplied this behaves
        like the SQLite ``query_relationship(predicate)`` helper.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if subject is not None:
            clauses.append("t.subject = %s")
            params.append(self._entity_id(subject))
        if predicate is not None:
            clauses.append("t.predicate = %s")
            params.append(self._predicate_id(predicate))
        if obj is not None:
            clauses.append("t.object = %s")
            params.append(self._entity_id(obj))
        if as_of:
            clauses.append(
                "(t.valid_from IS NULL OR t.valid_from <= %s) "
                "AND (t.valid_to IS NULL OR t.valid_to >= %s)"
            )
            params.extend([as_of, as_of])

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = (
            "SELECT t.*, s.name AS sub_name, o.name AS obj_name "
            "FROM mp_kg_triples t "
            "JOIN mp_kg_entities s ON t.subject = s.id "
            "JOIN mp_kg_entities o ON t.object = o.id" + where_sql
        )

        results: list[dict[str, Any]] = []
        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(query, params)
                for row in cur.fetchall():
                    results.append(
                        {
                            "subject": row["sub_name"],
                            "predicate": row["predicate"],
                            "object": row["obj_name"],
                            "valid_from": row["valid_from"],
                            "valid_to": row["valid_to"],
                            "confidence": row["confidence"],
                            "current": row["valid_to"] is None,
                        }
                    )
        return results

    def timeline(
        self,
        entity_name: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get facts in chronological order, optionally filtered by entity."""
        results: list[dict[str, Any]] = []
        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                if entity_name:
                    eid = self._entity_id(entity_name)
                    cur.execute(
                        """
                        SELECT t.*, s.name AS sub_name, o.name AS obj_name
                        FROM mp_kg_triples t
                        JOIN mp_kg_entities s ON t.subject = s.id
                        JOIN mp_kg_entities o ON t.object = o.id
                        WHERE t.subject = %s OR t.object = %s
                        ORDER BY t.valid_from ASC NULLS LAST
                        LIMIT %s
                        """,
                        (eid, eid, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT t.*, s.name AS sub_name, o.name AS obj_name
                        FROM mp_kg_triples t
                        JOIN mp_kg_entities s ON t.subject = s.id
                        JOIN mp_kg_entities o ON t.object = o.id
                        ORDER BY t.valid_from ASC NULLS LAST
                        LIMIT %s
                        """,
                        (limit,),
                    )
                for row in cur.fetchall():
                    results.append(
                        {
                            "subject": row["sub_name"],
                            "predicate": row["predicate"],
                            "object": row["obj_name"],
                            "valid_from": row["valid_from"],
                            "valid_to": row["valid_to"],
                            "current": row["valid_to"] is None,
                        }
                    )
        return results

    def find_similar_entities(
        self,
        name: str,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fuzzy-match entity names via pg_trgm similarity.

        Returns entities whose name similarity is ``>= threshold``, ordered by
        similarity descending.
        """
        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, type, properties,
                           similarity(name, %s) AS sim
                    FROM mp_kg_entities
                    WHERE similarity(name, %s) >= %s
                    ORDER BY sim DESC
                    LIMIT %s
                    """,
                    (name, name, threshold, limit),
                )
                return [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "type": row["type"],
                        "properties": row["properties"],
                        "similarity": float(row["sim"]),
                    }
                    for row in cur.fetchall()
                ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def graph_stats(self) -> dict[str, Any]:
        """Count entities, triples, current/expired facts, and predicates."""
        with self._lock:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM mp_kg_entities")
                entities = cur.fetchone()["cnt"]
                cur.execute("SELECT COUNT(*) AS cnt FROM mp_kg_triples")
                triples = cur.fetchone()["cnt"]
                cur.execute("SELECT COUNT(*) AS cnt FROM mp_kg_triples WHERE valid_to IS NULL")
                current = cur.fetchone()["cnt"]
                cur.execute("SELECT DISTINCT predicate FROM mp_kg_triples ORDER BY predicate")
                predicates = [r["predicate"] for r in cur.fetchall()]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": triples - current,
            "relationship_types": predicates,
        }

    # Back-compat alias matching the SQLite class.
    stats = graph_stats
