"""Integration tests for PgKnowledgeGraph against a real PostgreSQL instance.

Requires a running PostgreSQL at the DSN below with the KG schema applied
(it will be applied automatically by ``_init_db`` on first use).
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import psycopg
import pytest

from mempalace_pgvector.knowledge_graph import PgKnowledgeGraph
from mempalace_pgvector.migrate_kg import migrate_kg


DSN = "postgresql://postgres:mempalace@localhost:5434/mempalace"


def _unique(prefix: str) -> str:
    """Return a unique entity name so parallel tests don't collide."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def kg():
    graph = PgKnowledgeGraph(dsn=DSN)
    yield graph
    # Clean up any entities/triples created with our unique-prefix names so
    # tests stay isolated without dropping shared tables.
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM mp_kg_triples WHERE subject LIKE 'kgtest-%' OR object LIKE 'kgtest-%'"
        )
        cur.execute("DELETE FROM mp_kg_entities WHERE id LIKE 'kgtest-%'")
        conn.commit()
    graph.close()


def test_add_and_query_triple(kg):
    subj = _unique("kgtest-max")
    obj = _unique("kgtest-alice")
    triple_id = kg.add_triple(subj, "child_of", obj, valid_from="2015-04-01")
    assert triple_id.startswith("t_")

    outgoing = kg.query_entity(subj, direction="outgoing")
    assert len(outgoing) == 1
    assert outgoing[0]["predicate"] == "child_of"
    assert outgoing[0]["object"] == obj
    assert outgoing[0]["current"] is True

    incoming = kg.query_entity(obj, direction="incoming")
    assert len(incoming) == 1
    assert incoming[0]["subject"] == subj

    both = kg.query_entity(subj, direction="both")
    assert len(both) == 1

    # Duplicate still-valid triple should return the same id.
    again = kg.add_triple(subj, "child_of", obj, valid_from="2015-04-01")
    assert again == triple_id


def test_query_entity_as_of(kg):
    subj = _unique("kgtest-max")
    swim = _unique("kgtest-swim")
    chess = _unique("kgtest-chess")

    kg.add_triple(subj, "does", swim, valid_from="2025-01-01", valid_to="2025-06-30")
    kg.add_triple(subj, "loves", chess, valid_from="2025-10-01")

    in_feb = kg.query_entity(subj, as_of="2025-02-15")
    predicates = {r["predicate"] for r in in_feb}
    assert predicates == {"does"}

    in_nov = kg.query_entity(subj, as_of="2025-11-01")
    predicates = {r["predicate"] for r in in_nov}
    assert predicates == {"loves"}

    in_jul = kg.query_entity(subj, as_of="2025-07-15")
    assert in_jul == []


def test_invalidate(kg):
    subj = _unique("kgtest-max")
    obj = _unique("kgtest-injury")
    kg.add_triple(subj, "has_issue", obj, valid_from="2026-01-01")

    before = kg.query_entity(subj)
    assert before[0]["current"] is True
    assert before[0]["valid_to"] is None

    kg.invalidate(subj, "has_issue", obj, ended="2026-02-15")

    after = kg.query_entity(subj)
    assert after[0]["current"] is False
    assert after[0]["valid_to"] == "2026-02-15"


def test_timeline(kg):
    subj = _unique("kgtest-max")
    a = _unique("kgtest-a")
    b = _unique("kgtest-b")
    c = _unique("kgtest-c")

    kg.add_triple(subj, "did", a, valid_from="2020-01-01")
    kg.add_triple(subj, "did", b, valid_from="2022-06-01")
    kg.add_triple(subj, "did", c, valid_from="2024-03-01")

    tl = kg.timeline(subj)
    valid_froms = [r["valid_from"] for r in tl]
    assert valid_froms == sorted(valid_froms)
    assert len(tl) == 3

    tl_limited = kg.timeline(subj, limit=2)
    assert len(tl_limited) == 2


def test_find_similar_entities(kg):
    # Low-threshold pg_trgm similarity across seeded names.
    kg.add_entity("kgtest-Jonathan Smith")
    kg.add_entity("kgtest-Jon Smith")
    kg.add_entity("kgtest-Zachary Black")

    results = kg.find_similar_entities("kgtest-Jon Smith", threshold=0.3)
    names = [r["name"] for r in results]
    assert "kgtest-Jon Smith" in names
    assert "kgtest-Jonathan Smith" in names
    # Unrelated name should rank below the two Smiths.
    top_two = names[:2]
    assert "kgtest-Zachary Black" not in top_two


def test_graph_stats(kg):
    subj = _unique("kgtest-max")
    obj1 = _unique("kgtest-o1")
    obj2 = _unique("kgtest-o2")

    before = kg.graph_stats()
    kg.add_triple(subj, "likes", obj1, valid_from="2025-01-01")
    kg.add_triple(subj, "hates", obj2, valid_from="2025-02-01")
    kg.invalidate(subj, "hates", obj2, ended="2025-03-01")

    after = kg.graph_stats()
    assert after["entities"] >= before["entities"] + 3
    assert after["triples"] >= before["triples"] + 2
    assert "likes" in after["relationship_types"]
    assert "hates" in after["relationship_types"]
    # One invalidated fact should show up as expired.
    assert after["expired_facts"] >= before["expired_facts"] + 1


def test_query_relationship_filters(kg):
    subj = _unique("kgtest-max")
    obj = _unique("kgtest-chess")
    kg.add_triple(subj, "loves", obj, valid_from="2025-10-01")

    by_pred = kg.query_relationship(predicate="loves")
    assert any(r["subject"] == subj and r["object"] == obj for r in by_pred)

    by_sub = kg.query_relationship(subject=subj)
    assert len(by_sub) >= 1

    triple = kg.query_relationship(subject=subj, predicate="loves", obj=obj)
    assert len(triple) == 1


def test_migrate_from_sqlite(tmp_path: Path):
    """End-to-end migration: seed SQLite KG, migrate, verify in PG."""
    # Build a tiny SQLite KG that mirrors the core schema used by MemPalace.
    sqlite_path = tmp_path / "kg.sqlite3"
    conn = sqlite3.connect(sqlite_path)
    conn.executescript(
        """
        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT DEFAULT 'unknown',
            properties TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE triples (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            confidence REAL DEFAULT 1.0,
            source_closet TEXT,
            source_file TEXT,
            source_drawer_id TEXT,
            adapter_name TEXT,
            extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    uid = uuid.uuid4().hex[:8]
    sub_id = f"kgtest-mig-sub-{uid}"
    obj_id = f"kgtest-mig-obj-{uid}"
    triple_id = f"kgtest-mig-triple-{uid}"
    conn.execute(
        "INSERT INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
        (sub_id, "MigSubject", "person", '{"gender":"x"}'),
    )
    conn.execute(
        "INSERT INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
        (obj_id, "MigObject", "thing", "{}"),
    )
    conn.execute(
        """INSERT INTO triples (
            id, subject, predicate, object, valid_from, confidence,
            source_closet, source_file, source_drawer_id, adapter_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            triple_id,
            sub_id,
            "knows",
            obj_id,
            "2024-01-01",
            0.9,
            "closetA",
            "file.md",
            "drawer-1",
            "adapterX",
        ),
    )
    conn.commit()
    conn.close()

    try:
        counts = migrate_kg(sqlite_path, DSN)
        assert counts["entities_read"] == 2
        assert counts["triples_read"] == 1

        with psycopg.connect(DSN) as pgc, pgc.cursor() as cur:
            cur.execute("SELECT id, name, type FROM mp_kg_entities WHERE id = %s", (sub_id,))
            row = cur.fetchone()
            assert row is not None
            assert row[1] == "MigSubject"
            assert row[2] == "person"

            cur.execute(
                "SELECT predicate, source_drawer_id, adapter_name FROM mp_kg_triples WHERE id = %s",
                (triple_id,),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == "knows"
            assert row[1] == "drawer-1"
            assert row[2] == "adapterX"

        # Re-running should be idempotent (no duplicate key errors).
        counts2 = migrate_kg(sqlite_path, DSN)
        assert counts2["entities_read"] == 2
        assert counts2["triples_read"] == 1
    finally:
        with psycopg.connect(DSN) as pgc, pgc.cursor() as cur:
            cur.execute("DELETE FROM mp_kg_triples WHERE id = %s", (triple_id,))
            cur.execute(
                "DELETE FROM mp_kg_entities WHERE id IN (%s, %s)",
                (sub_id, obj_id),
            )
            pgc.commit()
