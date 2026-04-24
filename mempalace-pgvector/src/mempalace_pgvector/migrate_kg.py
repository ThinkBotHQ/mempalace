"""
migrate_kg.py — Copy a SQLite MemPalace knowledge graph into PostgreSQL.

Reads all rows from the SQLite ``entities`` and ``triples`` tables and bulk-
inserts them into ``mp_kg_entities`` and ``mp_kg_triples``. Safe to re-run:
uses ``ON CONFLICT DO NOTHING`` so existing rows are preserved.

Usage (CLI):
    uv run python -m mempalace_pgvector.migrate_kg \\
        --sqlite ~/.mempalace/knowledge_graph.sqlite3 \\
        --dsn postgresql://postgres:mempalace@localhost:5434/mempalace

Usage (programmatic):
    from mempalace_pgvector.migrate_kg import migrate_kg
    counts = migrate_kg(sqlite_path, dsn)
    print(counts)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .knowledge_graph import PgKnowledgeGraph


logger = logging.getLogger(__name__)

DEFAULT_SQLITE_PATH = os.path.expanduser("~/.mempalace/knowledge_graph.sqlite3")


def _load_properties(raw: Any) -> dict[str, Any]:
    """Decode the SQLite ``properties`` TEXT column into a dict."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def migrate_kg(
    sqlite_path: str | Path,
    dsn: str,
    *,
    batch_size: int = 500,
) -> dict[str, int]:
    """Copy all entities and triples from a SQLite KG to PostgreSQL.

    Returns a dict with counts: ``{"entities_read", "entities_written",
    "triples_read", "triples_written"}``.
    """
    sqlite_path = Path(sqlite_path).expanduser()
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite KG not found: {sqlite_path}")

    # Ensure PG schema exists.
    pg_kg = PgKnowledgeGraph(dsn=dsn)
    pg_kg.close()

    src = sqlite3.connect(str(sqlite_path))
    src.row_factory = sqlite3.Row

    counts = {
        "entities_read": 0,
        "entities_written": 0,
        "triples_read": 0,
        "triples_written": 0,
    }

    with psycopg.connect(dsn) as dst:
        # ── Entities ─────────────────────────────────────────────────
        ent_rows = src.execute(
            "SELECT id, name, type, properties, created_at FROM entities"
        ).fetchall()
        counts["entities_read"] = len(ent_rows)

        with dst.cursor() as cur:
            batch: list[tuple[Any, ...]] = []
            for row in ent_rows:
                batch.append(
                    (
                        row["id"],
                        row["name"],
                        row["type"] or "unknown",
                        Jsonb(_load_properties(row["properties"])),
                    )
                )
                if len(batch) >= batch_size:
                    _flush_entities(cur, batch, counts)
                    batch.clear()
            if batch:
                _flush_entities(cur, batch, counts)

        # ── Triples ──────────────────────────────────────────────────
        tri_rows = src.execute(
            """
            SELECT id, subject, predicate, object,
                   valid_from, valid_to, confidence,
                   source_closet, source_file,
                   source_drawer_id, adapter_name,
                   extracted_at
            FROM triples
            """
        ).fetchall()
        counts["triples_read"] = len(tri_rows)

        with dst.cursor() as cur:
            batch = []
            for row in tri_rows:
                batch.append(
                    (
                        row["id"],
                        row["subject"],
                        row["predicate"],
                        row["object"],
                        row["valid_from"],
                        row["valid_to"],
                        row["confidence"] if row["confidence"] is not None else 1.0,
                        row["source_closet"],
                        row["source_file"],
                        row["source_drawer_id"] if "source_drawer_id" in row.keys() else None,
                        row["adapter_name"] if "adapter_name" in row.keys() else None,
                    )
                )
                if len(batch) >= batch_size:
                    _flush_triples(cur, batch, counts)
                    batch.clear()
            if batch:
                _flush_triples(cur, batch, counts)

        dst.commit()

    src.close()
    return counts


def _flush_entities(
    cur: psycopg.Cursor, batch: list[tuple[Any, ...]], counts: dict[str, int]
) -> None:
    cur.executemany(
        """
        INSERT INTO mp_kg_entities (id, name, type, properties)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        batch,
    )
    counts["entities_written"] += len(batch)


def _flush_triples(
    cur: psycopg.Cursor, batch: list[tuple[Any, ...]], counts: dict[str, int]
) -> None:
    cur.executemany(
        """
        INSERT INTO mp_kg_triples (
            id, subject, predicate, object,
            valid_from, valid_to, confidence,
            source_closet, source_file,
            source_drawer_id, adapter_name
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        batch,
    )
    counts["triples_written"] += len(batch)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mempalace_pgvector.migrate_kg",
        description="Migrate MemPalace knowledge graph from SQLite to PostgreSQL.",
    )
    parser.add_argument(
        "--sqlite",
        default=DEFAULT_SQLITE_PATH,
        help="Path to the SQLite knowledge_graph.sqlite3 file.",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("MEMPALACE_PGVECTOR_DSN")
        or os.environ.get("DATABASE_URL")
        or "postgresql://postgres:mempalace@localhost:5434/mempalace",
        help="PostgreSQL DSN for the destination database.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Migrating KG %s -> %s", args.sqlite, args.dsn)
    counts = migrate_kg(args.sqlite, args.dsn, batch_size=args.batch_size)
    logger.info(
        "Done. entities: read=%d written=%d | triples: read=%d written=%d",
        counts["entities_read"],
        counts["entities_written"],
        counts["triples_read"],
        counts["triples_written"],
    )
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
