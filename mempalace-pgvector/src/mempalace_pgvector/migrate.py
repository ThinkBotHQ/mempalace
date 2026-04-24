"""Migrate MemPalace data from ChromaDB to PostgreSQL + pgvector.

Usage:
    uv run python -m mempalace_pgvector.migrate \
        --chroma-path ~/.mempalace/palace \
        --collection mempalace_drawers \
        --pgvector-dsn postgresql://postgres:mempalace@localhost:5434/mempalace \
        --batch-size 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import psycopg

from mempalace.backends.base import PalaceRef

from .backend import PgvectorBackend
from .embedder import GeminiEmbedder

logger = logging.getLogger(__name__)

CURSOR_FILE = Path.home() / ".mempalace" / "pgvector_migrate_cursor.json"


def _load_cursor() -> dict:
    if CURSOR_FILE.is_file():
        return json.loads(CURSOR_FILE.read_text())
    return {"last_offset": 0, "total_migrated": 0}


def _save_cursor(cursor: dict) -> None:
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_FILE.write_text(json.dumps(cursor))


def migrate(
    chroma_path: str,
    collection_name: str,
    pgvector_dsn: str,
    batch_size: int = 100,
    palace_id: str | None = None,
    dry_run: bool = False,
) -> dict:
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        chroma_coll = client.get_collection(collection_name)
    except Exception as e:
        logger.error("Failed to open ChromaDB collection %r: %s", collection_name, e)
        return {"error": str(e)}

    total = chroma_coll.count()
    logger.info("ChromaDB collection %r has %d documents", collection_name, total)

    if dry_run:
        logger.info("Dry run — would migrate %d documents", total)
        return {"total": total, "dry_run": True}

    resolved_palace_id = palace_id or Path(chroma_path).name or "default"
    embedder = GeminiEmbedder()
    backend = PgvectorBackend(dsn=pgvector_dsn, embedder=embedder)
    palace = PalaceRef(id=resolved_palace_id, local_path=chroma_path)
    pg_coll = backend.get_collection(
        palace=palace, collection_name=collection_name, create=True
    )

    cursor = _load_cursor()
    offset = cursor["last_offset"]
    migrated = cursor["total_migrated"]

    logger.info("Resuming from offset %d (%d previously migrated)", offset, migrated)

    while offset < total:
        batch = chroma_coll.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        if not batch["ids"]:
            break

        ids = batch["ids"]
        docs = batch["documents"]
        metas = batch["metadatas"]

        logger.info(
            "Batch %d-%d of %d: embedding %d documents...",
            offset, offset + len(ids), total, len(ids),
        )

        start = time.monotonic()
        embeddings = embedder.embed(docs)
        embed_time = time.monotonic() - start

        pg_coll.upsert(
            documents=docs,
            ids=ids,
            metadatas=metas,
            embeddings=embeddings,
        )

        offset += len(ids)
        migrated += len(ids)

        _save_cursor({"last_offset": offset, "total_migrated": migrated})
        logger.info(
            "  Migrated %d/%d (%.1fs embed, %.1f docs/sec)",
            migrated, total, embed_time, len(ids) / max(embed_time, 0.001),
        )

    backend.close()
    CURSOR_FILE.unlink(missing_ok=True)

    logger.info("Migration complete: %d documents migrated", migrated)
    return {"total": total, "migrated": migrated}


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB → pgvector")
    parser.add_argument("--chroma-path", required=True, help="Path to ChromaDB palace directory")
    parser.add_argument("--collection", required=True, help="ChromaDB collection name")
    parser.add_argument("--pgvector-dsn", required=True, help="PostgreSQL DSN")
    parser.add_argument("--batch-size", type=int, default=100, help="Documents per batch")
    parser.add_argument("--palace-id", default=None, help="Palace ID (default: dir name)")
    parser.add_argument("--dry-run", action="store_true", help="Count documents without migrating")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = migrate(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        pgvector_dsn=args.pgvector_dsn,
        batch_size=args.batch_size,
        palace_id=args.palace_id,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2))
    sys.exit(0 if "error" not in result else 1)


if __name__ == "__main__":
    main()
