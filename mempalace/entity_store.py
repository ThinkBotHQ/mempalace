"""Unified entity store backed by a pgvector collection.

Provides vector-searchable entity resolution with fuzzy matching.
Consolidates entities from the legacy JSON registry, the KG entities
table, and drawer metadata into a single collection that supports
semantic similarity lookups.

Entity collection name: ``mempalace_entities``

Each entity is stored as a document of the form ``"name: description"``
with metadata carrying ``type``, ``aliases`` (JSON array), and
``properties`` (JSON object).  Entity IDs are slugified names
(e.g. ``"alice_smith"``).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from mempalace.backends.base import BaseCollection

logger = logging.getLogger(__name__)

# Collection name used by all EntityStore instances.
ENTITY_COLLECTION_NAME = "mempalace_entities"


def _slugify(name: str) -> str:
    """Convert a display name to a stable entity ID.

    ``"Alice Smith"`` -> ``"alice_smith"``
    ``"O'Brien"``     -> ``"obrien"``
    """
    slug = name.lower().strip()
    slug = slug.replace("'", "").replace("’", "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


class EntityStore:
    """Unified entity store backed by a :class:`BaseCollection`.

    Parameters
    ----------
    collection:
        A ``BaseCollection`` instance (typically a ``PgvectorCollection``)
        pointing at the ``mempalace_entities`` collection.
    embedder:
        An ``Embedder`` instance used to embed entity descriptions for
        vector similarity search.
    kg:
        Optional :class:`KnowledgeGraph` instance.  When provided,
        ``merge_entities`` will update KG triples referencing the
        source entity to point at the target.
    """

    def __init__(
        self,
        collection: BaseCollection,
        embedder: object,
        kg: object = None,
    ) -> None:
        self._collection = collection
        self._embedder = embedder
        self._kg = kg

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        type: str = "unknown",
        description: str = "",
        aliases: Optional[list[str]] = None,
        properties: Optional[dict] = None,
    ) -> str:
        """Add or update an entity.  Returns the entity ID (slug)."""
        entity_id = _slugify(name)
        if not entity_id:
            raise ValueError(f"Cannot slugify name: {name!r}")

        aliases = aliases or []
        properties = properties or {}

        document = f"{name}: {description}" if description else name
        metadata = {
            "type": type,
            "aliases": json.dumps(aliases),
            "properties": json.dumps(properties),
            "name": name,
        }

        self._collection.upsert(
            documents=[document],
            ids=[entity_id],
            metadatas=[metadata],
        )

        logger.debug("entity_store: upserted %s (type=%s)", entity_id, type)
        return entity_id

    # ------------------------------------------------------------------
    # Read / Resolve
    # ------------------------------------------------------------------

    def resolve(
        self,
        query: str,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Fuzzy-resolve a name to matching entities via vector similarity.

        Returns a list of dicts with keys:
        ``id``, ``name``, ``type``, ``description``, ``similarity``.

        Only results whose cosine similarity >= *threshold* are included.
        """
        result = self._collection.query(
            query_texts=[query],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )

        matches: list[dict] = []
        if not result.ids or not result.ids[0]:
            return matches

        ids = result.ids[0]
        docs = result.documents[0] if result.documents else [""] * len(ids)
        metas = result.metadatas[0] if result.metadatas else [{}] * len(ids)
        dists = result.distances[0] if result.distances else [1.0] * len(ids)

        for i, eid in enumerate(ids):
            # distances are cosine distances: similarity = 1 - distance
            similarity = 1.0 - dists[i]
            if similarity < threshold:
                continue
            meta = metas[i] if i < len(metas) else {}
            matches.append(
                {
                    "id": eid,
                    "name": meta.get("name", eid),
                    "type": meta.get("type", "unknown"),
                    "description": docs[i] if i < len(docs) else "",
                    "similarity": round(similarity, 4),
                }
            )

        return matches

    def list_entities(
        self,
        type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List all entities, optionally filtered by type.

        Returns a list of dicts with keys:
        ``id``, ``name``, ``type``, ``description``, ``aliases``, ``properties``.
        """
        if type is not None:
            get_result = self._collection.get(
                where={"type": type},
                limit=limit,
                include=["documents", "metadatas"],
            )
        else:
            get_result = self._collection.get(
                limit=limit,
                include=["documents", "metadatas"],
            )

        entities: list[dict] = []
        for i, eid in enumerate(get_result.ids):
            meta = get_result.metadatas[i] if i < len(get_result.metadatas) else {}
            doc = get_result.documents[i] if i < len(get_result.documents) else ""
            try:
                aliases = json.loads(meta.get("aliases", "[]"))
            except (json.JSONDecodeError, TypeError):
                aliases = []
            try:
                properties = json.loads(meta.get("properties", "{}"))
            except (json.JSONDecodeError, TypeError):
                properties = {}

            entities.append(
                {
                    "id": eid,
                    "name": meta.get("name", eid),
                    "type": meta.get("type", "unknown"),
                    "description": doc,
                    "aliases": aliases,
                    "properties": properties,
                }
            )

        return entities

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_entities(self, source_id: str, target_id: str) -> dict:
        """Merge the source entity into the target.

        1. Fetches both entities.
        2. Merges aliases and properties from source into target.
        3. Deletes the source document.
        4. If a KG is attached, updates all triples referencing source_id
           to point at target_id.

        Returns a dict summarising the merge:
        ``{"target_id", "merged_aliases", "kg_triples_updated"}``.
        """
        # Fetch both
        source_result = self._collection.get(ids=[source_id], include=["documents", "metadatas"])
        target_result = self._collection.get(ids=[target_id], include=["documents", "metadatas"])

        if not source_result.ids:
            raise ValueError(f"Source entity not found: {source_id!r}")
        if not target_result.ids:
            raise ValueError(f"Target entity not found: {target_id!r}")

        source_meta = source_result.metadatas[0]
        target_meta = target_result.metadatas[0]

        # Merge aliases
        try:
            source_aliases = json.loads(source_meta.get("aliases", "[]"))
        except (json.JSONDecodeError, TypeError):
            source_aliases = []
        try:
            target_aliases = json.loads(target_meta.get("aliases", "[]"))
        except (json.JSONDecodeError, TypeError):
            target_aliases = []

        # Add source name as alias if it differs from target name
        source_name = source_meta.get("name", source_id)
        target_name = target_meta.get("name", target_id)
        merged_aliases = list(set(target_aliases + source_aliases))
        if source_name != target_name and source_name not in merged_aliases:
            merged_aliases.append(source_name)

        # Merge properties
        try:
            source_props = json.loads(source_meta.get("properties", "{}"))
        except (json.JSONDecodeError, TypeError):
            source_props = {}
        try:
            target_props = json.loads(target_meta.get("properties", "{}"))
        except (json.JSONDecodeError, TypeError):
            target_props = {}
        merged_props = {**source_props, **target_props}

        # Update target with merged data
        target_meta_new = dict(target_meta)
        target_meta_new["aliases"] = json.dumps(merged_aliases)
        target_meta_new["properties"] = json.dumps(merged_props)

        self._collection.upsert(
            documents=[target_result.documents[0]],
            ids=[target_id],
            metadatas=[target_meta_new],
        )

        # Delete source
        self._collection.delete(ids=[source_id])

        # Update KG triples if KG is available
        kg_triples_updated = 0
        if self._kg is not None:
            kg_triples_updated = self._update_kg_references(source_id, target_id)

        logger.info(
            "entity_store: merged %s -> %s (aliases=%s, kg_triples=%d)",
            source_id,
            target_id,
            merged_aliases,
            kg_triples_updated,
        )
        return {
            "target_id": target_id,
            "merged_aliases": merged_aliases,
            "kg_triples_updated": kg_triples_updated,
        }

    def _update_kg_references(self, source_id: str, target_id: str) -> int:
        """Update all KG triples that reference *source_id* to point at *target_id*.

        Works with the KnowledgeGraph SQLite schema directly.
        Returns the number of triples updated.
        """
        kg = self._kg
        count = 0
        try:
            conn = kg._conn()
            with kg._lock:
                with conn:
                    # Update subject references
                    cur = conn.execute(
                        "UPDATE triples SET subject = ? WHERE subject = ?",
                        (target_id, source_id),
                    )
                    count += cur.rowcount

                    # Update object references
                    cur = conn.execute(
                        "UPDATE triples SET object = ? WHERE object = ?",
                        (target_id, source_id),
                    )
                    count += cur.rowcount

                    # Merge entity rows: delete source, ensure target exists
                    conn.execute(
                        "DELETE FROM entities WHERE id = ?",
                        (source_id,),
                    )
        except Exception:
            logger.exception("entity_store: failed to update KG references")
        return count

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill_from_registry(self, registry_path: str) -> dict:
        """Import entities from the existing JSON registry file.

        Reads ``~/.mempalace/entity_registry.json`` (or the given path),
        and creates entity documents for every person and project.

        Returns ``{"imported": int, "skipped": int, "errors": list[str]}``.
        """
        path = Path(registry_path)
        if not path.exists():
            return {"imported": 0, "skipped": 0, "errors": [f"File not found: {registry_path}"]}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"imported": 0, "skipped": 0, "errors": [str(exc)]}

        imported = 0
        skipped = 0
        errors: list[str] = []

        # People
        people = data.get("people", {})
        for name, info in people.items():
            try:
                # Skip alias entries that just point at a canonical name
                if info.get("canonical"):
                    skipped += 1
                    continue

                description_parts = []
                relationship = info.get("relationship", "")
                if relationship:
                    description_parts.append(f"relationship: {relationship}")
                contexts = info.get("contexts", [])
                if contexts:
                    description_parts.append(f"context: {', '.join(contexts)}")
                source = info.get("source", "")
                if source:
                    description_parts.append(f"source: {source}")
                description = "; ".join(description_parts)

                self.add_entity(
                    name=name,
                    type="person",
                    description=description,
                    aliases=info.get("aliases", []),
                    properties={
                        "confidence": info.get("confidence", 1.0),
                        "source": source,
                        "relationship": relationship,
                    },
                )
                imported += 1
            except Exception as exc:
                errors.append(f"Failed to import person {name!r}: {exc}")

        # Projects
        projects = data.get("projects", [])
        for proj in projects:
            try:
                self.add_entity(
                    name=proj,
                    type="project",
                    description=f"project: {proj}",
                )
                imported += 1
            except Exception as exc:
                errors.append(f"Failed to import project {proj!r}: {exc}")

        logger.info(
            "entity_store: backfilled from registry — imported=%d skipped=%d errors=%d",
            imported,
            skipped,
            len(errors),
        )
        return {"imported": imported, "skipped": skipped, "errors": errors}

    def backfill_from_kg(self, kg: object) -> dict:
        """Import entities from the KG entities table.

        Reads all rows from ``entities`` and creates corresponding
        entity documents in the vector collection.

        Returns ``{"imported": int, "skipped": int, "errors": list[str]}``.
        """
        imported = 0
        skipped = 0
        errors: list[str] = []

        try:
            conn = kg._conn()
            with kg._lock:
                rows = conn.execute("SELECT id, name, type, properties FROM entities").fetchall()
        except Exception as exc:
            return {"imported": 0, "skipped": 0, "errors": [f"KG read failed: {exc}"]}

        for row in rows:
            try:
                name = row["name"] if isinstance(row, dict) else row[1]
                entity_type = row["type"] if isinstance(row, dict) else row[2]
                props_raw = row["properties"] if isinstance(row, dict) else row[3]

                try:
                    properties = json.loads(props_raw) if props_raw else {}
                except (json.JSONDecodeError, TypeError):
                    properties = {}

                description_parts = []
                if entity_type and entity_type != "unknown":
                    description_parts.append(f"type: {entity_type}")
                for key, value in properties.items():
                    if value:
                        description_parts.append(f"{key}: {value}")
                description = "; ".join(description_parts)

                self.add_entity(
                    name=name,
                    type=entity_type or "unknown",
                    description=description,
                    properties=properties,
                )
                imported += 1
            except Exception as exc:
                entity_name = row[1] if not isinstance(row, dict) else row.get("name", "?")
                errors.append(f"Failed to import KG entity {entity_name!r}: {exc}")

        logger.info(
            "entity_store: backfilled from KG — imported=%d skipped=%d errors=%d",
            imported,
            skipped,
            len(errors),
        )
        return {"imported": imported, "skipped": skipped, "errors": errors}
