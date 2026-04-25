"""Integration tests for EntityStore against a live pgvector instance."""

import json
import tempfile
import uuid
from pathlib import Path

import psycopg
import pytest

from mempalace.entity_store import EntityStore, _slugify
from mempalace_pgvector.collection import PgvectorCollection

DSN = "postgresql://postgres:mempalace@localhost:5434/mempalace"
DIM = 4


# ---------------------------------------------------------------------------
# MockEmbedder — deterministic 4-dim embeddings for test reproducibility
# ---------------------------------------------------------------------------


class MockEmbedder:
    """Embedder that produces deterministic vectors.

    Documents get embeddings proportional to their index position;
    queries always get [0.5, 0.5, 0.5, 0.5] so the first document
    (index 0 -> [0.0]*4) is farthest and the middle-range ones are
    closest by cosine.

    For entity resolution tests we embed the document *text* into a
    stable vector so that identical or similar texts produce the same
    embedding.  The simple hash approach here ensures that
    ``add_entity("Alice Smith")`` followed by ``resolve("Alice Smith")``
    yields high similarity because both pass through ``embed`` / ``embed_query``
    with the same text and thus get very similar vectors.
    """

    @property
    def name(self) -> str:
        return "mock-embedder"

    @property
    def dimension(self) -> int:
        return DIM

    def _text_to_vec(self, text: str) -> list[float]:
        """Hash-based embedding: same text -> same vector."""
        h = hash(text) & 0xFFFFFFFF
        return [
            ((h >> 0) & 0xFF) / 255.0,
            ((h >> 8) & 0xFF) / 255.0,
            ((h >> 16) & 0xFF) / 255.0,
            ((h >> 24) & 0xFF) / 255.0,
        ]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def entity_store():
    """Create a temporary entity collection and EntityStore, clean up after."""
    conn = psycopg.connect(DSN, autocommit=True)
    coll_id = uuid.uuid4()
    coll_name = f"test-entities-{coll_id}"

    conn.execute(
        """
        INSERT INTO mp_collections (id, palace_id, collection_name, embedder_name, embedder_dim)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (str(coll_id), "test-palace", coll_name, "mock-embedder", DIM),
    )

    embedder = MockEmbedder()
    collection = PgvectorCollection(conn=conn, collection_id=coll_id, embedder=embedder)
    store = EntityStore(collection=collection, embedder=embedder)
    yield store

    conn.execute("DELETE FROM mp_documents WHERE collection_id = %s", (str(coll_id),))
    conn.execute("DELETE FROM mp_collections WHERE id = %s", (str(coll_id),))
    conn.close()


@pytest.fixture
def entity_store_with_kg(tmp_path):
    """EntityStore with a real SQLite KG for merge tests."""
    from mempalace.knowledge_graph import KnowledgeGraph

    kg_path = str(tmp_path / "test_kg.sqlite3")
    kg = KnowledgeGraph(db_path=kg_path)

    conn = psycopg.connect(DSN, autocommit=True)
    coll_id = uuid.uuid4()
    coll_name = f"test-entities-kg-{coll_id}"

    conn.execute(
        """
        INSERT INTO mp_collections (id, palace_id, collection_name, embedder_name, embedder_dim)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (str(coll_id), "test-palace", coll_name, "mock-embedder", DIM),
    )

    embedder = MockEmbedder()
    collection = PgvectorCollection(conn=conn, collection_id=coll_id, embedder=embedder)
    store = EntityStore(collection=collection, embedder=embedder, kg=kg)
    yield store, kg

    conn.execute("DELETE FROM mp_documents WHERE collection_id = %s", (str(coll_id),))
    conn.execute("DELETE FROM mp_collections WHERE id = %s", (str(coll_id),))
    conn.close()
    kg.close()


# ---------------------------------------------------------------------------
# Unit tests for _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_simple_name(self):
        assert _slugify("Alice Smith") == "alice_smith"

    def test_apostrophe(self):
        assert _slugify("O'Brien") == "obrien"

    def test_special_chars(self):
        assert _slugify("Jean-Pierre") == "jean_pierre"

    def test_extra_spaces(self):
        assert _slugify("  Bob  ") == "bob"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAddAndResolve:
    def test_add_and_resolve_exact(self, entity_store):
        """Add an entity and resolve it by its exact name."""
        eid = entity_store.add_entity(
            name="Alice Smith",
            type="person",
            description="Justin's wife and project partner",
        )
        assert eid == "alice_smith"

        # Resolve using the exact document text should yield high similarity
        matches = entity_store.resolve("Alice Smith: Justin's wife and project partner")
        assert len(matches) >= 1
        assert matches[0]["id"] == "alice_smith"
        assert matches[0]["type"] == "person"
        assert matches[0]["similarity"] > 0.0

    def test_add_returns_id(self, entity_store):
        """add_entity returns a stable slug ID."""
        eid = entity_store.add_entity(name="Bob Jones", type="person")
        assert eid == "bob_jones"

    def test_upsert_updates(self, entity_store):
        """Adding the same entity twice updates rather than duplicating."""
        entity_store.add_entity(name="Charlie", type="person", description="v1")
        entity_store.add_entity(name="Charlie", type="person", description="v2")

        entities = entity_store.list_entities()
        charlie_entries = [e for e in entities if e["id"] == "charlie"]
        assert len(charlie_entries) == 1
        assert "v2" in charlie_entries[0]["description"]


class TestFuzzyResolve:
    def test_fuzzy_resolve_partial(self, entity_store):
        """Resolve with a partial/related query returns the entity."""
        entity_store.add_entity(
            name="Alice Smith",
            type="person",
            description="software engineer and co-founder",
        )
        # Even a loosely related query should return results with threshold=0.0
        matches = entity_store.resolve("Alice Smith", threshold=0.0)
        assert len(matches) >= 1
        # The top match should be Alice
        names = [m["name"] for m in matches]
        assert "Alice Smith" in names

    def test_no_match_above_threshold(self, entity_store):
        """An unrelated query with high threshold returns empty."""
        entity_store.add_entity(name="Zebra Corp", type="project", description="fintech startup")
        # With a very high threshold, an unrelated query should return nothing
        matches = entity_store.resolve("quantum physics research", threshold=0.99)
        assert len(matches) == 0


class TestListEntities:
    def test_list_all(self, entity_store):
        """List all entities without type filter."""
        entity_store.add_entity(name="Alice", type="person")
        entity_store.add_entity(name="MemPalace", type="project")
        entity_store.add_entity(name="Python", type="concept")

        entities = entity_store.list_entities()
        assert len(entities) == 3
        names = {e["name"] for e in entities}
        assert names == {"Alice", "MemPalace", "Python"}

    def test_list_with_type_filter(self, entity_store):
        """List entities filtered by type."""
        entity_store.add_entity(name="Alice", type="person")
        entity_store.add_entity(name="Bob", type="person")
        entity_store.add_entity(name="MemPalace", type="project")

        people = entity_store.list_entities(type="person")
        assert len(people) == 2
        names = {e["name"] for e in people}
        assert names == {"Alice", "Bob"}

    def test_list_with_aliases_and_properties(self, entity_store):
        """Aliases and properties are returned correctly."""
        entity_store.add_entity(
            name="Maxwell",
            type="person",
            aliases=["Max"],
            properties={"birthday": "2015-04-01"},
        )
        entities = entity_store.list_entities()
        assert len(entities) == 1
        assert entities[0]["aliases"] == ["Max"]
        assert entities[0]["properties"] == {"birthday": "2015-04-01"}


class TestBackfillFromKG:
    def test_backfill_imports_entities(self, entity_store_with_kg):
        """Backfill from KG creates entity documents for each KG entity."""
        store, kg = entity_store_with_kg

        kg.add_entity("Alice", entity_type="person", properties={"birthday": "1990-01-15"})
        kg.add_entity("MemPalace", entity_type="project")
        kg.add_entity("Chess", entity_type="concept")

        result = store.backfill_from_kg(kg)
        assert result["imported"] == 3
        assert result["errors"] == []

        entities = store.list_entities()
        assert len(entities) == 3
        names = {e["name"] for e in entities}
        assert "Alice" in names
        assert "MemPalace" in names
        assert "Chess" in names

    def test_backfill_preserves_properties(self, entity_store_with_kg):
        """Properties from KG entities are preserved during backfill."""
        store, kg = entity_store_with_kg

        kg.add_entity("Alice", entity_type="person", properties={"birthday": "1990-01-15"})
        store.backfill_from_kg(kg)

        entities = store.list_entities()
        alice = [e for e in entities if e["name"] == "Alice"][0]
        assert alice["properties"]["birthday"] == "1990-01-15"


class TestBackfillFromRegistry:
    def test_backfill_from_registry_file(self, entity_store):
        """Backfill reads a JSON registry and imports people + projects."""
        registry_data = {
            "version": 1,
            "mode": "personal",
            "people": {
                "Riley": {
                    "source": "onboarding",
                    "contexts": ["personal"],
                    "aliases": [],
                    "relationship": "daughter",
                    "confidence": 1.0,
                },
                "Max": {
                    "source": "onboarding",
                    "contexts": ["personal"],
                    "aliases": ["Maxwell"],
                    "relationship": "son",
                    "confidence": 1.0,
                },
                "Maxwell": {
                    "source": "onboarding",
                    "contexts": ["personal"],
                    "aliases": ["Max"],
                    "relationship": "son",
                    "confidence": 1.0,
                    "canonical": "Max",
                },
            },
            "projects": ["MemPalace", "Acme"],
            "ambiguous_flags": ["max"],
            "wiki_cache": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(registry_data, f)
            registry_path = f.name

        try:
            result = entity_store.backfill_from_registry(registry_path)
            # Riley + Max imported; Maxwell skipped (canonical alias)
            assert result["imported"] == 4  # 2 people + 2 projects
            assert result["skipped"] == 1  # Maxwell (canonical)
            assert result["errors"] == []

            entities = entity_store.list_entities()
            names = {e["name"] for e in entities}
            assert "Riley" in names
            assert "Max" in names
            assert "MemPalace" in names
            assert "Acme" in names
            # Maxwell should NOT be imported (it's an alias entry)
            assert "Maxwell" not in names
        finally:
            Path(registry_path).unlink(missing_ok=True)

    def test_backfill_missing_file(self, entity_store):
        """Backfill from a nonexistent file returns an error."""
        result = entity_store.backfill_from_registry("/nonexistent/path.json")
        assert result["imported"] == 0
        assert len(result["errors"]) == 1


class TestMerge:
    def test_merge_entities(self, entity_store):
        """Merge source into target: source deleted, aliases merged."""
        entity_store.add_entity(name="Bob", type="person", description="colleague")
        entity_store.add_entity(name="Robert", type="person", description="Bob's full name")

        result = entity_store.merge_entities("bob", "robert")
        assert result["target_id"] == "robert"
        assert "Bob" in result["merged_aliases"]

        # Source should be gone
        entities = entity_store.list_entities()
        ids = {e["id"] for e in entities}
        assert "bob" not in ids
        assert "robert" in ids

    def test_merge_updates_kg(self, entity_store_with_kg):
        """Merge updates KG triples to point at the target entity."""
        store, kg = entity_store_with_kg

        # Create entities in both stores
        store.add_entity(name="Bob", type="person")
        store.add_entity(name="Robert", type="person")

        # Create KG triples referencing "bob"
        kg.add_entity("Bob", entity_type="person")
        kg.add_entity("Robert", entity_type="person")
        kg.add_entity("Chess", entity_type="concept")
        kg.add_triple("Bob", "loves", "Chess")

        result = store.merge_entities("bob", "robert")
        assert result["kg_triples_updated"] >= 1

        # Verify the triple now references "robert" as subject
        conn = kg._conn()
        rows = conn.execute(
            "SELECT subject FROM triples WHERE predicate = ?", ("loves",)
        ).fetchall()
        subjects = [r[0] if not isinstance(r, dict) else r["subject"] for r in rows]
        assert "robert" in subjects
        assert "bob" not in subjects

    def test_merge_nonexistent_source_raises(self, entity_store):
        """Merging a nonexistent source entity raises ValueError."""
        entity_store.add_entity(name="Target", type="person")
        with pytest.raises(ValueError, match="Source entity not found"):
            entity_store.merge_entities("nonexistent", "target")

    def test_merge_nonexistent_target_raises(self, entity_store):
        """Merging into a nonexistent target entity raises ValueError."""
        entity_store.add_entity(name="Source", type="person")
        with pytest.raises(ValueError, match="Target entity not found"):
            entity_store.merge_entities("source", "nonexistent")
