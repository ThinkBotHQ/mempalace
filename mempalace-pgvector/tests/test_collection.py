"""Integration tests for PgvectorCollection against a real PostgreSQL instance."""

import uuid
from pathlib import Path

import psycopg
import pytest

from mempalace_pgvector.collection import PgvectorCollection


DSN = "postgresql://postgres:mempalace@localhost:5434/mempalace"
SCHEMA_PATH = Path(__file__).parent.parent / "src" / "mempalace_pgvector" / "schema.sql"
DIM = 4


class MockEmbedder:
    @property
    def name(self) -> str:
        return "mock-embedder"

    @property
    def dimension(self) -> int:
        return DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i) / 10.0] * DIM for i in range(len(texts))]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * DIM for _ in texts]


@pytest.fixture
def collection():
    conn = psycopg.connect(DSN, autocommit=True)
    coll_id = uuid.uuid4()

    conn.execute(
        """
        INSERT INTO mp_collections (id, palace_id, collection_name, embedder_name, embedder_dim)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (str(coll_id), "test-palace", f"test-{coll_id}", "mock-embedder", DIM),
    )

    coll = PgvectorCollection(conn=conn, collection_id=coll_id, embedder=MockEmbedder())
    yield coll

    conn.execute("DELETE FROM mp_documents WHERE collection_id = %s", (str(coll_id),))
    conn.execute("DELETE FROM mp_collections WHERE id = %s", (str(coll_id),))
    conn.close()


def test_add_and_get(collection):
    collection.add(
        documents=["hello world", "goodbye world"],
        ids=["d1", "d2"],
        metadatas=[{"wing": "test"}, {"wing": "test"}],
        embeddings=[[0.1] * DIM, [0.9] * DIM],
    )
    result = collection.get(ids=["d1"])
    assert result.ids == ["d1"]
    assert result.documents == ["hello world"]


def test_add_and_count(collection):
    collection.add(
        documents=["a", "b", "c"],
        ids=["c1", "c2", "c3"],
        embeddings=[[0.1] * DIM, [0.5] * DIM, [0.9] * DIM],
    )
    assert collection.count() == 3


def test_upsert_overwrites(collection):
    collection.add(
        documents=["original"],
        ids=["u1"],
        embeddings=[[0.1] * DIM],
    )
    collection.upsert(
        documents=["updated"],
        ids=["u1"],
        embeddings=[[0.2] * DIM],
    )
    result = collection.get(ids=["u1"])
    assert result.documents == ["updated"]
    assert collection.count() == 1


def test_delete_by_id(collection):
    collection.add(
        documents=["keep", "remove"],
        ids=["k1", "r1"],
        embeddings=[[0.1] * DIM, [0.9] * DIM],
    )
    collection.delete(ids=["r1"])
    assert collection.count() == 1
    result = collection.get(ids=["r1"])
    assert result.ids == []


def test_query_returns_ranked(collection):
    collection.add(
        documents=["close match", "far match"],
        ids=["q1", "q2"],
        embeddings=[[0.5] * DIM, [0.0] * DIM],
    )
    result = collection.query(
        query_embeddings=[[0.5] * DIM],
        n_results=2,
    )
    assert len(result.ids) == 1
    assert len(result.ids[0]) == 2
    assert result.ids[0][0] == "q1"


def test_get_with_where(collection):
    collection.add(
        documents=["alpha", "beta"],
        ids=["w1", "w2"],
        metadatas=[{"wing": "people"}, {"wing": "projects"}],
        embeddings=[[0.1] * DIM, [0.9] * DIM],
    )
    result = collection.get(where={"wing": "projects"})
    assert result.ids == ["w2"]


def test_health(collection):
    status = collection.health()
    assert status.ok is True


def test_auto_embed(collection):
    collection.add(
        documents=["auto embedded doc"],
        ids=["ae1"],
    )
    assert collection.count() == 1
