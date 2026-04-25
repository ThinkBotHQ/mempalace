"""Integration tests for hybrid (vector + FTS) search and FlashRank reranker.

Runs against the live pgvector container at localhost:5434, mirroring the
fixture style of test_collection.py.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from mempalace_pgvector.collection import PgvectorCollection
from mempalace_pgvector.reranker import rerank


DSN = "postgresql://postgres:mempalace@localhost:5434/mempalace"
DIM = 4


class MockEmbedder:
    """Deterministic embedder. Returns the embedding the caller stored
    explicitly when possible; otherwise a fixed query vector."""

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


def test_hybrid_query_returns_results(collection):
    """Smoke test: hybrid_query returns a populated QueryResult."""
    collection.add(
        documents=[
            "the quick brown fox jumps over the lazy dog",
            "machine learning models embed text into vectors",
            "postgres is a relational database",
        ],
        ids=["h1", "h2", "h3"],
        embeddings=[[0.5] * DIM, [0.4] * DIM, [0.0] * DIM],
    )

    result = collection.hybrid_query(
        query_text="quick fox",
        query_embedding=[0.5] * DIM,
        n_results=10,
    )

    assert len(result.ids) == 1
    assert len(result.ids[0]) >= 1
    # h1 should rank highly: it has the closest embedding AND keyword match.
    assert result.ids[0][0] == "h1"


def test_hybrid_text_only_match_surfaces_via_fts(collection):
    """A document with poor vector similarity but strong keyword overlap
    must still surface through the FTS branch of RRF."""
    collection.add(
        documents=[
            # Strong keyword overlap ("zebra"), bad vector match (far from query 0.5).
            "a wild zebra galloped across the savanna",
            # Decent vector, no keyword overlap.
            "generic content about clouds and weather patterns",
            "more generic filler text without any relevant terms",
        ],
        ids=["t1", "t2", "t3"],
        embeddings=[[0.0] * DIM, [0.5] * DIM, [0.5] * DIM],
    )

    result = collection.hybrid_query(
        query_text="zebra",
        query_embedding=[0.5] * DIM,
        n_results=10,
    )

    # t1 has zero vector similarity to the query but should still appear
    # because the FTS branch ranks it #1 for "zebra".
    assert "t1" in result.ids[0]


def test_hybrid_vector_only_match_surfaces_via_embedding(collection):
    """A document with no keyword overlap must still surface through the
    vector branch of RRF when its embedding is closest to the query."""
    collection.add(
        documents=[
            "completely unrelated lexical content here",
            "another bag of words with no overlap to the query",
            "a third unrelated document for padding",
        ],
        ids=["v1", "v2", "v3"],
        embeddings=[[0.5] * DIM, [0.0] * DIM, [-0.5] * DIM],
    )

    result = collection.hybrid_query(
        # Query text that does NOT match any document via FTS.
        query_text="xylophone",
        query_embedding=[0.5] * DIM,
        n_results=10,
    )

    # v1 has the closest embedding; FTS returns nothing; RRF should still
    # surface v1 from the vector branch alone.
    assert "v1" in result.ids[0]
    assert result.ids[0][0] == "v1"


def test_hybrid_query_respects_where_filter(collection):
    """Metadata filters apply to both branches of the RRF query."""
    collection.add(
        documents=[
            "alpha keyword document one",
            "alpha keyword document two",
        ],
        ids=["f1", "f2"],
        metadatas=[{"wing": "include"}, {"wing": "exclude"}],
        embeddings=[[0.5] * DIM, [0.5] * DIM],
    )

    result = collection.hybrid_query(
        query_text="alpha",
        query_embedding=[0.5] * DIM,
        n_results=10,
        where={"wing": "include"},
    )

    assert result.ids[0] == ["f1"]


def test_flashrank_reranker_reorders(collection):
    """FlashRank should pull the topically-relevant doc to the top even
    when it's not first in the input list."""
    docs = [
        "the price of tea in china has been rising",
        "what is the best programming language for machine learning",
        "a recipe for chocolate chip cookies",
    ]
    indices = rerank(
        query="which programming language should I use for ML?",
        documents=docs,
        top_n=3,
    )
    # Top result must be the ML programming doc (index 1).
    assert indices[0] == 1
    assert len(indices) == 3
    assert sorted(indices) == [0, 1, 2]


def test_flashrank_reranker_top_n(collection):
    """top_n bounds the returned list."""
    docs = ["doc one", "doc two", "doc three", "doc four"]
    indices = rerank(query="anything", documents=docs, top_n=2)
    assert len(indices) == 2


def test_flashrank_empty_documents():
    """Empty input must not blow up."""
    assert rerank(query="anything", documents=[], top_n=10) == []
