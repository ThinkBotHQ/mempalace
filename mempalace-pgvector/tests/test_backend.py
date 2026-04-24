"""Integration tests for PgvectorBackend against a real PostgreSQL instance."""

import pytest

from mempalace.backends.base import EmbedderIdentityMismatchError, PalaceNotFoundError, PalaceRef
from mempalace_pgvector.backend import PgvectorBackend


DSN = "postgresql://postgres:mempalace@localhost:5434/mempalace"
DIM = 4


class MockEmbedder:
    @property
    def name(self) -> str:
        return "mock-embedder"

    @property
    def dimension(self) -> int:
        return DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * DIM for _ in texts]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * DIM for _ in texts]


@pytest.fixture
def backend():
    b = PgvectorBackend(dsn=DSN, embedder=MockEmbedder(), pool_max=2)
    yield b
    b.close()


@pytest.fixture
def palace():
    return PalaceRef(id="test-backend-palace")


def test_create_and_get_collection(backend, palace):
    coll = backend.get_collection(palace=palace, collection_name="test-coll-1", create=True)
    coll.add(documents=["test doc"], ids=["t1"], embeddings=[[0.1] * DIM])
    assert coll.count() == 1

    coll2 = backend.get_collection(palace=palace, collection_name="test-coll-1")
    assert coll2.count() == 1

    coll.delete(ids=["t1"])


def test_get_nonexistent_raises(backend, palace):
    with pytest.raises(PalaceNotFoundError):
        backend.get_collection(palace=palace, collection_name="does-not-exist-ever")


def test_embedder_mismatch_raises(backend, palace):
    backend.get_collection(palace=palace, collection_name="test-coll-mismatch", create=True)
    backend._collections.clear()

    class OtherEmbedder(MockEmbedder):
        @property
        def name(self) -> str:
            return "different-model"

    backend2 = PgvectorBackend(dsn=DSN, embedder=OtherEmbedder(), pool_max=1)
    with pytest.raises(EmbedderIdentityMismatchError, match="mock-embedder"):
        backend2.get_collection(palace=palace, collection_name="test-coll-mismatch")
    backend2.close()


def test_health(backend, palace):
    status = backend.health()
    assert status.ok is True


def test_close_palace(backend, palace):
    backend.get_collection(palace=palace, collection_name="test-coll-close", create=True)
    assert any("test-coll-close" in k for k in backend._collections)
    backend.close_palace(palace)
    assert not any("test-coll-close" in k for k in backend._collections)
