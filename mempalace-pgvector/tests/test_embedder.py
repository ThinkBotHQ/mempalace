"""Tests for the Embedder protocol and GeminiEmbedder."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from mempalace_pgvector.embedder import Embedder, GeminiEmbedder


def _make_embedding(dim=768):
    return SimpleNamespace(values=[0.1] * dim)


def _mock_embed_response(n=1, dim=768):
    return SimpleNamespace(embeddings=[_make_embedding(dim) for _ in range(n)])


@pytest.fixture
def embedder():
    with patch("mempalace_pgvector.embedder.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.embed_content.return_value = _mock_embed_response(1)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            emb = GeminiEmbedder()
        emb._client = mock_client
        yield emb


def test_protocol_compliance(embedder):
    assert isinstance(embedder, Embedder)


def test_name_and_dimension(embedder):
    assert embedder.name == "gemini-embedding-2"
    assert embedder.dimension == 768


def test_custom_model_and_dimension():
    with patch("mempalace_pgvector.embedder.genai"):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            emb = GeminiEmbedder(model="custom-model", dimension=512)
    assert emb.name == "custom-model"
    assert emb.dimension == 512


def test_embed_adds_doc_prefix(embedder):
    embedder._client.models.embed_content.return_value = _mock_embed_response(2)
    embedder.embed(["hello", "world"])

    call_args = embedder._client.models.embed_content.call_args
    contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
    assert contents[0].startswith("task: retrieval document | ")
    assert contents[1].startswith("task: retrieval document | ")


def test_embed_query_adds_query_prefix(embedder):
    embedder._client.models.embed_content.return_value = _mock_embed_response(1)
    embedder.embed_query(["search term"])

    call_args = embedder._client.models.embed_content.call_args
    contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
    assert contents[0].startswith("task: search query | query: ")


def test_batching(embedder):
    embedder._client.models.embed_content.side_effect = [
        _mock_embed_response(100),
        _mock_embed_response(50),
    ]
    result = embedder.embed(["text"] * 150)
    assert len(result) == 150
    assert embedder._client.models.embed_content.call_count == 2


def test_query_cache(embedder):
    embedder._client.models.embed_content.return_value = _mock_embed_response(1)

    result1 = embedder.embed_query(["cached query"])
    result2 = embedder.embed_query(["cached query"])

    assert result1 == result2
    assert embedder._client.models.embed_content.call_count == 1


def test_no_api_key_raises():
    with patch("mempalace_pgvector.embedder.genai"):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                GeminiEmbedder()


def test_returns_vectors(embedder):
    embedder._client.models.embed_content.return_value = _mock_embed_response(2)
    result = embedder.embed(["a", "b"])
    assert len(result) == 2
    assert len(result[0]) == 768
