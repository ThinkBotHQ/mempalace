"""Local cross-encoder reranker using FlashRank (ONNX, no PyTorch).

FlashRank ships small ONNX-quantized cross-encoder models that run on CPU
with ~50ms latency per query for the MiniLM-L-12 model. This avoids the
~1GB PyTorch dependency while delivering near-SOTA reranking quality.

The ranker is loaded lazily on first use and cached at module scope so
subsequent calls are free of model-load cost.
"""

from __future__ import annotations

from flashrank import Ranker, RerankRequest

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    """Lazy-init the FlashRank cross-encoder (downloads ~30MB on first call)."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=512)
    return _ranker


def rerank(query: str, documents: list[str], top_n: int = 10) -> list[int]:
    """Rerank ``documents`` against ``query`` and return original indices.

    Returns the indices into ``documents`` sorted by descending relevance,
    truncated to ``top_n``. The caller can use the returned indices to
    reorder a parallel list of ids / metadatas.
    """
    if not documents:
        return []
    ranker = _get_ranker()
    passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    return [r["id"] for r in results[:top_n]]
