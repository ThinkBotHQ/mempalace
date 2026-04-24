"""Embedder protocol and Gemini implementation for mempalace-pgvector."""

from __future__ import annotations

import functools
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol, runtime_checkable

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_PARALLEL_WORKERS = 15
_DOC_PREFIX = "task: retrieval document | "
_QUERY_PREFIX = "task: search query | query: "


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    def __init__(
        self,
        model: str = "gemini-embedding-2",
        dimension: int = 768,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._dimension = dimension

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key: set GEMINI_API_KEY or GOOGLE_API_KEY, "
                "or pass api_key= to GeminiEmbedder"
            )

        self._client = genai.Client(api_key=resolved_key)

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_DOC_PREFIX + t for t in texts]
        return self._embed_batched(prefixed)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        key = tuple(texts)
        cached = self._query_cache_get(key)
        if cached is not None:
            return cached
        prefixed = [_QUERY_PREFIX + t for t in texts]
        result = self._embed_batched(prefixed)
        self._query_cache_put(key, result)
        return result

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= 1:
            return [self._call_api(t).embeddings[0].values for t in texts]

        results: list[tuple[int, list[float]]] = []
        with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as pool:
            futures = {
                pool.submit(self._call_api, text): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results.append((idx, result.embeddings[0].values))

        results.sort(key=lambda x: x[0])
        return [vec for _, vec in results]

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=lambda rs: logger.warning(
            "Gemini embed retry %d: %s", rs.attempt_number, rs.outcome.exception()
        ),
    )
    def _call_api(self, text: str):
        return self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self._dimension,
            ),
        )

    _CACHE_MAX = 256

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @functools.cached_property
    def _query_cache(self) -> dict:
        return {}

    def _query_cache_get(self, key: tuple) -> list[list[float]] | None:
        return self._query_cache.get(key)

    def _query_cache_put(self, key: tuple, value: list[list[float]]) -> None:
        if len(self._query_cache) >= self._CACHE_MAX:
            oldest = next(iter(self._query_cache))
            del self._query_cache[oldest]
        self._query_cache[key] = value
