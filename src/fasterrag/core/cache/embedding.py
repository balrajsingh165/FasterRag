"""Embedding cache: never pay twice for the same text under the same model.

Keyed by content hash plus model plus model version (``docs/architecture.md`` §7). All three
belong in the key: the hash alone would serve a vector from the wrong model, and the model
name alone would serve a stale vector after a provider silently upgraded the weights behind
a stable name. Because the key pins the model version, a stale hit is not merely unlikely —
it is unrepresentable.

Implemented as a wrapper around any ``EmbeddingAdapter`` rather than as a branch inside each
one, so a third-party adapter registered through the entry point gets caching without
knowing the cache exists.

Batches are resolved per text, not per batch: a batch that is nine-tenths cached sends only
the missing tenth to the provider. Vectors are returned in the caller's original order, since
retrieval matches vectors to chunks positionally and a reordered batch would silently attach
every vector to the wrong chunk.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from typing import ClassVar, Final

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.core.cache.stats import CacheStats
from fasterrag.core.cache.store import CacheStore
from fasterrag.errors import CacheError
from fasterrag.observability.logging import get_logger

__all__ = ["CachingEmbeddingAdapter", "decode_vector", "embedding_key", "encode_vector"]

_ELEMENT: Final = struct.Struct("<f")

_logger = get_logger(__name__)


def embedding_key(text: str, model: str, model_version: str) -> str:
    """Return the cache key for one text under one model.

    The text is hashed rather than embedded in the key: keys reach a Redis keyspace and a
    filename, and a document chunk is neither short enough nor safe enough to put in either.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"emb:{model}:{model_version}:{digest}"


def encode_vector(vector: Sequence[float]) -> bytes:
    """Return ``vector`` packed as little-endian float32.

    Not JSON: a 1536-dimension vector is roughly 6 KB packed against 30 KB as text, and the
    cache is sized in entries, so the difference is real disk and real Redis memory. float32
    is what every provider returns and what every vector database stores.
    """
    return b"".join(_ELEMENT.pack(value) for value in vector)


def decode_vector(raw: bytes) -> list[float] | None:
    """Return the vector ``raw`` encodes, or ``None`` if it is not a whole number of floats."""
    if not raw or len(raw) % _ELEMENT.size:
        return None
    return [value for (value,) in _ELEMENT.iter_unpack(raw)]


class CachingEmbeddingAdapter(EmbeddingAdapter):
    """An embedding adapter that consults a cache before its provider."""

    provider: ClassVar[str] = "cached"

    def __init__(self, inner: EmbeddingAdapter, store: CacheStore) -> None:
        """Wrap ``inner``, resolving what it can from ``store`` first.

        Args:
            inner: The adapter that actually embeds.
            store: Where vectors are kept between calls.
        """
        super().__init__(inner.settings)
        self.inner = inner
        self.store = store
        self.stats = CacheStats(name="embedding")

    @property
    def model(self) -> str:
        """Return the wrapped adapter's model."""
        return self.inner.model

    @property
    def model_version(self) -> str:
        """Return the wrapped adapter's model version."""
        return self.inner.model_version

    @property
    def dimensions(self) -> int | None:
        """Return the wrapped adapter's vector size."""
        return self.inner.dimensions

    async def _lookup(self, text: str) -> list[float] | None:
        """Return the cached vector for ``text``, or ``None`` on a miss or a cache failure."""
        key = embedding_key(text, self.model, self.model_version)
        try:
            raw = await self.store.get(key)
        except CacheError as exc:
            self.stats.record_error()
            _logger.warning(
                "embedding cache read failed; embedding directly",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return None

        return decode_vector(raw) if raw is not None else None

    async def _remember(self, text: str, vector: Sequence[float]) -> None:
        """Store ``vector`` for ``text``, treating a cache failure as nothing worth failing for."""
        key = embedding_key(text, self.model, self.model_version)
        try:
            await self.store.set(key, encode_vector(vector))
        except CacheError as exc:
            self.stats.record_error()
            _logger.warning(
                "embedding cache write failed; the vector is still returned",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed ``texts``, sending only the ones the cache could not supply.

        Returns:
            Vectors in the order ``texts`` were given. ``total_tokens`` reflects only what
            the provider actually billed for, so a cached batch reports the tokens it truly
            spent rather than the tokens it would have spent.
        """
        resolved: list[list[float] | None] = []
        for text in texts:
            vector = await self._lookup(text)
            if vector is None:
                self.stats.record_miss()
            else:
                self.stats.record_hit()
            resolved.append(vector)

        missing = [index for index, vector in enumerate(resolved) if vector is None]
        if not missing:
            return EmbeddingResult(
                vectors=[vector for vector in resolved if vector is not None],
                model=self.model,
                model_version=self.model_version,
                total_tokens=0,
            )

        fetched = await self.inner.embed_documents([texts[index] for index in missing])
        for index, vector in zip(missing, fetched.vectors, strict=True):
            resolved[index] = vector
            await self._remember(texts[index], vector)

        return EmbeddingResult(
            vectors=[vector for vector in resolved if vector is not None],
            model=fetched.model,
            model_version=fetched.model_version,
            total_tokens=fetched.total_tokens,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query, consulting the cache first.

        Query vectors are cached alongside document vectors under the same key scheme: a
        repeated question is common, and the model does not care which side asked.
        """
        vector = await self._lookup(text)
        if vector is not None:
            self.stats.record_hit()
            return vector

        self.stats.record_miss()
        vector = await self.inner.embed_query(text)
        await self._remember(text, vector)
        return vector

    async def health(self) -> HealthStatus:
        """Report the wrapped provider's health. A cache is not a dependency."""
        return await self.inner.health()

    async def close(self) -> None:
        """Release the wrapped provider and the cache backend."""
        await self.inner.close()
        await self.store.close()
