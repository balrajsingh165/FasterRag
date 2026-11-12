"""Embedding cache: never pay twice for the same text under the same model.

Keyed by everything that decides what vector a text gets (``docs/architecture.md`` §7): the
provider, the model, the model version, the configured ``embeddings.dimensions``, whether
the text is embedded as a document or as a query, and the text itself. Each one belongs in
the key. Without the model version a provider that upgraded the weights behind a stable
name would be served stale vectors. Without the provider two providers serving a model of
the same name would share entries. Without the dimensions a vector shortened to 256 would
be served after the setting changed to 512. Without the input type a query would be served
a passage's vector, and several providers embed the two differently on purpose (Cohere's
``search_query`` and ``search_document``). Because the key pins all of them, a stale hit is
not merely unlikely — it is unrepresentable.

The dimensions in the key are the *configured* ones, never ``adapter.dimensions``: a local
model reports ``None`` until it has loaded, so the reported size changes during the first
call and a key built from it would miss everything the previous process stored.

The key names its fields in clear so an operator can read a Redis keyspace, and ends in a
digest taken over every field with its length. Model ids contain colons
(``nomic-embed-text:latest``), so the readable part alone cannot tell ``a:b`` + ``c`` from
``a`` + ``b:c``; the digest can. Entries written under the earlier
``emb:{model}:{version}:{sha256(text)}`` scheme can never match a key of this one, so they
are simply never read again and age out through eviction.

Implemented as a wrapper around any ``EmbeddingAdapter`` rather than as a branch inside each
one, so a third-party adapter registered through the entry point gets caching without
knowing the cache exists.

Batches are resolved per text, not per batch: a batch that is nine-tenths cached sends only
the missing tenth to the provider. Vectors are returned in the caller's original order, since
retrieval matches vectors to chunks positionally and a reordered batch would silently attach
every vector to the wrong chunk.

The store is asked once per batch, not once per text: one ``get_many`` before the provider
and one ``set_many`` after it, whatever the batch size. A text that appears several times in
a batch is sent to the provider once, since paying twice inside one call is no better than
paying twice across two.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from typing import ClassVar, Final, Literal

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.core.cache.stats import CacheStats
from fasterrag.core.cache.store import CacheStore
from fasterrag.errors import CacheError, EmbedError
from fasterrag.observability.logging import get_logger

__all__ = [
    "CachingEmbeddingAdapter",
    "InputType",
    "decode_vector",
    "embedding_key",
    "encode_vector",
]

InputType = Literal["document", "query"]

_ELEMENT: Final = struct.Struct("<f")
_NATIVE_SIZE: Final = "native"
_LENGTH_BYTES: Final = 8

_logger = get_logger(__name__)


def embedding_key(
    text: str,
    *,
    provider: str,
    model: str,
    model_version: str,
    dimensions: int | None,
    input_type: InputType,
) -> str:
    """Return the cache key for ``text`` embedded one way by one provider's model.

    The text is hashed, never placed in the key: keys reach a Redis keyspace and a filename.
    """
    size = _NATIVE_SIZE if dimensions is None else str(dimensions)
    identity = (provider, model, model_version, size, input_type)
    digest = hashlib.sha256()
    for part in (*identity, text):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(_LENGTH_BYTES, "big"))
        digest.update(encoded)
    return f"emb:{':'.join(identity)}:{digest.hexdigest()}"


def encode_vector(vector: Sequence[float]) -> bytes:
    """Return ``vector`` packed as little-endian float32."""
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
        """Wrap ``inner``, resolving what it can from ``store`` first."""
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

    def _key(self, text: str, input_type: InputType) -> str:
        """Return the key for ``text`` under the wrapped adapter's identity, read now."""
        return embedding_key(
            text,
            provider=self.inner.provider,
            model=self.model,
            model_version=self.model_version,
            dimensions=self.inner.config.dimensions,
            input_type=input_type,
        )

    async def _lookup(
        self, texts: Sequence[str], input_type: InputType
    ) -> list[list[float] | None]:
        """Return the cached vector for each text in one store call, counting hits and misses."""
        resolved: list[list[float] | None] = [None] * len(texts)
        if texts:
            try:
                raws = await self.store.get_many([self._key(text, input_type) for text in texts])
            except CacheError as exc:
                self.stats.record_error()
                _logger.warning(
                    "embedding cache read failed; embedding directly",
                    extra={"code": exc.code.value, "trace_id": exc.trace_id},
                )
            else:
                if len(raws) == len(texts):
                    resolved = [decode_vector(raw) if raw is not None else None for raw in raws]
                else:
                    self.stats.record_error()
                    _logger.warning(
                        "embedding cache answered a batch with the wrong number of values; "
                        "embedding directly",
                        extra={"asked": len(texts), "answered": len(raws)},
                    )

        for vector in resolved:
            if vector is None:
                self.stats.record_miss()
            else:
                self.stats.record_hit()
        return resolved

    async def _remember(
        self, vectors: Mapping[str, Sequence[float]], input_type: InputType
    ) -> None:
        """Store each text's vector in one store call; a cache failure fails nothing."""
        entries = {
            self._key(text, input_type): encode_vector(vector) for text, vector in vectors.items()
        }
        try:
            await self.store.set_many(entries)
        except CacheError as exc:
            self.stats.record_error()
            _logger.warning(
                "embedding cache write failed; the vectors are still returned",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed ``texts``, sending only the ones the cache could not supply."""
        resolved = await self._lookup(texts, "document")

        missing = list(
            dict.fromkeys(
                text for text, vector in zip(texts, resolved, strict=True) if vector is None
            )
        )
        if not missing:
            return EmbeddingResult(
                vectors=[vector for vector in resolved if vector is not None],
                model=self.model,
                model_version=self.model_version,
                total_tokens=0,
            )

        fetched = await self.inner.embed_documents(missing)
        if len(fetched.vectors) != len(missing):
            raise EmbedError(
                f"the embedding provider returned {len(fetched.vectors)} vectors for "
                f"{len(missing)} texts",
                retryable=False,
            )

        embedded = dict(zip(missing, fetched.vectors, strict=True))
        await self._remember(embedded, "document")
        for index, text in enumerate(texts):
            if resolved[index] is None:
                resolved[index] = embedded[text]

        return EmbeddingResult(
            vectors=[vector for vector in resolved if vector is not None],
            model=fetched.model,
            model_version=fetched.model_version,
            total_tokens=fetched.total_tokens,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query, consulting the cache first.

        A repeated question is common, so queries are cached too — under their own input
        type, never under the entry a passage of the same text would use.
        """
        (vector,) = await self._lookup([text], "query")
        if vector is not None:
            return vector

        vector = await self.inner.embed_query(text)
        await self._remember({text: vector}, "query")
        return vector

    async def health(self) -> HealthStatus:
        """Report the wrapped provider's health. A cache is not a dependency."""
        return await self.inner.health()

    async def close(self) -> None:
        """Release the wrapped provider and the cache backend."""
        await self.inner.close()
        await self.store.close()
