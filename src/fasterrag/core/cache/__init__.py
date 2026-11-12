"""Caching: the embedding cache and the semantic response cache.

Two caches with different jobs. The embedding cache is on by default
(``embeddings.cache.enabled``) and keyed by the exact text together with everything that
decides its vector — provider, model, model version, configured dimensions, and whether it
is embedded as a document or as a query — so a hit is the vector the provider would have
returned, at the float32 precision the cache packs vectors in. The semantic cache is off by
default and keyed by similarity, so a hit is a judgement call bounded by
``cache.similarity_threshold``.

The embedding cache is applied in exactly one place,
``services.composition.create_embedding_router``, which wraps every adapter of the router
that the API, the CLI, the facade and the ingestion service all build. Nothing in this
package applies it, and an adapter constructed directly through
``adapters.embeddings.create_embedding_adapter`` is not cached. Vectors produced by late
chunking pass through uncached, because they depend on the whole document rather than on
the chunk's text (see ``embedding.py``).

Both share one rule, from FMEA row 23: **a cache failure degrades to cache-off**. Every
backend error is caught, counted, and stepped past. Correctness never depends on a cache
being reachable — only speed does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.core.cache.embedding import (
    CachingEmbeddingAdapter,
    PoolingCachingEmbeddingAdapter,
    cache_embeddings,
)
from fasterrag.core.cache.redis import (
    DEFAULT_REDIS_TIMEOUT_MS,
    EMBEDDING_NAMESPACE,
    SEMANTIC_NAMESPACE,
    RedisStore,
)
from fasterrag.core.cache.semantic import CacheHit, SemanticCache, cosine_similarity
from fasterrag.core.cache.stats import CacheStats
from fasterrag.core.cache.store import (
    DEFAULT_CACHE_ROOT,
    MAXIMUM_ENTRIES,
    CacheStore,
    DiskStore,
    MemoryStore,
)

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "EMBEDDING_CACHE_ROOT",
    "EMBEDDING_NAMESPACE",
    "MAXIMUM_ENTRIES",
    "SEMANTIC_NAMESPACE",
    "CacheHit",
    "CacheStats",
    "CacheStore",
    "CachingEmbeddingAdapter",
    "DiskStore",
    "MemoryStore",
    "PoolingCachingEmbeddingAdapter",
    "RedisStore",
    "SemanticCache",
    "cache_embeddings",
    "cosine_similarity",
    "create_embedding_store",
    "create_semantic_store",
]


# CRITICAL: the embedding cache gets a directory of its own. A disk store owns every entry
# file in its directory — it counts them against its ceiling, and `clear` deletes them all —
# and every ingest clears the semantic cache. Sharing one directory would empty the
# embedding cache on each ingest, which is exactly when it is about to be useful. The
# semantic cache keeps the parent directory, where its entries already are.
EMBEDDING_CACHE_ROOT: Final = DEFAULT_CACHE_ROOT / "embedding"


def _create(
    backend: str,
    setting: str,
    maximum_entries: int,
    redis_url: str,
    namespace: str,
    root: Path | None = None,
    redis_timeout_ms: int = DEFAULT_REDIS_TIMEOUT_MS,
) -> CacheStore:
    """Return the store ``backend`` names, holding at most ``maximum_entries`` entries."""
    if backend == "memory":
        return MemoryStore(maximum_entries)
    if backend == "disk":
        return DiskStore(root, maximum_entries=maximum_entries)

    return RedisStore(
        redis_url,
        namespace=namespace,
        maximum_entries=maximum_entries,
        setting=setting,
        timeout_ms=redis_timeout_ms,
    )


def create_embedding_store(settings: Settings) -> CacheStore:
    """Return the store backing the embedding cache, per ``embeddings.cache.backend``."""
    return _create(
        settings.embeddings.cache.backend,
        "embeddings.cache.backend",
        settings.embeddings.cache.max_entries,
        settings.embeddings.cache.redis_url,
        EMBEDDING_NAMESPACE,
        EMBEDDING_CACHE_ROOT,
        redis_timeout_ms=settings.reliability.timeouts.redis_ms,
    )


def create_semantic_store(settings: Settings) -> CacheStore:
    """Return the store backing the semantic cache, per ``cache.backend``."""
    return _create(
        settings.cache.backend,
        "cache.backend",
        settings.cache.max_entries,
        settings.cache.redis_url,
        SEMANTIC_NAMESPACE,
        redis_timeout_ms=settings.reliability.timeouts.redis_ms,
    )
