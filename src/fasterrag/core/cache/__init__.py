"""Caching: the embedding cache and the semantic response cache.

Two caches with different jobs. The embedding cache is on by default and keyed by exact
content, so a hit is always correct; the semantic cache is off by default and keyed by
similarity, so a hit is a judgement call bounded by ``cache.similarity_threshold``.

Both share one rule, from FMEA row 23: **a cache failure degrades to cache-off**. Every
backend error is caught, counted, and stepped past. Correctness never depends on a cache
being reachable — only speed does.
"""

from __future__ import annotations

from fasterrag.config.schema import Settings
from fasterrag.core.cache.embedding import CachingEmbeddingAdapter
from fasterrag.core.cache.redis import EMBEDDING_NAMESPACE, SEMANTIC_NAMESPACE, RedisStore
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
    "EMBEDDING_NAMESPACE",
    "MAXIMUM_ENTRIES",
    "SEMANTIC_NAMESPACE",
    "CacheHit",
    "CacheStats",
    "CacheStore",
    "CachingEmbeddingAdapter",
    "DiskStore",
    "MemoryStore",
    "RedisStore",
    "SemanticCache",
    "cosine_similarity",
    "create_embedding_store",
    "create_semantic_store",
]


def _create(
    backend: str,
    setting: str,
    maximum_entries: int,
    redis_url: str,
    namespace: str,
) -> CacheStore:
    """Return the store ``backend`` names, holding at most ``maximum_entries`` entries.

    The two caches are handed different namespaces so that one Redis server can back both
    without either one's ``clear`` reaching the other's entries.

    Raises:
        ConfigError: If the backend is Redis and the optional client is not installed, or
            the configured URL cannot be parsed. Naming the setting keeps the error
            actionable rather than generic.
    """
    if backend == "memory":
        return MemoryStore(maximum_entries)
    if backend == "disk":
        return DiskStore(maximum_entries=maximum_entries)

    return RedisStore(
        redis_url,
        namespace=namespace,
        maximum_entries=maximum_entries,
        setting=setting,
    )


def create_embedding_store(settings: Settings) -> CacheStore:
    """Return the store backing the embedding cache, per ``embeddings.cache.backend``."""
    return _create(
        settings.embeddings.cache.backend,
        "embeddings.cache.backend",
        settings.embeddings.cache.max_entries,
        settings.embeddings.cache.redis_url,
        EMBEDDING_NAMESPACE,
    )


def create_semantic_store(settings: Settings) -> CacheStore:
    """Return the store backing the semantic cache, per ``cache.backend``."""
    return _create(
        settings.cache.backend,
        "cache.backend",
        settings.cache.max_entries,
        settings.cache.redis_url,
        SEMANTIC_NAMESPACE,
    )
