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
from fasterrag.core.cache.semantic import CacheHit, SemanticCache, cosine_similarity
from fasterrag.core.cache.stats import CacheStats
from fasterrag.core.cache.store import (
    DEFAULT_CACHE_ROOT,
    MAXIMUM_ENTRIES,
    CacheStore,
    DiskStore,
    MemoryStore,
)
from fasterrag.errors import ConfigError

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "MAXIMUM_ENTRIES",
    "CacheHit",
    "CacheStats",
    "CacheStore",
    "CachingEmbeddingAdapter",
    "DiskStore",
    "MemoryStore",
    "SemanticCache",
    "cosine_similarity",
    "create_embedding_store",
    "create_semantic_store",
]


def _create(backend: str, setting: str, maximum_entries: int = MAXIMUM_ENTRIES) -> CacheStore:
    """Return the store ``backend`` names, holding at most ``maximum_entries`` entries.

    Raises:
        ConfigError: If the backend is Redis, which needs an optional install that is not
            yet shipped. Naming the setting keeps the error actionable rather than generic.
    """
    if backend == "memory":
        return MemoryStore(maximum_entries)
    if backend == "disk":
        return DiskStore(maximum_entries=maximum_entries)

    # TODO: the redis backend ships with TASK-0124; config validation accepts the value now
    # so the schema stays faithful to docs/config-reference.md.
    raise ConfigError(
        f"{setting} is 'redis', which is not implemented yet; "
        "use 'memory' or 'disk' until the redis backend ships"
    )


def create_embedding_store(settings: Settings) -> CacheStore:
    """Return the store backing the embedding cache, per ``embeddings.cache.backend``."""
    return _create(
        settings.embeddings.cache.backend,
        "embeddings.cache.backend",
        settings.embeddings.cache.max_entries,
    )


def create_semantic_store(settings: Settings) -> CacheStore:
    """Return the store backing the semantic cache, per ``cache.backend``."""
    return _create(settings.cache.backend, "cache.backend", settings.cache.max_entries)
