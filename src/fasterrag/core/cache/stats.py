"""Cache hit/miss accounting.

The source behind ``fasterrag_cache_events_total{cache, result}`` in
``docs/observability.md``. Counters live here rather than in a metrics library so both caches
report the same three outcomes without depending on an exporter that has not shipped yet;
the metrics catalogue (TASK-0042) reads these rather than instrumenting the caches again.

A hit rate is the only way to tell a cache that is working from one that is merely present.
An embedding cache that never hits costs a hash per chunk and saves nothing, and a semantic
cache whose threshold is too tight looks identical to a disabled one from the outside.
"""

from __future__ import annotations

from dataclasses import dataclass

from fasterrag.observability import metrics

__all__ = ["CacheStats"]


@dataclass(slots=True)
class CacheStats:
    """Outcome counts for one cache."""

    name: str
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    errors: int = 0

    @property
    def lookups(self) -> int:
        """Return how many times the cache was consulted."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Return the fraction of lookups that hit, or zero before the first lookup."""
        return self.hits / self.lookups if self.lookups else 0.0

    def record_hit(self) -> None:
        """Count a lookup that was served from the cache."""
        self.hits += 1
        metrics.CACHE_EVENTS.increment(cache=self.name, result="hit")

    def record_miss(self) -> None:
        """Count a lookup that had to run the real work."""
        self.misses += 1
        metrics.CACHE_EVENTS.increment(cache=self.name, result="miss")

    def record_invalidation(self, count: int = 1) -> None:
        """Count entries dropped by a corpus change rather than by expiry."""
        self.invalidations += count
        if count:
            metrics.CACHE_EVENTS.increment(float(count), cache=self.name, result="invalidated")

    def record_error(self) -> None:
        """Count a backend failure the caller proceeded past uncached."""
        self.errors += 1

    def as_dict(self) -> dict[str, float | int | str]:
        """Return the form the status endpoint and the CLI report."""
        return {
            "cache": self.name,
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
        }
