"""Redis-backed cache store: one cache shared by every process that reaches the same server.

The backend the other two cannot be. ``memory`` dies with its process and ``disk`` is
private to one filesystem, so neither lets a pool of API replicas or a stream of short-lived
CLI invocations share a single cache — each one re-embeds and re-answers what a sibling
already paid for. Selected by ``embeddings.cache.backend`` or ``cache.backend`` and
connected through the matching ``redis_url``.

**The entry ceiling is enforced by a sorted-set index, not by Redis.** Redis has no
per-prefix LRU: ``maxmemory-policy`` is a server-wide setting on a server we do not own, and
it would evict a co-tenant application's keys as readily as ours. Counting our own keys by
scanning the keyspace is O(everything stored) and races every concurrent write. So each
write records its key as a member of one sorted set scored by the moment it was last
touched, and eviction removes the lowest-scored members past ``maximum_entries``. Reads
restack the score, which makes the eviction order least-recently-*used* rather than
least-recently-written, matching ``MemoryStore``. The same index is what ``items`` walks and
what ``clear`` deletes, so neither ever reads or removes a key this namespace did not write.

**Index first, value second.** The pair is deliberately not wrapped in a transaction.
Ordered this way, an interrupted write leaves an index member pointing at a value that does
not exist, which the next read prunes. The reverse order would leave a value that no index
knows about — invisible to the ceiling, to ``items``, and to ``clear`` — and an entry
nothing can find or evict is a leak rather than a miss.

TTL is written twice: into the payload, where ``Entry`` holds the deadline that makes an
expired entry a miss on every backend, and as the key's native expiry, so Redis reclaims the
memory instead of holding a dead payload until something happens to read it.

Every failure is translated into ``CacheError`` at this boundary, because callers degrade to
cache-off on that one type (FMEA row 23) and a raw connection error would escape as a query
failure instead.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Final

from fasterrag.core.cache.store import MAXIMUM_ENTRIES, CacheStore, Entry, deadline_for
from fasterrag.errors import CacheError, ConfigError
from fasterrag.observability.logging import get_logger

__all__ = ["EMBEDDING_NAMESPACE", "SEMANTIC_NAMESPACE", "RedisStore"]

EMBEDDING_NAMESPACE: Final = "fasterrag:embedding"
SEMANTIC_NAMESPACE: Final = "fasterrag:semantic"

_logger = get_logger(__name__)


def _install_hint(setting: str) -> str:
    """Return the message shown when the backend is selected without its client."""
    return (
        f"{setting} is 'redis', which needs the redis client; "
        "install it with 'pip install fasterrag[redis]'"
    )


def _member(raw: object) -> str:
    """Return a sorted-set member as text, whether the client decoded it or not."""
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


class RedisStore(CacheStore):
    """A cache store held in Redis, bounded by a sorted-set index."""

    def __init__(
        self,
        url: str,
        *,
        namespace: str,
        maximum_entries: int = MAXIMUM_ENTRIES,
        setting: str = "cache.backend",
        client: Any | None = None,
    ) -> None:
        """Build a store over ``url``, keeping its keys under ``namespace``.

        Connecting is deferred: ``from_url`` builds a connection pool without opening a
        socket, so a store is constructible at startup and only a real operation can fail.

        Args:
            url: Redis connection URL, from ``cache.redis_url`` or
                ``embeddings.cache.redis_url``.
            namespace: Key prefix isolating this cache from the other one and from anything
                else sharing the server.
            maximum_entries: Ceiling past which the least recently used entries are evicted.
            setting: The configuration key that selected this backend, so an install or URL
                error names the knob the operator set rather than the module that failed.
            client: A pre-built client, used by tests. Supplying one skips the import, so
                the backend is testable without the optional dependency installed.

        Raises:
            ConfigError: If the redis client is not installed, or the URL is malformed.
        """
        self._namespace = namespace
        self._index = f"{namespace}:index"
        self._maximum = maximum_entries
        self._failures: tuple[type[BaseException], ...] = (OSError,)

        try:
            from redis.asyncio import Redis
            from redis.exceptions import RedisError
        except ImportError as exc:
            if client is None:
                raise ConfigError(_install_hint(setting)) from exc
            self._client: Any = client
            return

        self._failures = (RedisError, OSError)
        if client is not None:
            self._client = client
            return

        try:
            # CRITICAL: decode_responses must stay false. Values are binary — packed float32
            # vectors behind a struct-packed deadline header — and a client that decodes
            # replies to str mangles every one of them into an unparseable entry, which
            # reads as a permanent cache miss rather than as an error. A URL may ask for
            # decoding via its query string, so it is overridden here rather than trusted.
            self._client = Redis.from_url(url, decode_responses=False)
        except ValueError as exc:
            raise ConfigError(
                f"{setting} is 'redis' but the configured redis_url could not be parsed: {exc}"
            ) from exc

    def _key(self, key: str) -> str:
        """Return the namespaced Redis key holding ``key``."""
        return f"{self._namespace}:entry:{key}"

    @asynccontextmanager
    async def _guard(self, operation: str) -> AsyncIterator[None]:
        """Translate any backend failure inside the block into ``CacheError``.

        Raises:
            CacheError: If the wrapped operation fails. The message names the operation and
                the exception type but never the value or the connection URL, either of
                which can carry a credential into a log line.
        """
        try:
            yield
        except self._failures as exc:
            raise CacheError(
                f"the redis cache backend failed to {operation}: {type(exc).__name__}"
            ) from exc

    async def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, refreshing its recency, or ``None`` on a miss."""
        async with self._guard("read an entry"):
            raw = await self._client.get(self._key(key))
            entry = Entry.decode(raw) if isinstance(raw, bytes) else None
            if entry is None or entry.expired:
                if raw is not None:
                    await self._client.delete(self._key(key))
                await self._client.zrem(self._index, key)
                return None

            await self._client.zadd(self._index, {key: time.time()})
            return entry.value

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store ``value`` under ``key``, then evict down to the entry ceiling."""
        payload = Entry(value, deadline_for(ttl)).encode()
        async with self._guard("write an entry"):
            await self._client.zadd(self._index, {key: time.time()})
            await self._client.set(self._key(key), payload, ex=ttl or None)
            await self._evict()

    async def _evict(self) -> None:
        """Drop the least recently used entries once the index exceeds the ceiling."""
        excess = int(await self._client.zcard(self._index)) - self._maximum
        if excess <= 0:
            return

        victims = [_member(raw) for raw in await self._client.zrange(self._index, 0, excess - 1)]
        if not victims:
            return

        await self._forget(victims)
        _logger.debug(
            "redis cache evicted entries past its ceiling",
            extra={"namespace": self._namespace, "evicted": len(victims)},
        )

    async def _forget(self, keys: Sequence[str]) -> None:
        """Delete ``keys`` and their index members, values first so nothing is orphaned."""
        await self._client.delete(*(self._key(key) for key in keys))
        await self._client.zrem(self._index, *keys)

    async def delete(self, key: str) -> None:
        """Remove ``key``, whether or not it was present."""
        async with self._guard("delete an entry"):
            await self._forget([key])

    async def clear(self) -> None:
        """Drop every entry this namespace owns, leaving co-tenant keys untouched.

        Scoped through the index rather than ``FLUSHDB``: the server may be shared with the
        other cache, with an unrelated application, or with a queue, and a cache that wipes
        someone else's data on a corpus change is a fault, not an invalidation.
        """
        async with self._guard("clear the cache"):
            members = [_member(raw) for raw in await self._client.zrange(self._index, 0, -1)]
            if members:
                await self._client.delete(*(self._key(member) for member in members))
            await self._client.delete(self._index)

    async def items(self) -> list[tuple[str, bytes]]:
        """Return every live entry, pruning the index of anything Redis no longer holds."""
        async with self._guard("list entries"):
            members = [_member(raw) for raw in await self._client.zrange(self._index, 0, -1)]
            if not members:
                return []

            values = await self._client.mget([self._key(member) for member in members])
            found: list[tuple[str, bytes]] = []
            stale: list[str] = []
            for member, raw in zip(members, values, strict=True):
                entry = Entry.decode(raw) if isinstance(raw, bytes) else None
                if entry is None or entry.expired:
                    stale.append(member)
                else:
                    found.append((member, entry.value))

            if stale:
                await self._forget(stale)
            return found

    async def close(self) -> None:
        """Release the connection pool.

        A failed disconnect is logged rather than raised. ``close`` runs on the shutdown
        path, where the only thing raising achieves is turning an orderly stop into a
        traceback over a resource the process is about to drop anyway.
        """
        try:
            await self._client.aclose()
        except self._failures as exc:
            _logger.warning(
                "the redis cache backend did not close cleanly",
                extra={"namespace": self._namespace, "error": type(exc).__name__},
            )
