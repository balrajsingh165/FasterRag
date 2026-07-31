"""Cache storage backends.

One narrow interface — get, set, delete, sweep — over the backends
``embeddings.cache.backend`` and ``cache.backend`` name. Values are opaque bytes so a
backend never has to know whether it is holding a vector or an answer.

Expiry is stored *inside* the value rather than delegated to the backend, because the three
backends disagree about it: memory has none, disk has none, and Redis has its own. Reading
the deadline back from the payload means an entry that outlived its TTL is a miss on every
backend, including one restored from disk after a restart.

**A cache failure is never a query failure.** Every backend raises ``CacheError``, and callers
catch it and proceed uncached — FMEA row 23. A cache exists to make the pipeline faster; one
that can take the pipeline down with it is a liability, not an optimization.
"""

from __future__ import annotations

import asyncio
import hashlib
import struct
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Final

from fasterrag.errors import CacheError
from fasterrag.observability.logging import get_logger

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "MAXIMUM_ENTRIES",
    "CacheStore",
    "DiskStore",
    "Entry",
    "MemoryStore",
]

DEFAULT_CACHE_ROOT: Final = Path(".fasterrag") / "cache"

# CRITICAL: no configuration key bounds cache size, and an unbounded cache is a memory leak
# that presents as a slow OOM days into a run. Least-recently-used eviction at this ceiling
# keeps the working set without letting a long-lived process grow without limit.
MAXIMUM_ENTRIES: Final = 10_000

_HEADER: Final = struct.Struct("<d")
_NO_DEADLINE: Final = 0.0

_logger = get_logger(__name__)


class Entry:
    """A stored value and the moment it stops being valid."""

    __slots__ = ("deadline", "value")

    def __init__(self, value: bytes, deadline: float) -> None:
        """Hold ``value`` until ``deadline``; a deadline of zero never expires."""
        self.value = value
        self.deadline = deadline

    @property
    def expired(self) -> bool:
        """Return whether this entry has outlived its TTL."""
        return self.deadline != _NO_DEADLINE and time.time() >= self.deadline

    def encode(self) -> bytes:
        """Return the on-disk form: the deadline, then the value."""
        return _HEADER.pack(self.deadline) + self.value

    @classmethod
    def decode(cls, raw: bytes) -> Entry | None:
        """Return the entry ``raw`` encodes, or ``None`` if it is truncated."""
        if len(raw) < _HEADER.size:
            return None
        (deadline,) = _HEADER.unpack_from(raw)
        return cls(raw[_HEADER.size :], deadline)


def _deadline_for(ttl: int | None) -> float:
    """Return the absolute expiry for ``ttl`` seconds, or zero for no expiry."""
    return time.time() + ttl if ttl else _NO_DEADLINE


class CacheStore(ABC):
    """A key-value store with expiry, backing one of the two caches."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, or ``None`` if absent or expired."""

    @abstractmethod
    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store ``value`` under ``key`` for ``ttl`` seconds, or forever if ``None``."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove ``key``, whether or not it was present."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove every entry.

        Used for event-driven invalidation: a corpus change makes cached answers stale
        regardless of how much TTL they had left.
        """

    @abstractmethod
    async def items(self) -> list[tuple[str, bytes]]:
        """Return every live entry.

        The semantic cache needs to compare a query against every candidate, which no
        key-value backend can do for it. Bounded by ``MAXIMUM_ENTRIES``, so this scans a
        cache-sized collection rather than an unbounded one.
        """

    async def close(self) -> None:
        """Release any backend resources. The default holds none."""
        return


class MemoryStore(CacheStore):
    """In-process LRU cache. The default for the semantic cache.

    Contents die with the process, which is correct for a cache and is why it is not the
    default for embeddings — re-embedding a corpus after a restart is expensive, whereas
    re-answering a query is not.
    """

    def __init__(self, maximum_entries: int = MAXIMUM_ENTRIES) -> None:
        """Build a store holding at most ``maximum_entries`` entries."""
        self._entries: OrderedDict[str, Entry] = OrderedDict()
        self._maximum = maximum_entries
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, refreshing its recency."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expired:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store ``value``, evicting the least recently used entry if full."""
        async with self._lock:
            self._entries[key] = Entry(value, _deadline_for(ttl))
            self._entries.move_to_end(key)
            while len(self._entries) > self._maximum:
                self._entries.popitem(last=False)

    async def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        async with self._lock:
            self._entries.pop(key, None)

    async def clear(self) -> None:
        """Drop every entry."""
        async with self._lock:
            self._entries.clear()

    async def items(self) -> list[tuple[str, bytes]]:
        """Return every unexpired entry, dropping the rest as it goes."""
        async with self._lock:
            expired = [key for key, entry in self._entries.items() if entry.expired]
            for key in expired:
                del self._entries[key]
            return [(key, entry.value) for key, entry in self._entries.items()]


class DiskStore(CacheStore):
    """File-backed cache. The default for embeddings.

    Entries survive a restart, which is the whole point: an embedding is expensive enough
    that paying for it twice because a process restarted is a real cost, and the content
    hash in the key makes a stale hit impossible.

    Keys are hashed into filenames because a cache key contains a model name and a content
    hash, neither of which is guaranteed to be a legal path component on every platform.
    """

    def __init__(self, root: Path | None = None, maximum_entries: int = MAXIMUM_ENTRIES) -> None:
        """Build a store rooted at ``root``, defaulting to ``.fasterrag/cache``."""
        self.root = root or DEFAULT_CACHE_ROOT
        self._maximum = maximum_entries

    def _path(self, key: str) -> Path:
        """Return the file holding ``key``."""
        return self.root / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.bin"

    def _read(self, path: Path) -> Entry | None:
        """Return the live entry in ``path``, deleting it if expired or corrupt."""
        try:
            entry = Entry.decode(path.read_bytes())
        except OSError as exc:
            _logger.warning("cache entry unreadable", extra={"path": str(path), "error": str(exc)})
            return None

        if entry is None or entry.expired:
            path.unlink(missing_ok=True)
            return None
        return entry

    async def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, or ``None`` if absent, expired, or unreadable."""
        entry = await asyncio.to_thread(self._read, self._path(key))
        return entry.value if entry else None

    def _write(self, key: str, payload: bytes) -> None:
        """Write ``payload`` for ``key`` atomically, then evict down to the ceiling."""
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(payload)
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise CacheError(f"could not write the cache entry at {path}: {exc}") from exc

        self._evict()

    def _evict(self) -> None:
        """Delete the oldest files once the directory exceeds the entry ceiling."""
        files = sorted(self.root.glob("*.bin"), key=lambda path: path.stat().st_mtime)
        for path in files[: max(0, len(files) - self._maximum)]:
            path.unlink(missing_ok=True)

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store ``value`` under ``key``, replacing any existing entry atomically."""
        await asyncio.to_thread(self._write, key, Entry(value, _deadline_for(ttl)).encode())

    async def delete(self, key: str) -> None:
        """Remove ``key``'s file if it exists."""
        await asyncio.to_thread(self._path(key).unlink, True)

    def _clear(self) -> None:
        """Delete every cache file, leaving the directory in place."""
        if not self.root.exists():
            return
        for path in self.root.glob("*.bin"):
            path.unlink(missing_ok=True)

    async def clear(self) -> None:
        """Drop every entry."""
        await asyncio.to_thread(self._clear)

    def _items(self) -> list[tuple[str, bytes]]:
        """Return every live entry, keyed by the filename stem.

        The original key is not recoverable from a hashed filename. The semantic cache
        matches on the vector inside the value rather than on the key, so the stem is
        identity enough for it, and the embedding cache never scans.
        """
        if not self.root.exists():
            return []

        found: list[tuple[str, bytes]] = []
        for path in self.root.glob("*.bin"):
            entry = self._read(path)
            if entry is not None:
                found.append((path.stem, entry.value))
        return found

    async def items(self) -> list[tuple[str, bytes]]:
        """Return every unexpired entry."""
        return await asyncio.to_thread(self._items)
