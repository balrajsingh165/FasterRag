"""Cache storage backends.

One narrow interface — get, set, delete, clear, items, close — over the backends
``embeddings.cache.backend`` and ``cache.backend`` name. Values are opaque bytes so a
backend never has to know whether it is holding a vector or an answer.

``get_many`` and ``set_many`` exist because the embedding cache is consulted a batch at a
time: sixty-four texts resolved one ``get`` each are sixty-four thread hops on disk and
sixty-four round trips to Redis before the provider is even asked. Both have a default that
loops over ``get`` and ``set``, so a backend that only implements the single-key calls is
still correct, merely not fast; every backend in this package overrides them.

Expiry is stored *inside* the value rather than delegated to the backend, because the three
backends disagree about it: memory has none, disk has none, and Redis has its own. Reading
the deadline back from the payload means an entry that outlived its TTL is a miss on every
backend, including one restored from disk after a restart.

**A cache failure is never a query failure.** Every backend raises ``CacheError``, and callers
catch it and proceed uncached — FMEA row 23. A cache exists to make the pipeline faster; one
that can take the pipeline down with it is a liability, not an optimization.

**The disk store counts its entries instead of listing them.** Enforcing the ceiling by
listing and sorting the directory after every write made each write cost a pass over the
whole cache. ``DiskStore`` keeps a count instead: one pass when the first write of a process
finds out what is already there, then one added for every file it creates. Only when that
count passes the ceiling is the directory listed again, and that pass evicts the oldest
entries down to a low-water mark a tenth below the ceiling rather than to the ceiling
itself, so the writes that follow have room before the next one. Two things follow.

*The ceiling still holds.* Writes and evictions in one process take turns, so the count is
never lower than the number of files that process has put there: replacing an entry adds
nothing, and a file removed without the count being told (an expired entry dropped by a
read, a ``delete``) only makes it too high, which costs an early pass and never a late one.
So once a write has returned, the directory holds at most ``maximum_entries`` entries, as it
did before. What the count cannot see is another process writing to the same directory;
that excess is bounded by what the other process wrote since this one last listed, and is
removed at this one's next eviction.

*A pass is paid for by the writes it makes room for.* With a ceiling of ten thousand an
eviction frees a thousand slots, so at most one write in a thousand lists the directory.
For a ceiling under ten the low-water mark is the ceiling itself and every write past it
lists a directory of fewer than ten files, which is not worth amortising.

Eviction order is by modification time: the least recently *written* entries go first. A
read does not refresh it, because that would turn every hit into a write.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import suppress
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
    "deadline_for",
]

DEFAULT_CACHE_ROOT: Final = Path(".fasterrag") / "cache"

# CRITICAL: no configuration key bounds cache size, and an unbounded cache is a memory leak
# that presents as a slow OOM days into a run. Least-recently-used eviction at this ceiling
# keeps the working set without letting a long-lived process grow without limit.
MAXIMUM_ENTRIES: Final = 10_000

_HEADER: Final = struct.Struct("<d")
_NO_DEADLINE: Final = 0.0
_ENTRY_SUFFIX: Final = ".bin"
_HEADROOM_DIVISOR: Final = 10

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


def deadline_for(ttl: int | None) -> float:
    """Return the absolute expiry for ``ttl`` seconds, or zero for no expiry.

    Shared by every backend so all three agree on what "no TTL" means: a deadline of zero,
    never a timestamp already in the past.
    """
    return time.time() + ttl if ttl else _NO_DEADLINE


class CacheStore(ABC):
    """A key-value store with expiry, backing one of the two caches."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, or ``None`` if absent or expired."""

    @abstractmethod
    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store ``value`` under ``key`` for ``ttl`` seconds, or forever if ``None``."""

    async def get_many(self, keys: Sequence[str]) -> list[bytes | None]:
        """Return the value for each of ``keys`` in order, ``None`` where absent or expired."""
        return [await self.get(key) for key in keys]

    async def set_many(self, entries: Mapping[str, bytes], *, ttl: int | None = None) -> None:
        """Store every entry in ``entries`` for ``ttl`` seconds, or forever if ``None``."""
        for key, value in entries.items():
            await self.set(key, value, ttl=ttl)

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
        """Return every live entry."""

    async def close(self) -> None:
        """Release any backend resources. The default holds none."""
        return


class MemoryStore(CacheStore):
    """In-process LRU cache. The default for the semantic cache."""

    def __init__(self, maximum_entries: int = MAXIMUM_ENTRIES) -> None:
        """Build a store holding at most ``maximum_entries`` entries."""
        self._entries: OrderedDict[str, Entry] = OrderedDict()
        self._maximum = maximum_entries
        self._lock = asyncio.Lock()

    def _live(self, key: str) -> bytes | None:
        """Return the live value for ``key`` and refresh its recency; the lock is held."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expired:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return entry.value

    async def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, refreshing its recency."""
        async with self._lock:
            return self._live(key)

    async def get_many(self, keys: Sequence[str]) -> list[bytes | None]:
        """Return the value for each of ``keys`` under one acquisition of the lock."""
        async with self._lock:
            return [self._live(key) for key in keys]

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store ``value``, evicting the least recently used entry if full."""
        await self.set_many({key: value}, ttl=ttl)

    async def set_many(self, entries: Mapping[str, bytes], *, ttl: int | None = None) -> None:
        """Store every entry under one acquisition of the lock, then evict down to the ceiling."""
        deadline = deadline_for(ttl)
        async with self._lock:
            for key, value in entries.items():
                self._entries[key] = Entry(value, deadline)
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
    """File-backed cache. The default for embeddings."""

    def __init__(self, root: Path | None = None, maximum_entries: int = MAXIMUM_ENTRIES) -> None:
        """Build a store rooted at ``root``, defaulting to ``.fasterrag/cache``."""
        self.root = root or DEFAULT_CACHE_ROOT
        self._maximum = maximum_entries
        self._low_water = maximum_entries - maximum_entries // _HEADROOM_DIVISOR
        self._held: int | None = None
        self._writing = threading.Lock()

    def _path(self, key: str) -> Path:
        """Return the file holding ``key``."""
        return self.root / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}{_ENTRY_SUFFIX}"

    def _discard(self, path: Path) -> bool:
        """Delete ``path`` and say whether it is gone, logging a refusal instead of raising."""
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            _logger.warning(
                "cache entry could not be removed", extra={"path": str(path), "error": str(exc)}
            )
            return False
        return True

    def _read(self, path: Path) -> Entry | None:
        """Return the live entry in ``path``, deleting it if expired or corrupt."""
        try:
            entry = Entry.decode(path.read_bytes())
        except FileNotFoundError:
            # CRITICAL: an absent file is the ordinary miss, not a fault. Logging it at
            # warning made every first-time chunk of an ingest write a line saying the
            # cache was unreadable, which buried the one warning that would have meant it.
            return None
        except OSError as exc:
            _logger.warning("cache entry unreadable", extra={"path": str(path), "error": str(exc)})
            return None

        if entry is None or entry.expired:
            self._discard(path)
            return None
        return entry

    async def get(self, key: str) -> bytes | None:
        """Return the value for ``key``, or ``None`` if absent, expired, or unreadable."""
        entry = await asyncio.to_thread(self._read, self._path(key))
        return entry.value if entry else None

    def _read_many(self, paths: Sequence[Path]) -> list[bytes | None]:
        """Return the live value in each of ``paths``, in order."""
        entries = [self._read(path) for path in paths]
        return [entry.value if entry else None for entry in entries]

    async def get_many(self, keys: Sequence[str]) -> list[bytes | None]:
        """Return the value for each of ``keys``, reading them all in one worker thread."""
        if not keys:
            return []
        return await asyncio.to_thread(self._read_many, [self._path(key) for key in keys])

    def _write(self, key: str, payload: bytes) -> None:
        """Write ``payload`` for ``key`` atomically, then hold the directory to its ceiling."""
        self._write_many({key: payload})

    def _write_many(self, payloads: Mapping[str, bytes]) -> None:
        """Write every payload atomically, then hold the directory to its ceiling once."""
        # CRITICAL: one writer at a time, eviction included. Unlocked, a writer that saw
        # its file present could have it evicted and then recreate it uncounted, and a
        # count that is too low is the one error that lets the directory pass its ceiling.
        with self._writing:
            created = 0
            try:
                for key, payload in payloads.items():
                    created += self._store(key, payload)
            finally:
                self._settle(created)

    def _store(self, key: str, payload: bytes) -> int:
        """Write ``payload`` for ``key`` atomically and return how many files that added."""
        path = self._path(key)
        # CRITICAL: the temporary name is private to this thread of this process. Two
        # writers of one key sharing a name could rename each other's half-written file
        # into place, and a truncated vector that still divides into float32s decodes as a
        # shorter, wrong vector rather than as a miss.
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            created = 0 if path.exists() else 1
            temporary.write_bytes(payload)
            temporary.replace(path)
        except OSError as exc:
            # CRITICAL: the cleanup is itself suppressed. On Linux, when the parent is a
            # file rather than a directory, `unlink` raises NotADirectoryError from inside
            # this handler — replacing the error being handled with a second one that
            # escapes, so a degradation path becomes a crash. Windows does not raise here,
            # which is why this only ever failed on the Linux CI leg.
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise CacheError(f"could not write the cache entry at {path}: {exc}") from exc
        return created

    def _settle(self, created: int) -> None:
        """Add ``created`` files to the count, listing the directory only past the ceiling."""
        if self._held is None:
            self._held = len(self._scan())
        else:
            self._held += created
        if self._held > self._maximum:
            self._held = self._evict()

    def _scan(self) -> list[tuple[int, str]]:
        """Return the modification time and name of every entry file, in one pass."""
        found: list[tuple[int, str]] = []
        try:
            with os.scandir(self.root) as listing:
                for item in listing:
                    if not item.name.endswith(_ENTRY_SUFFIX):
                        continue
                    try:
                        found.append((item.stat().st_mtime_ns, item.name))
                    except FileNotFoundError:
                        _logger.debug(
                            "cache entry vanished during a scan", extra={"entry": item.name}
                        )
        except FileNotFoundError:
            _logger.debug("cache directory does not exist yet", extra={"path": str(self.root)})
        except OSError as exc:
            raise CacheError(f"could not list the cache directory {self.root}: {exc}") from exc
        return found

    def _evict(self) -> int:
        """Delete the oldest entries down to the low-water mark; return how many remain."""
        files = sorted(self._scan())
        if len(files) <= self._maximum:
            return len(files)

        victims = files[: len(files) - self._low_water]
        removed = sum(1 for _, name in victims if self._discard(self.root / name))
        _logger.debug(
            "disk cache evicted entries past its ceiling",
            extra={"path": str(self.root), "evicted": removed, "kept": len(files) - removed},
        )
        return len(files) - removed

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        """Store ``value`` under ``key``, replacing any existing entry atomically."""
        await asyncio.to_thread(self._write, key, Entry(value, deadline_for(ttl)).encode())

    async def set_many(self, entries: Mapping[str, bytes], *, ttl: int | None = None) -> None:
        """Store every entry in one worker thread, evicting once at the end."""
        if not entries:
            return
        deadline = deadline_for(ttl)
        payloads = {key: Entry(value, deadline).encode() for key, value in entries.items()}
        await asyncio.to_thread(self._write_many, payloads)

    async def delete(self, key: str) -> None:
        """Remove ``key``'s file if it exists."""
        await asyncio.to_thread(self._path(key).unlink, True)

    def _clear(self) -> None:
        """Delete every cache file, leaving the directory in place."""
        with self._writing:
            self._held = None
            if not self.root.exists():
                return
            for path in self.root.glob("*.bin"):
                path.unlink(missing_ok=True)

    async def clear(self) -> None:
        """Drop every entry."""
        await asyncio.to_thread(self._clear)

    def _items(self) -> list[tuple[str, bytes]]:
        """Return every live entry, keyed by the filename stem."""
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
