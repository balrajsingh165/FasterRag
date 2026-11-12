import asyncio
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fasterrag.core.cache.store import CacheStore, DiskStore, Entry, MemoryStore
from fasterrag.errors import CacheError


@pytest.fixture(params=["memory", "disk"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> MemoryStore | DiskStore:
    if request.param == "memory":
        return MemoryStore()
    return DiskStore(tmp_path)


async def test_a_stored_value_reads_back(store: MemoryStore | DiskStore) -> None:
    await store.set("k", b"value")

    assert await store.get("k") == b"value"


async def test_an_absent_key_is_a_miss(store: MemoryStore | DiskStore) -> None:
    assert await store.get("never-written") is None


async def test_a_value_is_replaced_rather_than_appended(store: MemoryStore | DiskStore) -> None:
    await store.set("k", b"first")
    await store.set("k", b"second")

    assert await store.get("k") == b"second"


async def test_an_empty_value_round_trips(store: MemoryStore | DiskStore) -> None:
    await store.set("k", b"")

    assert await store.get("k") == b""


async def test_deleting_removes_the_entry(store: MemoryStore | DiskStore) -> None:
    await store.set("k", b"value")
    await store.delete("k")

    assert await store.get("k") is None


async def test_deleting_an_absent_key_is_not_an_error(store: MemoryStore | DiskStore) -> None:
    await store.delete("never-written")


async def test_clearing_removes_everything(store: MemoryStore | DiskStore) -> None:
    await store.set("a", b"1")
    await store.set("b", b"2")
    await store.clear()

    assert await store.items() == []


async def test_an_expired_entry_is_a_miss(store: MemoryStore | DiskStore) -> None:
    await store.set("k", b"value", ttl=1)
    time.sleep(1.05)

    assert await store.get("k") is None


async def test_a_live_ttl_entry_is_a_hit(store: MemoryStore | DiskStore) -> None:
    await store.set("k", b"value", ttl=60)

    assert await store.get("k") == b"value"


async def test_no_ttl_means_no_expiry(store: MemoryStore | DiskStore) -> None:
    await store.set("k", b"value")

    assert await store.get("k") == b"value"


async def test_items_returns_every_live_entry(store: MemoryStore | DiskStore) -> None:
    await store.set("a", b"1")
    await store.set("b", b"2")

    assert sorted(value for _, value in await store.items()) == [b"1", b"2"]


async def test_items_omits_expired_entries(store: MemoryStore | DiskStore) -> None:
    await store.set("live", b"1", ttl=60)
    await store.set("dead", b"2", ttl=1)
    time.sleep(1.05)

    assert [value for _, value in await store.items()] == [b"1"]


async def test_a_key_with_path_separators_is_usable(store: MemoryStore | DiskStore) -> None:
    key = "emb:sentence-transformers/all-MiniLM-L6-v2:1.0:abcdef"
    await store.set(key, b"value")

    assert await store.get(key) == b"value"


async def test_the_memory_store_evicts_the_least_recently_used() -> None:
    store = MemoryStore(maximum_entries=2)
    await store.set("a", b"1")
    await store.set("b", b"2")
    await store.get("a")
    await store.set("c", b"3")

    assert await store.get("b") is None
    assert await store.get("a") == b"1"
    assert await store.get("c") == b"3"


async def test_the_memory_store_never_exceeds_its_ceiling() -> None:
    store = MemoryStore(maximum_entries=3)
    for index in range(20):
        await store.set(f"k{index}", b"v")

    assert len(await store.items()) == 3


async def test_the_disk_store_survives_a_new_instance(tmp_path: Path) -> None:
    await DiskStore(tmp_path).set("k", b"value")

    assert await DiskStore(tmp_path).get("k") == b"value"


async def test_the_disk_store_evicts_down_to_its_ceiling(tmp_path: Path) -> None:
    store = DiskStore(tmp_path, maximum_entries=3)
    for index in range(10):
        await store.set(f"k{index}", b"v")

    assert len(list(tmp_path.glob("*.bin"))) == 3


async def test_a_truncated_disk_entry_is_a_miss_rather_than_a_crash(tmp_path: Path) -> None:
    store = DiskStore(tmp_path)
    await store.set("k", b"value")
    next(tmp_path.glob("*.bin")).write_bytes(b"xx")

    assert await store.get("k") is None


async def test_a_corrupt_disk_entry_is_dropped_from_items(tmp_path: Path) -> None:
    store = DiskStore(tmp_path)
    await store.set("k", b"value")
    next(tmp_path.glob("*.bin")).write_bytes(b"xx")

    assert await store.items() == []


async def test_an_unwritable_disk_root_raises_a_typed_cache_error(tmp_path: Path) -> None:
    blocker = tmp_path / "cache"
    blocker.write_text("not a directory")

    with pytest.raises(CacheError):
        await DiskStore(blocker).set("k", b"value")


async def test_the_disk_store_leaves_no_temporary_files(tmp_path: Path) -> None:
    await DiskStore(tmp_path).set("k", b"value")

    assert list(tmp_path.glob("*.tmp")) == []


def test_a_truncated_payload_decodes_to_nothing() -> None:
    assert Entry.decode(b"") is None
    assert Entry.decode(b"abc") is None


def test_an_entry_round_trips_through_its_encoding() -> None:
    decoded = Entry.decode(Entry(b"payload", 0.0).encode())

    assert decoded is not None
    assert decoded.value == b"payload"
    assert decoded.expired is False


class SingleKeyStore(CacheStore):
    """A third-party backend written before the batch calls existed."""

    def __init__(self) -> None:
        self.held: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.held.get(key)

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        self.held[key] = value

    async def delete(self, key: str) -> None:
        self.held.pop(key, None)

    async def clear(self) -> None:
        self.held.clear()

    async def items(self) -> list[tuple[str, bytes]]:
        return list(self.held.items())


async def test_a_batch_reads_back_in_the_order_it_was_asked_for(
    store: MemoryStore | DiskStore,
) -> None:
    await store.set_many({"a": b"1", "b": b"2", "c": b"3"})

    assert await store.get_many(["c", "absent", "a", "b"]) == [b"3", None, b"1", b"2"]


async def test_a_batch_write_is_readable_one_key_at_a_time(store: MemoryStore | DiskStore) -> None:
    await store.set_many({"a": b"1", "b": b"2"})

    assert await store.get("a") == b"1"
    assert await store.get("b") == b"2"


async def test_an_empty_batch_is_not_an_error(store: MemoryStore | DiskStore) -> None:
    await store.set_many({})

    assert await store.get_many([]) == []


async def test_a_batch_read_treats_an_expired_entry_as_a_miss(
    store: MemoryStore | DiskStore,
) -> None:
    await store.set_many({"dead": b"1"}, ttl=1)
    await store.set_many({"live": b"2"}, ttl=60)
    time.sleep(1.05)

    assert await store.get_many(["dead", "live"]) == [None, b"2"]


async def test_a_batch_write_replaces_existing_entries(store: MemoryStore | DiskStore) -> None:
    await store.set("a", b"old")
    await store.set_many({"a": b"new"})

    assert await store.get_many(["a"]) == [b"new"]


async def test_a_backend_with_only_single_key_calls_still_answers_batches() -> None:
    """The batch calls are additive: a store written against the old interface keeps working."""
    store = SingleKeyStore()

    await store.set_many({"a": b"1", "b": b"2"})

    assert await store.get_many(["b", "absent", "a"]) == [b"2", None, b"1"]


async def test_the_memory_store_evicts_once_a_batch_passes_its_ceiling() -> None:
    store = MemoryStore(maximum_entries=3)

    await store.set_many({f"k{index}": b"v" for index in range(10)})

    assert [key for key, _ in await store.items()] == ["k7", "k8", "k9"]


async def test_a_batch_read_refreshes_recency_in_the_memory_store() -> None:
    store = MemoryStore(maximum_entries=2)
    await store.set_many({"a": b"1", "b": b"2"})
    await store.get_many(["a"])
    await store.set("c", b"3")

    assert await store.get_many(["a", "b", "c"]) == [b"1", None, b"3"]


def _counting_threads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every hop to a worker thread the store makes."""
    hops: list[str] = []
    real = asyncio.to_thread

    async def counted(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        hops.append(getattr(function, "__name__", repr(function)))
        return await real(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", counted)
    return hops


async def test_the_disk_store_reads_a_batch_in_one_thread_hop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sixty-four texts looked up one at a time were sixty-four hops to a worker thread."""
    store = DiskStore(tmp_path)
    keys = [f"k{index}" for index in range(64)]
    await store.set_many(dict.fromkeys(keys, b"v"))
    hops = _counting_threads(monkeypatch)

    assert await store.get_many(keys) == [b"v"] * 64
    assert len(hops) == 1


async def test_the_disk_store_writes_a_batch_in_one_thread_hop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DiskStore(tmp_path)
    hops = _counting_threads(monkeypatch)

    await store.set_many({f"k{index}": b"v" for index in range(64)})

    assert len(hops) == 1
    assert len(list(tmp_path.glob("*.bin"))) == 64


async def test_a_disk_miss_is_not_logged_as_an_unreadable_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every miss logged a warning, so a first ingest wrote one line per chunk."""
    store = DiskStore(tmp_path)

    with caplog.at_level(logging.DEBUG):
        assert await store.get("never-written") is None
        assert await store.get_many(["never-written", "nor-this"]) == [None, None]

    assert [record for record in caplog.records if record.name.startswith("fasterrag")] == []


def entries_on_disk(root: Path) -> int:
    return len(list(root.glob("*.bin")))


def counting_scans(store: DiskStore, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every pass the store makes over its directory, by how many entries it saw."""
    scans: list[int] = []
    real = store._scan

    def counted() -> list[tuple[int, str]]:
        found = real()
        scans.append(len(found))
        return found

    monkeypatch.setattr(store, "_scan", counted)
    return scans


async def test_the_disk_store_is_at_or_under_its_ceiling_after_every_write(tmp_path: Path) -> None:
    """The bound that per-write eviction gave must survive eviction becoming occasional."""
    store = DiskStore(tmp_path, maximum_entries=20)
    most = 0

    for index in range(300):
        await store.set(f"k{index}", b"v")
        most = max(most, entries_on_disk(tmp_path))

    assert most == 20


@pytest.mark.parametrize("batch", [1, 7, 20, 21, 64])
async def test_the_disk_store_is_at_or_under_its_ceiling_after_every_batch(
    tmp_path: Path, batch: int
) -> None:
    store = DiskStore(tmp_path, maximum_entries=20)

    for start in range(0, 320, batch):
        await store.set_many({f"k{index}": b"v" for index in range(start, start + batch)})

        assert entries_on_disk(tmp_path) <= 20


async def test_the_disk_store_does_not_list_its_directory_on_every_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every write listed, stat-ed and sorted the whole cache to see if it was over."""
    store = DiskStore(tmp_path, maximum_entries=100)
    scans = counting_scans(store, monkeypatch)

    for index in range(1000):
        await store.set(f"k{index}", b"v")

    assert len(scans) <= 1000 // 10
    assert max(scans) <= 101
    assert entries_on_disk(tmp_path) <= 100


async def test_writes_under_the_ceiling_list_the_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pass to learn what an earlier process left behind, and none after it."""
    store = DiskStore(tmp_path, maximum_entries=1000)
    scans = counting_scans(store, monkeypatch)

    for index in range(200):
        await store.set(f"k{index}", b"v")
    await store.set_many({f"batch{index}": b"v" for index in range(200)})

    assert scans == [1]
    assert entries_on_disk(tmp_path) == 400


async def test_replacing_entries_at_a_full_ceiling_never_lists_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewrite adds no file, so it must not be counted as one and trigger a pass."""
    store = DiskStore(tmp_path, maximum_entries=5)
    await store.set_many({f"k{index}": b"old" for index in range(5)})
    scans = counting_scans(store, monkeypatch)

    for _ in range(50):
        await store.set_many({f"k{index}": b"new" for index in range(5)})

    assert scans == []
    assert await store.get_many([f"k{index}" for index in range(5)]) == [b"new"] * 5


async def test_an_eviction_frees_headroom_below_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evicting to the ceiling itself would put the very next write over it again."""
    store = DiskStore(tmp_path, maximum_entries=100)
    await store.set_many({f"k{index}": b"v" for index in range(100)})
    scans = counting_scans(store, monkeypatch)

    await store.set("one-too-many", b"v")
    after_eviction = entries_on_disk(tmp_path)
    for index in range(10):
        await store.set(f"room{index}", b"v")

    assert after_eviction == 90
    assert scans == [101]
    assert entries_on_disk(tmp_path) == 100


async def test_eviction_removes_the_least_recently_written_entries(tmp_path: Path) -> None:
    store = DiskStore(tmp_path, maximum_entries=10)
    for index in range(10):
        await store.set(f"k{index}", b"v")
        os.utime(store._path(f"k{index}"), (1_000_000 + index, 1_000_000 + index))

    await store.set("newest", b"v")

    survivors = await store.get_many([f"k{index}" for index in range(10)])
    assert survivors == [None, None] + [b"v"] * 8
    assert await store.get("newest") == b"v"


async def test_a_new_instance_over_a_full_directory_still_honours_the_ceiling(
    tmp_path: Path,
) -> None:
    """A count that started from zero in each process would let every restart add a ceiling."""
    earlier = DiskStore(tmp_path, maximum_entries=5)
    await earlier.set_many({f"k{index}": b"v" for index in range(5)})
    for index in range(5):
        os.utime(earlier._path(f"k{index}"), (1_000_000 + index, 1_000_000 + index))

    restarted = DiskStore(tmp_path, maximum_entries=5)
    await restarted.set("after-restart", b"v")

    assert entries_on_disk(tmp_path) == 5
    assert await restarted.get("after-restart") == b"v"


async def test_a_lowered_ceiling_trims_an_existing_directory_on_the_first_write(
    tmp_path: Path,
) -> None:
    await DiskStore(tmp_path).set_many({f"k{index}": b"v" for index in range(30)})

    await DiskStore(tmp_path, maximum_entries=4).set("k0", b"rewritten")

    assert entries_on_disk(tmp_path) == 4


async def test_clearing_forgets_the_count_as_well_as_the_files(tmp_path: Path) -> None:
    """A count left at the ceiling after a clear would evict from an empty directory."""
    store = DiskStore(tmp_path, maximum_entries=5)
    await store.set_many({f"old{index}": b"v" for index in range(5)})
    await store.clear()

    await store.set_many({f"new{index}": b"v" for index in range(5)})

    assert await store.get_many([f"new{index}" for index in range(5)]) == [b"v"] * 5


async def test_concurrent_writers_end_at_or_under_the_ceiling(tmp_path: Path) -> None:
    """Several embedding workers write at once; the count must not lose any of them."""
    store = DiskStore(tmp_path, maximum_entries=10)

    await asyncio.gather(
        *(store.set_many({f"w{worker}-{n}": b"v" for n in range(8)}) for worker in range(12))
    )

    assert entries_on_disk(tmp_path) <= 10
    assert list(tmp_path.glob("*.tmp")) == []


async def test_files_that_are_not_entries_are_neither_counted_nor_evicted(tmp_path: Path) -> None:
    store = DiskStore(tmp_path, maximum_entries=3)
    (tmp_path / "embedding").mkdir()
    (tmp_path / "notes.txt").write_text("not a cache entry")

    for index in range(10):
        await store.set(f"k{index}", b"v")

    assert entries_on_disk(tmp_path) == 3
    assert (tmp_path / "notes.txt").exists()
    assert (tmp_path / "embedding").is_dir()


async def test_an_entry_that_cannot_be_evicted_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Another process holding a file open must cost an entry of headroom, not the write."""
    store = DiskStore(tmp_path, maximum_entries=3)
    await store.set_many({f"k{index}": b"v" for index in range(3)})

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("the file is in use by another process")

    monkeypatch.setattr(Path, "unlink", refuse)
    with caplog.at_level(logging.WARNING):
        await store.set("one-too-many", b"v")

    assert await store.get("one-too-many") == b"v"
    assert "cache entry could not be removed" in caplog.text


async def test_an_expired_entry_that_cannot_be_removed_is_still_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unlink ran unguarded, so a locked file turned a miss into a failed lookup."""
    store = DiskStore(tmp_path)
    await store.set("k", b"value", ttl=1)
    time.sleep(1.05)

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("the file is in use by another process")

    monkeypatch.setattr(Path, "unlink", refuse)

    assert await store.get("k") is None
    assert await store.get_many(["k"]) == [None]
