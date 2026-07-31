import time
from pathlib import Path

import pytest

from fasterrag.core.cache.store import DiskStore, Entry, MemoryStore
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
