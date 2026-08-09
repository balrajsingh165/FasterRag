import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.cache import (
    EMBEDDING_NAMESPACE,
    SEMANTIC_NAMESPACE,
    RedisStore,
    create_embedding_store,
    create_semantic_store,
)
from fasterrag.errors import CacheError, ConfigError

NAMESPACE = "test:cache"


class FakeRedis:
    """The subset of the redis-py async client RedisStore actually uses.

    Replies come back as bytes and sorted-set members as bytes, as a client built with
    `decode_responses=False` returns them, so the store's decoding is exercised rather
    than bypassed.
    """

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.scores: dict[str, dict[str, float]] = {}
        self.expiries: dict[str, int | None] = {}
        self.closed = False

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        return [self.values.get(key) for key in keys]

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self.values[key] = value
        self.expiries[key] = ex

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            self.expiries.pop(key, None)
            if self.values.pop(key, None) is not None:
                removed += 1
            if self.scores.pop(key, None) is not None:
                removed += 1
        return removed

    async def zadd(self, index: str, mapping: dict[str, float]) -> int:
        self.scores.setdefault(index, {}).update(mapping)
        return len(mapping)

    async def zrem(self, index: str, *members: str) -> int:
        held = self.scores.get(index, {})
        return sum(1 for member in members if held.pop(member, None) is not None)

    async def zcard(self, index: str) -> int:
        return len(self.scores.get(index, {}))

    async def zrange(self, index: str, start: int, stop: int) -> list[bytes]:
        ordered = sorted(self.scores.get(index, {}).items(), key=lambda item: item[1])
        members = [member.encode("utf-8") for member, _ in ordered]
        return members[start:] if stop == -1 else members[start : stop + 1]

    async def aclose(self) -> None:
        self.closed = True


class BrokenRedis(FakeRedis):
    """Fails every command, as an unreachable server would."""

    async def get(self, key: str) -> bytes | None:
        raise ConnectionError("redis is down")

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        raise ConnectionError("redis is down")

    async def zadd(self, index: str, mapping: dict[str, float]) -> int:
        raise ConnectionError("redis is down")

    async def zrange(self, index: str, start: int, stop: int) -> list[bytes]:
        raise ConnectionError("redis is down")

    async def delete(self, *keys: str) -> int:
        raise ConnectionError("redis is down")

    async def aclose(self) -> None:
        raise ConnectionError("redis is down")


@pytest.fixture
def client() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def store(client: FakeRedis) -> RedisStore:
    return RedisStore("redis://localhost:6379/0", namespace=NAMESPACE, client=client)


async def test_a_stored_value_reads_back(store: RedisStore) -> None:
    await store.set("k", b"value")

    assert await store.get("k") == b"value"


async def test_an_absent_key_is_a_miss(store: RedisStore) -> None:
    assert await store.get("never-written") is None


async def test_a_value_is_replaced_rather_than_appended(store: RedisStore) -> None:
    await store.set("k", b"first")
    await store.set("k", b"second")

    assert await store.get("k") == b"second"


async def test_an_empty_value_round_trips(store: RedisStore) -> None:
    await store.set("k", b"")

    assert await store.get("k") == b""


async def test_deleting_removes_the_entry(store: RedisStore) -> None:
    await store.set("k", b"value")
    await store.delete("k")

    assert await store.get("k") is None
    assert await store.items() == []


async def test_deleting_an_absent_key_is_not_an_error(store: RedisStore) -> None:
    await store.delete("never-written")


async def test_clearing_removes_everything(store: RedisStore) -> None:
    await store.set("a", b"1")
    await store.set("b", b"2")
    await store.clear()

    assert await store.items() == []


async def test_clearing_leaves_keys_outside_the_namespace_untouched(
    store: RedisStore, client: FakeRedis
) -> None:
    """A shared server is the normal case, so `clear` must never behave like FLUSHDB."""
    client.values["someone-elses:key"] = b"not ours"
    await store.set("k", b"value")
    await store.clear()

    assert client.values == {"someone-elses:key": b"not ours"}


async def test_an_expired_entry_is_a_miss(store: RedisStore) -> None:
    await store.set("k", b"value", ttl=1)
    time.sleep(1.05)

    assert await store.get("k") is None


async def test_a_live_ttl_entry_is_a_hit(store: RedisStore) -> None:
    await store.set("k", b"value", ttl=60)

    assert await store.get("k") == b"value"


async def test_no_ttl_means_no_expiry(store: RedisStore) -> None:
    await store.set("k", b"value")

    assert await store.get("k") == b"value"


async def test_a_ttl_is_also_handed_to_redis(store: RedisStore, client: FakeRedis) -> None:
    """The payload deadline is authoritative; the native expiry reclaims the memory."""
    await store.set("k", b"value", ttl=60)

    assert client.expiries[f"{NAMESPACE}:entry:k"] == 60


async def test_no_ttl_leaves_the_key_without_a_native_expiry(
    store: RedisStore, client: FakeRedis
) -> None:
    await store.set("k", b"value")

    assert client.expiries[f"{NAMESPACE}:entry:k"] is None


async def test_items_returns_every_live_entry(store: RedisStore) -> None:
    await store.set("a", b"1")
    await store.set("b", b"2")

    assert sorted(value for _, value in await store.items()) == [b"1", b"2"]


async def test_items_returns_the_original_keys(store: RedisStore) -> None:
    await store.set("sem:-:question", b"1")

    assert [key for key, _ in await store.items()] == ["sem:-:question"]


async def test_items_omits_expired_entries(store: RedisStore) -> None:
    await store.set("live", b"1", ttl=60)
    await store.set("dead", b"2", ttl=1)
    time.sleep(1.05)

    assert [value for _, value in await store.items()] == [b"1"]


async def test_items_prunes_the_index_of_entries_redis_dropped(
    store: RedisStore, client: FakeRedis
) -> None:
    """A natively expired key leaves its member behind; the next scan must clean it up."""
    await store.set("k", b"value", ttl=60)
    del client.values[f"{NAMESPACE}:entry:k"]

    assert await store.items() == []
    assert client.scores[f"{NAMESPACE}:index"] == {}


async def test_a_key_with_path_separators_is_usable(store: RedisStore) -> None:
    key = "emb:sentence-transformers/all-MiniLM-L6-v2:1.0:abcdef"
    await store.set(key, b"value")

    assert await store.get(key) == b"value"


async def test_a_corrupt_payload_is_a_miss_rather_than_a_crash(
    store: RedisStore, client: FakeRedis
) -> None:
    await store.set("k", b"value")
    client.values[f"{NAMESPACE}:entry:k"] = b"xx"

    assert await store.get("k") is None


async def test_a_corrupt_payload_is_dropped_from_items(
    store: RedisStore, client: FakeRedis
) -> None:
    await store.set("k", b"value")
    client.values[f"{NAMESPACE}:entry:k"] = b"xx"

    assert await store.items() == []


async def test_the_store_never_exceeds_its_ceiling(client: FakeRedis) -> None:
    store = RedisStore("redis://localhost", namespace=NAMESPACE, maximum_entries=3, client=client)
    for index in range(20):
        await store.set(f"k{index}", b"v")

    assert len(await store.items()) == 3


async def test_eviction_drops_the_least_recently_used(client: FakeRedis) -> None:
    store = RedisStore("redis://localhost", namespace=NAMESPACE, maximum_entries=2, client=client)
    await store.set("a", b"1")
    await store.set("b", b"2")
    await store.get("a")
    await store.set("c", b"3")

    assert await store.get("b") is None
    assert await store.get("a") == b"1"
    assert await store.get("c") == b"3"


async def test_eviction_removes_the_value_as_well_as_the_index_member(
    client: FakeRedis,
) -> None:
    """An evicted value left behind would be unreachable and never counted again."""
    store = RedisStore("redis://localhost", namespace=NAMESPACE, maximum_entries=1, client=client)
    await store.set("a", b"1")
    await store.set("b", b"2")

    assert list(client.values) == [f"{NAMESPACE}:entry:b"]


async def test_two_stores_on_one_server_do_not_see_each_other(client: FakeRedis) -> None:
    embedding = RedisStore("redis://localhost", namespace=EMBEDDING_NAMESPACE, client=client)
    semantic = RedisStore("redis://localhost", namespace=SEMANTIC_NAMESPACE, client=client)
    await embedding.set("k", b"vector")
    await semantic.set("k", b"answer")
    await semantic.clear()

    assert await embedding.get("k") == b"vector"
    assert await semantic.get("k") is None


async def test_closing_releases_the_client(store: RedisStore, client: FakeRedis) -> None:
    await store.close()

    assert client.closed is True


async def test_closing_a_downed_backend_does_not_raise() -> None:
    """Shutdown must not become a traceback over a connection being dropped anyway."""
    store = RedisStore("redis://localhost", namespace=NAMESPACE, client=BrokenRedis())

    await store.close()


@pytest.mark.parametrize("operation", ["get", "set", "delete", "clear", "items"])
async def test_a_downed_backend_raises_a_typed_cache_error(operation: str) -> None:
    """Callers degrade to cache-off on CacheError alone; anything else fails the query."""
    store = RedisStore("redis://localhost", namespace=NAMESPACE, client=BrokenRedis())
    calls: dict[str, Callable[[], Awaitable[object]]] = {
        "get": lambda: store.get("k"),
        "set": lambda: store.set("k", b"v"),
        "delete": lambda: store.delete("k"),
        "clear": lambda: store.clear(),
        "items": lambda: store.items(),
    }

    with pytest.raises(CacheError):
        await calls[operation]()


async def test_a_cache_error_never_echoes_the_connection_url() -> None:
    """A URL can carry a password, and this message reaches the logs."""
    store = RedisStore("redis://user:hunter2@localhost", namespace=NAMESPACE, client=BrokenRedis())

    with pytest.raises(CacheError) as raised:
        await store.get("k")

    assert "hunter2" not in str(raised.value)


def test_a_malformed_url_fails_at_construction_naming_the_setting() -> None:
    pytest.importorskip("redis")

    with pytest.raises(ConfigError, match=r"embeddings\.cache\.backend"):
        RedisStore(
            "redis://host:not-a-port",
            namespace=NAMESPACE,
            setting="embeddings.cache.backend",
        )


def settings(**sections: object) -> Settings:
    return Settings.model_validate(sections)


def test_the_semantic_factory_builds_a_namespaced_redis_store() -> None:
    pytest.importorskip("redis")
    store = create_semantic_store(
        settings(cache={"backend": "redis", "max_entries": 7, "redis_url": "redis://host:6380/1"})
    )

    assert isinstance(store, RedisStore)
    assert store._namespace == SEMANTIC_NAMESPACE
    assert store._maximum == 7


def test_the_embedding_factory_builds_a_namespaced_redis_store() -> None:
    pytest.importorskip("redis")
    store = create_embedding_store(settings(embeddings={"cache": {"backend": "redis"}}))

    assert isinstance(store, RedisStore)
    assert store._namespace == EMBEDDING_NAMESPACE


def test_selecting_redis_without_the_client_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one error an operator sees when the optional dependency is missing."""
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("redis"):
            raise ImportError("no module named redis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(ConfigError, match=r"fasterrag\[redis\]"):
        create_semantic_store(settings(cache={"backend": "redis"}))
