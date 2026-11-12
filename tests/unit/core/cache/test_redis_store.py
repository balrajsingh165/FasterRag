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

    ``round_trips`` counts what a network would: one per command sent on its own, and one
    per executed pipeline however many commands it carried.
    """

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.scores: dict[str, dict[str, float]] = {}
        self.counters: dict[str, int] = {}
        self.expiries: dict[str, int | None] = {}
        self.round_trips = 0
        self.closed = False

    def pipeline(self, transaction: bool = True) -> "FakePipeline":
        return FakePipeline(self, transaction)

    async def get(self, key: str) -> bytes | None:
        self.round_trips += 1
        return self.values.get(key)

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        self.round_trips += 1
        return [self.values.get(key) for key in keys]

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self.round_trips += 1
        self.values[key] = value
        self.expiries[key] = ex

    async def delete(self, *keys: str) -> int:
        self.round_trips += 1
        removed = 0
        for key in keys:
            self.expiries.pop(key, None)
            if self.values.pop(key, None) is not None:
                removed += 1
            if self.scores.pop(key, None) is not None:
                removed += 1
        return removed

    async def incr(self, key: str) -> int:
        return await self.incrby(key, 1)

    async def incrby(self, key: str, amount: int) -> int:
        self.round_trips += 1
        self.counters[key] = self.counters.get(key, 0) + amount
        return self.counters[key]

    async def zadd(self, index: str, mapping: dict[str, float]) -> int:
        self.round_trips += 1
        self.scores.setdefault(index, {}).update(mapping)
        return len(mapping)

    async def zrem(self, index: str, *members: str) -> int:
        self.round_trips += 1
        held = self.scores.get(index, {})
        return sum(1 for member in members if held.pop(member, None) is not None)

    async def zcard(self, index: str) -> int:
        self.round_trips += 1
        return len(self.scores.get(index, {}))

    async def zrange(self, index: str, start: int, stop: int) -> list[bytes]:
        self.round_trips += 1
        ordered = sorted(self.scores.get(index, {}).items(), key=lambda item: item[1])
        members = [member.encode("utf-8") for member, _ in ordered]
        return members[start:] if stop == -1 else members[start : stop + 1]

    async def aclose(self) -> None:
        self.closed = True


class FakePipeline:
    """Buffers commands and sends them as one round trip, in the order they were queued."""

    def __init__(self, server: FakeRedis, transaction: bool) -> None:
        self.server = server
        self.transaction = transaction
        self.queued: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _queue(self, name: str, *args: Any, **kwargs: Any) -> "FakePipeline":
        self.queued.append((name, args, kwargs))
        return self

    def mget(self, keys: list[str]) -> "FakePipeline":
        return self._queue("mget", keys)

    def incrby(self, key: str, amount: int) -> "FakePipeline":
        return self._queue("incrby", key, amount)

    def zadd(self, index: str, mapping: dict[str, float]) -> "FakePipeline":
        return self._queue("zadd", index, mapping)

    def zrem(self, index: str, *members: str) -> "FakePipeline":
        return self._queue("zrem", index, *members)

    def zcard(self, index: str) -> "FakePipeline":
        return self._queue("zcard", index)

    def set(self, key: str, value: bytes, ex: int | None = None) -> "FakePipeline":
        return self._queue("set", key, value, ex=ex)

    def delete(self, *keys: str) -> "FakePipeline":
        return self._queue("delete", *keys)

    async def execute(self) -> list[Any]:
        before = self.server.round_trips
        try:
            return [
                await getattr(self.server, name)(*args, **kwargs)
                for name, args, kwargs in self.queued
            ]
        finally:
            self.server.round_trips = before + 1


class BrokenRedis(FakeRedis):
    """Fails every command, as an unreachable server would."""

    async def get(self, key: str) -> bytes | None:
        raise ConnectionError("redis is down")

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        raise ConnectionError("redis is down")

    async def incr(self, key: str) -> int:
        raise ConnectionError("redis is down")

    async def incrby(self, key: str, amount: int) -> int:
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


@pytest.mark.parametrize(
    "operation", ["get", "get_many", "set", "set_many", "delete", "clear", "items"]
)
async def test_a_downed_backend_raises_a_typed_cache_error(operation: str) -> None:
    """Callers degrade to cache-off on CacheError alone; anything else fails the query."""
    store = RedisStore("redis://localhost", namespace=NAMESPACE, client=BrokenRedis())
    calls: dict[str, Callable[[], Awaitable[object]]] = {
        "get": lambda: store.get("k"),
        "get_many": lambda: store.get_many(["k", "j"]),
        "set": lambda: store.set("k", b"v"),
        "set_many": lambda: store.set_many({"k": b"v", "j": b"w"}),
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


async def test_eviction_is_least_recently_used_even_on_a_clock_that_never_ticks(
    client: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recency must not depend on the platform's clock resolution."""
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    store = RedisStore("redis://localhost", namespace=NAMESPACE, maximum_entries=2, client=client)

    await store.set("a", b"1")
    await store.set("b", b"2")
    await store.get("a")
    await store.set("c", b"3")

    assert await store.get("b") is None, "the least recently used entry survived"
    assert await store.get("a") == b"1", "a recently used entry was evicted"
    assert await store.get("c") == b"3"


async def test_a_batch_reads_back_in_the_order_it_was_asked_for(store: RedisStore) -> None:
    await store.set_many({"a": b"1", "b": b"2", "c": b"3"})

    assert await store.get_many(["c", "absent", "a", "b"]) == [b"3", None, b"1", b"2"]


async def test_a_batch_write_is_readable_one_key_at_a_time(store: RedisStore) -> None:
    await store.set_many({"a": b"1", "b": b"2"})

    assert await store.get("a") == b"1"
    assert sorted(key for key, _ in await store.items()) == ["a", "b"]


async def test_an_empty_batch_sends_nothing(store: RedisStore, client: FakeRedis) -> None:
    await store.set_many({})

    assert await store.get_many([]) == []
    assert client.round_trips == 0


@pytest.mark.parametrize("size", [1, 64, 500])
async def test_a_batch_read_costs_two_round_trips_whatever_its_size(
    store: RedisStore, client: FakeRedis, size: int
) -> None:
    """One key at a time, a batch of sixty-four was over a hundred round trips."""
    keys = [f"k{index}" for index in range(size)]
    await store.set_many(dict.fromkeys(keys[::2], b"v"))
    client.round_trips = 0

    values = await store.get_many(keys)

    assert values == [b"v" if index % 2 == 0 else None for index in range(size)]
    assert client.round_trips == 2


@pytest.mark.parametrize("size", [1, 64, 500])
async def test_a_batch_write_costs_two_round_trips_whatever_its_size(
    store: RedisStore, client: FakeRedis, size: int
) -> None:
    await store.set_many({f"k{index}": b"v" for index in range(size)})

    assert client.round_trips == 2
    assert len(await store.items()) == size


async def test_a_batch_write_hands_each_ttl_to_redis(store: RedisStore, client: FakeRedis) -> None:
    await store.set_many({"a": b"1", "b": b"2"}, ttl=60)

    assert client.expiries[f"{NAMESPACE}:entry:a"] == 60
    assert client.expiries[f"{NAMESPACE}:entry:b"] == 60


async def test_a_batch_read_treats_an_expired_entry_as_a_miss_and_prunes_it(
    store: RedisStore, client: FakeRedis
) -> None:
    await store.set_many({"dead": b"1"}, ttl=1)
    await store.set_many({"live": b"2"}, ttl=60)
    time.sleep(1.05)

    assert await store.get_many(["dead", "live"]) == [None, b"2"]
    assert f"{NAMESPACE}:entry:dead" not in client.values
    assert list(client.scores[f"{NAMESPACE}:index"]) == ["live"]


async def test_a_batch_read_prunes_the_index_of_entries_redis_dropped(
    store: RedisStore, client: FakeRedis
) -> None:
    """A natively expired key leaves its member behind; a batch read must clean it up too."""
    await store.set_many({"k": b"value"}, ttl=60)
    del client.values[f"{NAMESPACE}:entry:k"]

    assert await store.get_many(["k"]) == [None]
    assert client.scores[f"{NAMESPACE}:index"] == {}


async def test_a_batch_write_never_exceeds_the_ceiling(client: FakeRedis) -> None:
    store = RedisStore("redis://localhost", namespace=NAMESPACE, maximum_entries=3, client=client)

    await store.set_many({f"k{index}": b"v" for index in range(20)})

    assert sorted(key for key, _ in await store.items()) == ["k17", "k18", "k19"]
    assert len(client.values) == 3


async def test_a_batch_read_restacks_its_hits_as_recently_used(client: FakeRedis) -> None:
    """Eviction after a batch read must spare what the batch just used."""
    store = RedisStore("redis://localhost", namespace=NAMESPACE, maximum_entries=3, client=client)
    await store.set_many({"a": b"1", "b": b"2", "c": b"3"})
    await store.get_many(["a", "b"])

    await store.set_many({"d": b"4"})

    assert await store.get_many(["a", "b", "c", "d"]) == [b"1", b"2", None, b"4"]


async def test_a_batch_is_piped_without_a_transaction(
    store: RedisStore, client: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Index-first ordering is the safety argument; MULTI would only add a failure mode."""
    opened: list[bool] = []
    real = client.pipeline

    def recording(transaction: bool = True) -> FakePipeline:
        opened.append(transaction)
        return real(transaction)

    monkeypatch.setattr(client, "pipeline", recording)

    await store.set_many({"a": b"1"})
    await store.get_many(["a"])

    assert opened == [False, False, False]


async def test_a_batch_write_indexes_each_key_before_writing_its_value(
    store: RedisStore, client: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value with no index member is invisible to the ceiling, to items and to clear."""
    order: list[str] = []
    real_zadd, real_set = client.zadd, client.set

    async def zadd(index: str, mapping: dict[str, float]) -> int:
        order.append("index")
        return await real_zadd(index, mapping)

    async def set_value(key: str, value: bytes, ex: int | None = None) -> None:
        order.append("value")
        await real_set(key, value, ex=ex)

    monkeypatch.setattr(client, "zadd", zadd)
    monkeypatch.setattr(client, "set", set_value)

    await store.set_many({"a": b"1", "b": b"2"})

    assert order == ["index", "value", "value"]


async def test_a_single_hit_costs_two_round_trips(store: RedisStore, client: FakeRedis) -> None:
    """A hit was GET, INCR and ZADD sent one at a time: three round trips (TASK-0453)."""
    await store.set("k", b"value")
    client.round_trips = 0

    assert await store.get("k") == b"value"
    assert client.round_trips == 2


async def test_a_single_miss_costs_two_round_trips(store: RedisStore, client: FakeRedis) -> None:
    await store.get("never-written")

    assert client.round_trips == 2


async def test_a_single_write_costs_two_round_trips(store: RedisStore, client: FakeRedis) -> None:
    """A write was INCR, ZADD, SET and ZCARD sent one at a time: four round trips."""
    await store.set("k", b"value", ttl=60)

    assert client.round_trips == 2
    assert client.expiries[f"{NAMESPACE}:entry:k"] == 60
