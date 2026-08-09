"""The redis cache backend against a real Redis, not a fake.

The unit suite proves the store issues the right commands; only a real server proves those
commands mean what the store assumes. Three things in particular cannot be faked honestly:
Redis' own key expiry, whether a second process genuinely sees the first one's entries —
the entire reason this backend exists — and whether `clear` leaves a co-tenant's keys alone.
"""

from collections.abc import AsyncIterator

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.cache import RedisStore, create_semantic_store
from fasterrag.core.cache.semantic import SemanticCache

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(redis_url: str, redis_namespace: str) -> AsyncIterator[RedisStore]:
    built = RedisStore(redis_url, namespace=redis_namespace)
    yield built
    await built.clear()
    await built.close()


async def test_a_value_round_trips_through_a_real_server(store: RedisStore) -> None:
    await store.set("k", b"value")

    assert await store.get("k") == b"value"


async def test_a_binary_payload_survives_the_round_trip(store: RedisStore) -> None:
    """Vectors are packed float32; a client that decoded replies would mangle every one."""
    payload = bytes(range(256))
    await store.set("k", payload)

    assert await store.get("k") == payload


async def test_a_second_store_sees_the_first_one_s_entries(
    redis_url: str, redis_namespace: str, store: RedisStore
) -> None:
    """The point of the backend: a cache shared across processes rather than per-process."""
    await store.set("k", b"value")

    other = RedisStore(redis_url, namespace=redis_namespace)
    try:
        assert await other.get("k") == b"value"
    finally:
        await other.close()


async def test_redis_expires_the_key_itself(store: RedisStore, redis_url: str) -> None:
    """The payload deadline is authoritative, but the key must not outlive it in memory."""
    redis = pytest.importorskip("redis.asyncio")
    await store.set("k", b"value", ttl=30)

    client = redis.Redis.from_url(redis_url, decode_responses=False)
    try:
        remaining = await client.ttl(f"{store._namespace}:entry:k")
    finally:
        await client.aclose()

    assert 0 < remaining <= 30


async def test_an_entry_without_a_ttl_never_expires(store: RedisStore, redis_url: str) -> None:
    redis = pytest.importorskip("redis.asyncio")
    await store.set("k", b"value")

    client = redis.Redis.from_url(redis_url, decode_responses=False)
    try:
        remaining = await client.ttl(f"{store._namespace}:entry:k")
    finally:
        await client.aclose()

    assert remaining == -1


async def test_the_ceiling_is_enforced_against_a_real_server(
    redis_url: str, redis_namespace: str
) -> None:
    """A ceiling nothing enforces is a memory leak with a number next to it."""
    store = RedisStore(redis_url, namespace=redis_namespace, maximum_entries=5)
    try:
        for index in range(40):
            await store.set(f"k{index}", b"v")

        assert len(await store.items()) == 5
    finally:
        await store.clear()
        await store.close()


async def test_eviction_takes_the_least_recently_used(redis_url: str, redis_namespace: str) -> None:
    store = RedisStore(redis_url, namespace=redis_namespace, maximum_entries=2)
    try:
        await store.set("a", b"1")
        await store.set("b", b"2")
        await store.get("a")
        await store.set("c", b"3")

        assert await store.get("b") is None
        assert await store.get("a") == b"1"
    finally:
        await store.clear()
        await store.close()


async def test_clearing_leaves_another_namespace_alone(
    redis_url: str, redis_namespace: str, store: RedisStore
) -> None:
    """One server backs both caches, and may back an unrelated application too."""
    neighbour = RedisStore(redis_url, namespace=f"{redis_namespace}:neighbour")
    try:
        await neighbour.set("k", b"theirs")
        await store.set("k", b"ours")
        await store.clear()

        assert await neighbour.get("k") == b"theirs"
        assert await store.get("k") is None
    finally:
        await neighbour.clear()
        await neighbour.close()


async def test_the_semantic_cache_runs_on_a_real_redis(redis_url: str) -> None:
    """End to end through the factory, the way a configured deployment reaches it."""
    settings = Settings.model_validate(
        {"cache": {"semantic": True, "backend": "redis", "redis_url": redis_url}}
    )
    store = create_semantic_store(settings)
    cache = SemanticCache(settings, store)
    try:
        await cache.store_response("what is the notice period?", [1.0, 0.0], {"answer": "30 days"})
        hit = await cache.lookup([1.0, 0.001])

        assert hit is not None
        assert hit.response == {"answer": "30 days"}

        await cache.invalidate("corpus changed")

        assert await cache.lookup([1.0, 0.001]) is None
    finally:
        await store.clear()
        await cache.close()
