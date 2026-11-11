"""`RedisRateLimiter` against a Redis server process, which is the half fakeredis cannot be.

`test_redis_rate_limit.py` next door drives the real `redis.asyncio.Redis` client over
fakeredis, so the command encoding and the MULTI/EXEC round trip are the shipped ones and
only the socket is replaced. What it cannot prove is that a *server build an operator runs*
agrees with fakeredis about what `INCR`, `EXPIRE` and key eviction then do — that is the
same assumption restated, and TASK-0290 stayed open for it.

This module closes it by talking to a real `redis-server` over a real socket.
`deploy/compose.test-backends.yml` publishes one on the default port, so no environment
variable is needed against that stack::

    docker compose -f deploy/compose.test-backends.yml up -d --wait redis
    pytest -m integration tests/integration/test_redis_rate_limit_server.py -rs

Every case runs its own namespace, because a shared server is shared: scanning `*` on a
database somebody else is using would read their keys and a fixed key name would collide
with a concurrent run. The namespace is what makes these cases safe to point at a server
that is not empty, and it is also why they do not call `FLUSHDB`.

Still not proven here: timeouts, resets, partitions, failover and pool exhaustion — the
fail-open path is reachable only by injecting an error, exactly as before; and the two
limiters sharing a ceiling below share a server but still live in one interpreter, so this
is not the cross-process sharing the feature exists for. `docs/security.md` §2 carries the
same split.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import redis.asyncio
from fakeredis import FakeAsyncRedis, FakeServer
from redis.exceptions import RedisError

from fasterrag.core.rate_limit import WINDOW_SECONDS, RedisRateLimiter

pytestmark = pytest.mark.integration

URL_VAR = "FASTERRAG_TEST_REDIS_URL"
DEFAULT_URL = "redis://localhost:6379/0"

CALLER = "caller-identity-never-sent-to-the-server"
OTHER_CALLER = "a-second-caller-entirely"

# CRITICAL: bucket-aligned arithmetic depends on this being a plain second, so that
# FIXED_NOW and FIXED_NOW + WINDOW_SECONDS are guaranteed to be adjacent windows.
FIXED_NOW = 1_762_000_000.0

_PROBE_TIMEOUT_SECONDS = 5.0

LimiterFactory = Callable[..., RedisRateLimiter]


def _text(key: bytes | str) -> str:
    """Return a key name as text, whichever of the two the client decoding hands back."""
    return key if isinstance(key, str) else key.decode("utf-8")


@pytest.fixture(scope="session")
def redis_url() -> str:
    """Return the server this module talks to."""
    return os.environ.get(URL_VAR, DEFAULT_URL)


@pytest.fixture(scope="session")
def redis_server(redis_url: str) -> str:
    """Return a reachable server's URL, skipping with the reason when there is none."""

    # CRITICAL: the probe reports every failure as a skip reason rather than raising. An
    # unreachable server must skip this module, never fail collection from a fixture.
    async def ping() -> str | None:
        client = redis.asyncio.Redis.from_url(
            redis_url,
            socket_connect_timeout=_PROBE_TIMEOUT_SECONDS,
            socket_timeout=_PROBE_TIMEOUT_SECONDS,
        )
        try:
            await client.ping()
        except (RedisError, OSError) as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            await client.aclose()
        return None

    unreachable = asyncio.run(ping())
    if unreachable is not None:
        pytest.skip(f"no Redis at {redis_url} ({unreachable})")
    return redis_url


@pytest.fixture
def namespace() -> str:
    """Return a key prefix no other case or concurrent run can collide with."""
    return f"fasterrag:test:ratelimit:{uuid.uuid4().hex}"


@pytest.fixture
async def observer(redis_server: str, namespace: str) -> AsyncIterator[redis.asyncio.Redis]:
    """Yield a client used to read back what the limiter actually wrote, and to clean up."""
    client: redis.asyncio.Redis = redis.asyncio.Redis.from_url(redis_server)
    yield client
    found = list(await client.keys(f"{namespace}:*"))
    if found:
        await client.delete(*found)
    await client.aclose()


@pytest.fixture
async def limiters(redis_server: str, namespace: str) -> AsyncIterator[LimiterFactory]:
    """Yield a factory building real limiters over the real server, in one namespace."""
    built: list[RedisRateLimiter] = []

    def make(limit: int, window_seconds: float = WINDOW_SECONDS) -> RedisRateLimiter:
        made = RedisRateLimiter(
            redis_server,
            limit=limit,
            window_seconds=window_seconds,
            namespace=namespace,
        )
        built.append(made)
        return made

    yield make
    for made in built:
        await made.close()


async def test_the_limiter_opened_a_real_socket_to_a_real_server(
    limiters: LimiterFactory, redis_server: str
) -> None:
    """Every other case here is worth nothing if this is fakeredis under another name."""
    limiter = limiters(1)

    assert isinstance(limiter._client, redis.asyncio.Redis)
    assert not isinstance(limiter._client, FakeAsyncRedis)
    assert (await limiter._client.ping()) is True


async def test_the_ceiling_is_enforced_at_the_limit(limiters: LimiterFactory) -> None:
    """N requests pass and N+1 does not, decided by a real server's own INCR replies."""
    limiter = limiters(3)

    decisions = [await limiter.allow(CALLER, now=FIXED_NOW) for _ in range(4)]

    assert [decision.allowed for decision in decisions] == [True, True, True, False]
    assert decisions[-1].retry_after >= 1


async def test_a_refusal_is_not_permanent_once_the_window_rolls(limiters: LimiterFactory) -> None:
    """The next window is a different key, so the count restarts rather than being reset."""
    limiter = limiters(2)
    for _ in range(2):
        assert (await limiter.allow(CALLER, now=FIXED_NOW)).allowed
    assert not (await limiter.allow(CALLER, now=FIXED_NOW)).allowed

    assert (await limiter.allow(CALLER, now=FIXED_NOW + WINDOW_SECONDS)).allowed


async def test_the_server_expires_the_counter_itself(
    limiters: LimiterFactory, observer: redis.asyncio.Redis
) -> None:
    """An immortal counter at its ceiling refuses a key forever; the EXPIRE must land."""
    limiter = limiters(1, 0.5)
    await limiter.allow(CALLER, now=FIXED_NOW)
    key = limiter._key(CALLER, FIXED_NOW)

    assert 0 < await observer.ttl(key) <= 1

    await asyncio.sleep(1.5)

    assert await observer.exists(key) == 0


async def test_two_limiters_sharing_a_backend_share_one_ceiling(
    limiters: LimiterFactory,
) -> None:
    """The entire reason the backend exists: N replicas grant the limit, not N times it."""
    first = limiters(4)
    second = limiters(4)
    assert first._client is not second._client

    for limiter in (first, second, first, second):
        assert (await limiter.allow(CALLER, now=FIXED_NOW)).allowed

    assert not (await second.allow(CALLER, now=FIXED_NOW)).allowed
    assert not (await first.allow(CALLER, now=FIXED_NOW)).allowed


async def test_two_limiters_still_count_two_callers_apart(limiters: LimiterFactory) -> None:
    """A shared ceiling must be shared per identity, not collapsed into one global one."""
    first = limiters(1)
    second = limiters(1)

    assert (await first.allow(CALLER, now=FIXED_NOW)).allowed

    assert (await second.allow(OTHER_CALLER, now=FIXED_NOW)).allowed
    assert not (await second.allow(CALLER, now=FIXED_NOW)).allowed


async def test_the_key_carries_a_fingerprint_and_never_the_api_key(
    limiters: LimiterFactory, observer: redis.asyncio.Redis, namespace: str
) -> None:
    """Read back off the server, because the credential disclosure would be server-side."""
    limiter = limiters(5)
    await limiter.allow(CALLER, now=FIXED_NOW)

    stored = [_text(key) for key in await observer.keys(f"{namespace}:*")]

    fingerprint = hashlib.sha256(CALLER.encode("utf-8")).hexdigest()[:32]
    bucket = int(FIXED_NOW // WINDOW_SECONDS)
    assert stored == [f"{namespace}:{fingerprint}:{bucket}"]
    assert CALLER not in stored[0]


async def test_the_transaction_returns_what_the_unit_double_says_it_does(
    limiters: LimiterFactory,
) -> None:
    """The reply shapes the unit suite asserts against a double, read off a server."""
    limiter = limiters(10)
    key = limiter._key(CALLER, FIXED_NOW)

    pipeline = limiter._client.pipeline(transaction=True)
    pipeline.incr(key)
    pipeline.expire(key, 61)
    results = await pipeline.execute()

    assert results == [1, True]


async def test_the_real_server_and_fakeredis_agree_on_every_decision(
    limiters: LimiterFactory, namespace: str
) -> None:
    """A double that diverges from the thing it doubles is worse than no double at all.

    The unit suite's whole claim rests on fakeredis answering as redis-server does, and
    nothing until now compared the two.
    """
    against_server = limiters(3)
    against_fake = RedisRateLimiter(
        "redis://localhost:6379/0",
        limit=3,
        namespace=namespace,
        client=FakeAsyncRedis(server=FakeServer()),
    )

    try:
        moments = [FIXED_NOW] * 5 + [FIXED_NOW + WINDOW_SECONDS] * 2
        from_server = [await against_server.allow(CALLER, now=at) for at in moments]
        from_fake = [await against_fake.allow(CALLER, now=at) for at in moments]
    finally:
        await against_fake.close()

    assert [(d.allowed, d.retry_after) for d in from_server] == [
        (d.allowed, d.retry_after) for d in from_fake
    ]
