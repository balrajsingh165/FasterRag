"""The real `RedisRateLimiter` against the real redis-py client and a real Redis protocol.

The unit suite drives a hand-written double. That double can prove which commands this
module issues and that they travel in one transaction, because those are facts about our
code — and it cannot prove that redis-py and a server agree with it about what the commands
then *do*, because it is the same assumption restated.

fakeredis replaces only the socket. The client is `redis.asyncio.Redis`, so the command
encoding, the `MULTI`/`EXEC` round trip and the reply parsing are the shipped ones, and
`INCR`, `EXPIRE` and the eviction of an expired key are performed by a server
implementation rather than by this repository's idea of one. Every assertion below reads
state back out of that server instead of out of a recording.

What it still does not prove, and why TASK-0290 stays open: no real Redis release is
exercised — fakeredis emulates one and may diverge from any build an operator runs; nothing
crosses a socket, so timeouts, resets, partitions, failover and pool exhaustion are all
untested, and the fail-open path those produce is still only reachable here by injecting an
error; and the two clients sharing one ceiling below share a Python object inside one
interpreter, which is the strongest form of sharing this machine can offer and is not the
cross-process sharing the feature exists for. `docs/security.md` §2 states the same split.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Callable

import pytest
import redis.asyncio
from fakeredis import FakeAsyncRedis, FakeServer

from fasterrag.core.rate_limit import (
    RATE_LIMIT_NAMESPACE,
    WINDOW_SECONDS,
    RedisRateLimiter,
)

pytestmark = pytest.mark.integration

SECRET = "sk-live-an-api-key-that-must-never-reach-the-server"
OTHER_SECRET = "sk-live-a-second-caller-entirely"

# CRITICAL: bucket-aligned arithmetic depends on this being a plain second, so that
# FIXED_NOW and FIXED_NOW + WINDOW_SECONDS are guaranteed to be adjacent windows.
FIXED_NOW = 1_762_000_000.0

LimiterFactory = Callable[..., RedisRateLimiter]


def _text(key: bytes | str) -> str:
    """Return a key name as text, whichever of the two the client decoding hands back."""
    return key if isinstance(key, str) else key.decode("utf-8")


@pytest.fixture
def server() -> FakeServer:
    """Return the one server object every client in a case talks to."""
    return FakeServer()


@pytest.fixture
async def client(server: FakeServer) -> AsyncIterator[FakeAsyncRedis]:
    """Yield an observer client, used to read back what the limiter actually wrote."""
    built = FakeAsyncRedis(server=server)
    yield built
    await built.aclose()


@pytest.fixture
async def limiters(server: FakeServer) -> AsyncIterator[LimiterFactory]:
    """Yield a factory building real limiters over the case's one server."""
    built: list[RedisRateLimiter] = []

    def make(limit: int, window_seconds: float = WINDOW_SECONDS) -> RedisRateLimiter:
        made = RedisRateLimiter(
            "redis://localhost:6379/0",
            limit=limit,
            window_seconds=window_seconds,
            client=FakeAsyncRedis(server=server),
        )
        built.append(made)
        return made

    yield make
    for made in built:
        await made.close()


async def test_the_limiter_is_driving_the_real_redis_py_client(limiters: LimiterFactory) -> None:
    """Every other case in this file is worth nothing if the client is a stand-in."""
    limiter = limiters(1)

    assert isinstance(limiter._client, redis.asyncio.Redis)


async def test_the_ceiling_is_enforced_at_the_limit(limiters: LimiterFactory) -> None:
    """N requests pass and N+1 does not, decided by a server's own INCR replies."""
    limiter = limiters(3)

    decisions = [await limiter.allow(SECRET, now=FIXED_NOW) for _ in range(4)]

    assert [decision.allowed for decision in decisions] == [True, True, True, False]
    assert decisions[-1].retry_after >= 1


async def test_a_refusal_is_not_permanent_once_the_window_rolls(limiters: LimiterFactory) -> None:
    """The next window is a different key, so the count restarts rather than being reset."""
    limiter = limiters(2)
    for _ in range(2):
        assert (await limiter.allow(SECRET, now=FIXED_NOW)).allowed
    assert not (await limiter.allow(SECRET, now=FIXED_NOW)).allowed

    assert (await limiter.allow(SECRET, now=FIXED_NOW + WINDOW_SECONDS)).allowed


async def test_the_server_expires_the_counter_itself(
    limiters: LimiterFactory, client: FakeAsyncRedis
) -> None:
    """An immortal counter at its ceiling refuses a key forever; the EXPIRE must land."""
    limiter = limiters(1, 0.5)
    await limiter.allow(SECRET, now=FIXED_NOW)
    key = limiter._key(SECRET, FIXED_NOW)

    assert 0 < await client.ttl(key) <= 1

    await asyncio.sleep(1.2)

    assert await client.exists(key) == 0


async def test_two_limiters_sharing_a_backend_share_one_ceiling(
    limiters: LimiterFactory,
) -> None:
    """The entire reason the backend exists: N replicas grant the limit, not N times it."""
    first = limiters(4)
    second = limiters(4)
    assert first._client is not second._client

    for limiter in (first, second, first, second):
        assert (await limiter.allow(SECRET, now=FIXED_NOW)).allowed

    assert not (await second.allow(SECRET, now=FIXED_NOW)).allowed
    assert not (await first.allow(SECRET, now=FIXED_NOW)).allowed


async def test_two_limiters_still_count_two_callers_apart(limiters: LimiterFactory) -> None:
    """A shared ceiling must be shared per identity, not collapsed into one global one."""
    first = limiters(1)
    second = limiters(1)

    assert (await first.allow(SECRET, now=FIXED_NOW)).allowed

    assert (await second.allow(OTHER_SECRET, now=FIXED_NOW)).allowed
    assert not (await second.allow(SECRET, now=FIXED_NOW)).allowed


async def test_the_key_carries_a_fingerprint_and_never_the_api_key(
    limiters: LimiterFactory, client: FakeAsyncRedis
) -> None:
    """Read back off the server, because the credential disclosure would be server-side."""
    limiter = limiters(5)
    await limiter.allow(SECRET, now=FIXED_NOW)

    stored = [_text(key) for key in await client.keys("*")]

    fingerprint = hashlib.sha256(SECRET.encode("utf-8")).hexdigest()[:32]
    bucket = int(FIXED_NOW // WINDOW_SECONDS)
    assert stored == [f"{RATE_LIMIT_NAMESPACE}:{fingerprint}:{bucket}"]
    assert SECRET not in stored[0]
