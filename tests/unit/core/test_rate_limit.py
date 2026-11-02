"""The per-key limiter, both backends.

Every Redis test here drives `FakeRedis`, an in-memory double. No Redis server was
available where this was written, so "the shared window behaves this way against a real
server" is asserted nowhere in this file and is not claimed anywhere else (TASK-0216).

What the double is allowed to prove is the part that is ours: which commands are issued,
that they are issued inside one transaction, what the keys are made of, and what happens
when the backend refuses to answer. What it cannot prove is that redis-py and a real server
agree with the double about any of it.
"""

from __future__ import annotations

from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.rate_limit import (
    InProcessRateLimiter,
    RedisRateLimiter,
    create_rate_limiter,
)

SECRET = "an-api-key-that-must-not-be-stored"


class FakeRedis:
    """An in-memory stand-in recording every transaction a limiter sends it."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.expiries: dict[str, int] = {}
        self.transactions: list[list[tuple[Any, ...]]] = []
        self.error: Exception | None = None
        self.closed = False

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        return FakePipeline(self, transaction)

    async def aclose(self) -> None:
        self.closed = True


class FakePipeline:
    """Buffers commands and applies them all at once, the way MULTI/EXEC does."""

    def __init__(self, server: FakeRedis, transaction: bool) -> None:
        self.server = server
        self.transaction = transaction
        self.queued: list[tuple[Any, ...]] = []

    def incr(self, key: str) -> FakePipeline:
        self.queued.append(("incr", key))
        return self

    def expire(self, key: str, seconds: int) -> FakePipeline:
        self.queued.append(("expire", key, seconds))
        return self

    async def execute(self) -> list[Any]:
        if self.server.error is not None:
            raise self.server.error

        results: list[Any] = []
        for command in self.queued:
            if command[0] == "incr":
                self.server.counters[command[1]] = self.server.counters.get(command[1], 0) + 1
                results.append(self.server.counters[command[1]])
            else:
                self.server.expiries[command[1]] = int(command[2])
                results.append(True)

        self.server.transactions.append(self.queued)
        return results


def shared(server: FakeRedis, limit: int = 2) -> RedisRateLimiter:
    return RedisRateLimiter("redis://unused", limit=limit, client=server)


async def test_the_in_process_window_refuses_past_the_ceiling() -> None:
    limiter = InProcessRateLimiter(limit=2)

    assert (await limiter.allow("k", now=0.0)).allowed
    assert (await limiter.allow("k", now=0.1)).allowed
    decision = await limiter.allow("k", now=0.2)

    assert not decision.allowed
    assert decision.retry_after >= 1


async def test_the_in_process_window_rolls_forward() -> None:
    limiter = InProcessRateLimiter(limit=1)
    await limiter.allow("k", now=0.0)

    assert not (await limiter.allow("k", now=30.0)).allowed
    assert (await limiter.allow("k", now=61.0)).allowed


async def test_the_in_process_limit_is_per_key_not_global() -> None:
    """One noisy caller must not exhaust everyone else's budget."""
    limiter = InProcessRateLimiter(limit=1)
    await limiter.allow("first", now=0.0)

    assert (await limiter.allow("second", now=0.0)).allowed


async def test_the_in_process_window_is_per_process() -> None:
    """The defect TASK-0216 is about, asserted rather than described.

    Two limiters stand in for two Gunicorn workers. Each grants the full limit, so the
    deployment grants limit x workers. This stays true on purpose; `redis` is the fix.
    """
    first = InProcessRateLimiter(limit=1)
    second = InProcessRateLimiter(limit=1)
    await first.allow("k", now=0.0)

    assert not (await first.allow("k", now=0.0)).allowed
    assert (await second.allow("k", now=0.0)).allowed


async def test_the_shared_window_refuses_past_the_ceiling() -> None:
    limiter = shared(FakeRedis(), limit=2)

    assert (await limiter.allow("k", now=0.0)).allowed
    assert (await limiter.allow("k", now=0.1)).allowed
    decision = await limiter.allow("k", now=0.2)

    assert not decision.allowed
    assert decision.retry_after >= 1


async def test_every_replica_counts_into_one_shared_window() -> None:
    """The point of the whole change: N replicas must grant the limit once, not N times."""
    server = FakeRedis()
    first = shared(server, limit=1)
    second = shared(server, limit=1)

    assert (await first.allow("k", now=0.0)).allowed
    assert not (await second.allow("k", now=0.0)).allowed


async def test_the_shared_limit_is_per_key_not_global() -> None:
    server = FakeRedis()
    limiter = shared(server, limit=1)
    await limiter.allow("first", now=0.0)

    assert (await limiter.allow("second", now=0.0)).allowed


async def test_a_new_window_is_a_new_counter() -> None:
    """A window ends because its key stops being addressed, not because anything resets it."""
    server = FakeRedis()
    limiter = shared(server, limit=1)
    await limiter.allow("k", now=0.0)

    assert not (await limiter.allow("k", now=59.0)).allowed
    assert (await limiter.allow("k", now=60.0)).allowed
    assert len(server.counters) == 2


async def test_the_counter_and_its_expiry_travel_in_one_transaction() -> None:
    """Issued separately, a crash between them leaves an immortal key stuck at its ceiling."""
    server = FakeRedis()

    await shared(server).allow("k", now=0.0)

    assert len(server.transactions) == 1
    issued = server.transactions[0]
    assert [command[0] for command in issued] == ["incr", "expire"]
    assert issued[0][1] == issued[1][1]


async def test_every_counter_is_given_an_expiry() -> None:
    """A counter with no expiry is a leak per key per minute, forever."""
    server = FakeRedis()

    await shared(server).allow("k", now=0.0)

    assert set(server.expiries) == set(server.counters)
    assert all(seconds > 60 for seconds in server.expiries.values())


async def test_the_api_key_is_never_written_to_the_backend() -> None:
    """Redis is shared and often co-tenanted; a key stored there is a disclosure."""
    server = FakeRedis()

    await shared(server).allow(SECRET, now=0.0)

    assert server.counters
    assert all(SECRET not in key for key in server.counters)


async def test_a_backend_outage_serves_the_request_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fail open, loudly: the reasoning is in docs/security.md section 2."""
    server = FakeRedis()
    server.error = OSError("connection refused")
    limiter = shared(server, limit=1)

    with caplog.at_level("WARNING", logger="fasterrag.core.rate_limit"):
        decision = await limiter.allow("k", now=0.0)

    assert decision.allowed
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert getattr(caplog.records[0], "trace_id", None)


async def test_closing_releases_the_connection_pool() -> None:
    server = FakeRedis()

    await shared(server).close()

    assert server.closed


async def test_a_failure_while_closing_is_reported_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Shutdown must not be the thing that turns a sick backend into a crash."""
    server = FakeRedis()

    async def refuse() -> None:
        raise OSError("already gone")

    server.aclose = refuse  # type: ignore[method-assign]

    with caplog.at_level("WARNING", logger="fasterrag.core.rate_limit"):
        await shared(server).close()

    assert [record.levelname for record in caplog.records] == ["WARNING"]


def test_the_factory_defaults_to_the_in_process_window() -> None:
    limiter = create_rate_limiter(Settings.model_validate({}))

    assert isinstance(limiter, InProcessRateLimiter)
    assert limiter.limit == 600


def test_the_factory_builds_the_shared_window_when_asked() -> None:
    settings = Settings.model_validate({"security": {"rate_limit_backend": "redis"}})

    limiter = create_rate_limiter(settings)

    assert isinstance(limiter, RedisRateLimiter)
    assert limiter.limit == settings.security.rate_limit_per_minute


def test_an_unparseable_redis_url_is_refused_at_startup() -> None:
    """A typo must name the key at startup, not surface as a 429 storm hours later."""
    with pytest.raises(ValueError, match="rate_limit_redis_url"):
        Settings.model_validate({"security": {"rate_limit_redis_url": "localhost:6379"}})
