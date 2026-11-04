"""The per-key limiter, both backends.

Every Redis test here drives `FakeRedis`, an in-memory double. No Redis server was
available where this was written, so "the shared window behaves this way against a real
server" is asserted nowhere in this file and is not claimed anywhere else (TASK-0290).

What the double is allowed to prove is the part that is ours: which commands are issued,
that they are issued inside one transaction, what the keys are made of, and what happens
when the backend refuses to answer. What it cannot prove is that redis-py and a real server
agree with the double about any of it.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.rate_limit import (
    RATE_LIMIT_NAMESPACE,
    InProcessRateLimiter,
    RateLimitDecision,
    RedisRateLimiter,
    create_rate_limiter,
)
from fasterrag.errors import ConfigError

SECRET = "an-api-key-that-must-not-be-stored"


class FakeRedis:
    """An in-memory stand-in recording every transaction a limiter sends it."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.expiries: dict[str, int] = {}
        self.transactions: list[list[tuple[Any, ...]]] = []
        self.pipelines: list[FakePipeline] = []
        self.error: Exception | None = None
        self.closed = False

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        opened = FakePipeline(self, transaction)
        self.pipelines.append(opened)
        return opened

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
    assert decision.retry_after == 0, "a request served during an outage was told to back off"
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


@pytest.mark.parametrize("limit", [1, 2, 5])
async def test_the_in_process_ceiling_admits_exactly_the_configured_number(limit: int) -> None:
    """N-1 and N pass, N+1 does not: otherwise nothing tells a limit of N from one of N+1."""
    limiter = InProcessRateLimiter(limit=limit)
    moments = [1000.0 + index / 100 for index in range(limit + 1)]

    admitted = [(await limiter.allow("k", now=moment)).allowed for moment in moments]

    assert admitted == [True] * limit + [False]


@pytest.mark.parametrize("limit", [1, 2, 5])
async def test_the_shared_ceiling_admits_exactly_the_configured_number(limit: int) -> None:
    """The shared window must land on the same ceiling as the in-process one."""
    limiter = shared(FakeRedis(), limit=limit)
    moments = [1000.0 + index / 100 for index in range(limit + 1)]

    admitted = [(await limiter.allow("k", now=moment)).allowed for moment in moments]

    assert admitted == [True] * limit + [False]


async def test_a_request_exactly_one_window_old_has_already_left_the_window() -> None:
    """The boundary is inclusive, and the clock it is measured on does not start at zero."""
    limiter = InProcessRateLimiter(limit=1)
    await limiter.allow("k", now=1000.0)

    assert not (await limiter.allow("k", now=1059.999)).allowed
    assert (await limiter.allow("k", now=1060.0)).allowed


async def test_the_in_process_retry_after_counts_to_the_oldest_request_ageing_out() -> None:
    """Wrong arithmetic here is a wrong `Retry-After` on every 429 the deployment serves."""
    limiter = InProcessRateLimiter(limit=1)
    await limiter.allow("k", now=1000.0)

    assert (await limiter.allow("k", now=1010.0)).retry_after == 51
    assert (await limiter.allow("k", now=1059.5)).retry_after == 1


async def test_the_in_process_wait_is_long_enough_to_be_admitted() -> None:
    """A `Retry-After` a second short is a refused caller in a retry loop."""
    limiter = InProcessRateLimiter(limit=1)
    await limiter.allow("k", now=1000.0)
    refused = await limiter.allow("k", now=1010.0)

    assert (await limiter.allow("k", now=1010.0 + refused.retry_after)).allowed


async def test_the_shared_retry_after_counts_to_the_end_of_the_fixed_window() -> None:
    """The fixed window ends on a wall-clock bucket boundary, not a window after this call."""
    limiter = shared(FakeRedis(), limit=1)
    await limiter.allow("k", now=1000.0)

    assert (await limiter.allow("k", now=1010.0)).retry_after == 11
    assert (await limiter.allow("k", now=1019.5)).retry_after == 1


async def test_the_shared_wait_lands_in_the_next_window() -> None:
    """The advertised wait has to cross the bucket boundary, or the retry is refused too."""
    limiter = shared(FakeRedis(), limit=1)
    await limiter.allow("k", now=1000.0)
    refused = await limiter.allow("k", now=1010.0)

    assert (await limiter.allow("k", now=1010.0 + refused.retry_after)).allowed


async def test_neither_limiter_tells_a_permitted_caller_to_wait() -> None:
    """A `Retry-After` on a served request backs a healthy caller off for nothing."""
    in_process = await InProcessRateLimiter(limit=1).allow("k", now=1000.0)
    in_redis = await shared(FakeRedis(), limit=1).allow("k", now=1000.0)

    assert (in_process.allowed, in_process.retry_after) == (True, 0)
    assert (in_redis.allowed, in_redis.retry_after) == (True, 0)


def test_a_decision_is_a_frozen_value_that_defaults_to_no_wait() -> None:
    """One is allocated per authenticated request and handed to middleware that may not edit."""
    decision = RateLimitDecision(allowed=True)
    field = "retry_after"

    assert decision.retry_after == 0
    assert not hasattr(decision, "__dict__"), "the per-request decision grew an instance dict"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(decision, field, 5)


async def test_the_fingerprint_in_the_key_is_a_128_bit_prefix_of_the_digest() -> None:
    """The truncation length is a security property: shortening it is a collision budget."""
    server = FakeRedis()

    await shared(server).allow(SECRET, now=0.0)

    (key,) = server.counters
    namespace, fingerprint, bucket = key.rsplit(":", 2)
    assert namespace == RATE_LIMIT_NAMESPACE
    assert bucket == "0"
    assert len(fingerprint) == 32
    assert hashlib.sha256(SECRET.encode("utf-8")).hexdigest().startswith(fingerprint)


async def test_the_pair_is_sent_as_a_transaction_rather_than_a_bare_pipeline() -> None:
    """Pipelined without MULTI, a dying process can still land the INCR without the EXPIRE."""
    server = FakeRedis()

    await shared(server).allow("k", now=0.0)

    assert [pipeline.transaction for pipeline in server.pipelines] == [True]


def _without_the_redis_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``import redis`` inside the limiter fail the way a bare install does."""
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("redis"):
            raise ImportError("no module named redis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


def test_selecting_the_shared_window_without_the_client_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one error an operator sees when the optional dependency is missing."""
    _without_the_redis_extra(monkeypatch)

    with pytest.raises(ConfigError, match=r"fasterrag\[redis\]"):
        RedisRateLimiter("redis://unused", limit=1)


async def test_an_injected_client_is_honoured_even_without_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """How this very file would run on a machine that never installed redis-py."""
    server = FakeRedis()
    _without_the_redis_extra(monkeypatch)

    assert (await shared(server, limit=1).allow("k", now=0.0)).allowed
    assert server.counters
