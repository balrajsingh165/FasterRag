"""A Redis that accepts the connection and never answers (TASK-0453).

A stopped Redis refuses the connection and both clients fail open at once. A *hung* one —
overloaded, paused, or a partition that leaves the TCP handshake intact — accepted the
connection and then held every query and every authenticated request for as long as the
installed redis-py allowed, because neither client set a socket timeout: indefinitely on the
5.x releases ``pyproject.toml`` admits, five seconds per call on the release ``uv.lock`` pins.
These tests stand up exactly that server on the loopback interface: it accepts, reads whatever
it is sent, and never writes a byte.

Each case is wrapped in a hard ceiling far above the bound it asserts, so a regression fails
instead of hanging the suite.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.cache import RedisStore, SemanticCache, create_semantic_store
from fasterrag.core.rate_limit import RedisRateLimiter, create_rate_limiter
from fasterrag.errors import CacheError

pytest.importorskip("redis")

TIMEOUT_MS = 250

BOUND_SECONDS = 2.0
"""Above the configured timeout, below the five seconds the locked redis-py defaults to."""

CEILING_SECONDS = 30.0


@pytest.fixture
async def silent_redis() -> AsyncIterator[str]:
    """Serve a socket that accepts every connection and never replies; yield its URL."""

    async def swallow(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while await reader.read(65536):
            pass
        writer.close()

    server = await asyncio.start_server(swallow, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        server.close()


async def test_a_hung_cache_backend_fails_typed_within_the_timeout(silent_redis: str) -> None:
    """With no socket timeout a cache read on a hung Redis never returned."""
    store = RedisStore(silent_redis, namespace="test:hung", timeout_ms=TIMEOUT_MS)
    started = time.perf_counter()
    try:
        with pytest.raises(CacheError):
            await asyncio.wait_for(store.get("k"), CEILING_SECONDS)
    finally:
        elapsed = time.perf_counter() - started
        await store.close()

    assert elapsed < BOUND_SECONDS


async def test_a_hung_semantic_cache_reads_as_a_miss_within_the_timeout(
    silent_redis: str,
) -> None:
    """The configured key reaches the client: the query path sees a miss, not a hang."""
    settings = Settings.model_validate(
        {
            "cache": {"semantic": True, "backend": "redis", "redis_url": silent_redis},
            "reliability": {"timeouts": {"redis_ms": TIMEOUT_MS}},
        }
    )
    store = create_semantic_store(settings)
    cache = SemanticCache(settings, store)
    started = time.perf_counter()
    try:
        hit = await asyncio.wait_for(cache.lookup([1.0, 0.0, 0.0]), CEILING_SECONDS)
    finally:
        elapsed = time.perf_counter() - started
        await store.close()

    assert hit is None
    assert cache.stats.errors == 1
    assert elapsed < BOUND_SECONDS


async def test_a_hung_rate_limit_backend_fails_open_within_the_timeout(
    silent_redis: str, caplog: pytest.LogCaptureFixture
) -> None:
    """With no socket timeout a hung Redis held every authenticated request."""
    settings = Settings.model_validate(
        {
            "security": {"rate_limit_backend": "redis", "rate_limit_redis_url": silent_redis},
            "reliability": {"timeouts": {"redis_ms": TIMEOUT_MS}},
        }
    )
    limiter = create_rate_limiter(settings)
    assert isinstance(limiter, RedisRateLimiter)
    started = time.perf_counter()
    try:
        with caplog.at_level("WARNING", logger="fasterrag.core.rate_limit"):
            decision = await asyncio.wait_for(limiter.allow("key"), CEILING_SECONDS)
    finally:
        elapsed = time.perf_counter() - started
        await limiter.close()

    assert decision.allowed
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert elapsed < BOUND_SECONDS


async def test_the_configured_timeout_is_the_one_that_applies(silent_redis: str) -> None:
    """Waited out in full, so the bound is the configured one rather than a client default."""
    store = RedisStore(silent_redis, namespace="test:hung", timeout_ms=1000)
    started = time.perf_counter()
    try:
        with pytest.raises(CacheError):
            await asyncio.wait_for(store.get("k"), CEILING_SECONDS)
    finally:
        elapsed = time.perf_counter() - started
        await store.close()

    assert 0.5 <= elapsed < 3.0
