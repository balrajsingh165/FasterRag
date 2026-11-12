"""Per-key request limiting, with the window kept wherever the deployment chooses.

One narrow interface — ``allow`` and ``close`` — over the backends
``security.rate_limit_backend`` names, so the limiter is pluggable exactly the way every
other storage boundary in this repository is.

**Why this exists.** The limiter was a window held in one process's dictionary. The
documented production topology is Gunicorn with N Uvicorn workers
(``docs/deployment.md``), and N processes each holding their own window grant
``rate_limit_per_minute`` x N. The configured number silently became a *per-worker*
number, and for a control whose whole job is to cap abuse, that is the direction of error
that permits more than was asked for.

**``memory`` stays the default and stays the algorithm that already shipped**, unchanged.
A single-process deployment is correct today and must not start paying for a network round
trip per request to stay that way.

**``redis`` is a fixed window bucketed by wall-clock time.** The window index is part of
the key, so a new window is a new key rather than a counter something has to reset — no
replica can disagree about when the window began, and nothing has to run a sweep. The two
commands that make a window, ``INCR`` and ``EXPIRE``, are issued inside one
``MULTI``/``EXEC`` transaction. That is not decoration: issued separately, a process dying
between them leaves a counter with no expiry, and an immortal counter sitting at its
ceiling refuses that key forever. Inside a transaction Redis runs the pair or neither.

**Replicas must agree about the clock.** The window index comes from each host's wall
clock, so hosts skewed relative to each other step between buckets at different moments and
a caller may receive part of an extra window's allowance around a boundary. The effect is
bounded by the skew; the alternative — asking Redis for its own ``TIME`` — costs a second
round trip on every authenticated request. Run NTP.

**The identity is hashed before it is written.** The identity handed to this module is the
API key itself. A key written in plain text into a shared, frequently co-tenanted datastore
is a credential disclosure wearing a counter's clothes, so only a SHA-256 fingerprint of it
ever leaves the process.

**A backend outage fails open.** The request is served, the limit is not enforced for it,
and the failure is logged with the request's trace id — never swallowed. The reasoning for
choosing that over failing closed is recorded in ``docs/security.md`` §2, where an operator
deciding how to deploy this will actually read it.

**Now run against a live Redis server, and it agrees with the double.** This module was
written where no Redis was available, so every test covering :class:`RedisRateLimiter`
drove an in-memory double and "this behaves correctly against a real Redis" was an
untested claim. ``tests/integration/test_redis_rate_limit_server.py`` closes it against
``redis:7.4.2-alpine`` over a real socket (2026-11-11): the ceiling, the window roll, the
server-side ``EXPIRE``, the shared ceiling, the per-identity split and the fingerprinted
key all hold, and one case drives the same decisions through both the server and
``FakeAsyncRedis`` and finds **no divergence**. What is still untested, and keeps TASK-0290
open: resets, partitions, failover and pool exhaustion; and replicas sharing one ceiling
across *processes*, which is the topology this backend exists for.

**A hung server fails open as a dead one does** (TASK-0453). The client is bounded by
``reliability.timeouts.redis_ms``, the options shared with ``fasterrag.core.cache.redis``
through ``bounded_client_options``; without them a server that accepted the connection and
never answered held every authenticated request for as long as the installed redis-py
allowed — indefinitely on the 5.x releases ``pyproject.toml`` admits.
``tests/unit/core/test_redis_timeouts.py`` drives the fail-open path over a real socket that
accepts and never replies.

The connection handling deliberately mirrors ``fasterrag.core.cache.redis``: same optional
import, same ``ConfigError`` when the extra is missing, same ``from_url``.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.core.cache.redis import DEFAULT_REDIS_TIMEOUT_MS, bounded_client_options
from fasterrag.errors import ConfigError
from fasterrag.observability.logging import current_trace_id, get_logger

__all__ = [
    "RATE_LIMIT_NAMESPACE",
    "WINDOW_SECONDS",
    "InProcessRateLimiter",
    "RateLimitDecision",
    "RateLimiter",
    "RedisRateLimiter",
    "create_rate_limiter",
]

RATE_LIMIT_NAMESPACE: Final = "fasterrag:ratelimit"

# CRITICAL: the window is one minute because the only setting that sizes it is
# `security.rate_limit_per_minute`. Changing this constant alone would make that key's name
# a lie, and the key is the documented contract.
WINDOW_SECONDS: Final = 60.0

_BACKEND_SETTING: Final = "security.rate_limit_backend"

_logger = get_logger(__name__)


def _install_hint(setting: str) -> str:
    """Return the message shown when the backend is selected without its client."""
    return (
        f"{setting} is 'redis', which needs the redis client; "
        "install it with 'pip install fasterrag[redis]'"
    )


def _fingerprint(identity: str) -> str:
    """Return the name ``identity`` is counted under, which is never ``identity`` itself."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Whether a request may proceed, and the ``Retry-After`` seconds when it may not.

    A refused caller told to back off without being told for how long retries immediately,
    which is the behaviour the limit exists to stop.
    """

    allowed: bool
    retry_after: int = 0


class RateLimiter(ABC):
    """Counts requests per identity and refuses the ones past the configured ceiling."""

    @abstractmethod
    async def allow(self, identity: str) -> RateLimitDecision:
        """Record a request for ``identity`` and report whether it is permitted."""

    @abstractmethod
    async def close(self) -> None:
        """Release whatever this backend holds open."""


class InProcessRateLimiter(RateLimiter):
    """A per-identity window held in this process alone, correct for one replica."""

    def __init__(self, limit: int, window_seconds: float = WINDOW_SECONDS) -> None:
        """Permit ``limit`` requests per identity per window."""
        self.limit = limit
        self.window_seconds = window_seconds
        self._seen: dict[str, deque[float]] = {}

    async def allow(self, identity: str, now: float | None = None) -> RateLimitDecision:
        """Record a request and report whether it is permitted."""
        moment = time.monotonic() if now is None else now
        window = self._seen.setdefault(identity, deque())

        while window and moment - window[0] >= self.window_seconds:
            window.popleft()

        if len(window) >= self.limit:
            retry_after = self.window_seconds - (moment - window[0])
            return RateLimitDecision(False, max(1, int(retry_after) + 1))

        window.append(moment)
        return RateLimitDecision(True, 0)

    async def close(self) -> None:
        """Release nothing: this window is a dictionary that dies with its process."""


class RedisRateLimiter(RateLimiter):
    """A fixed window counted in Redis, so every replica shares one ceiling."""

    def __init__(
        self,
        url: str,
        *,
        limit: int,
        window_seconds: float = WINDOW_SECONDS,
        namespace: str = RATE_LIMIT_NAMESPACE,
        setting: str = _BACKEND_SETTING,
        client: Any | None = None,
        timeout_ms: int = DEFAULT_REDIS_TIMEOUT_MS,
    ) -> None:
        """Count ``limit`` requests per window against the Redis server at ``url``."""
        self.limit = limit
        self.window_seconds = window_seconds
        self._namespace = namespace
        self._expiry = int(window_seconds) + 1
        self._failures: tuple[type[BaseException], ...] = (OSError,)

        try:
            from redis.asyncio import Redis
            from redis.exceptions import RedisError
        except ImportError as exc:
            if client is None:
                raise ConfigError(_install_hint(setting)) from exc
            self._client: Any = client
            return

        self._failures = (RedisError, OSError)
        if client is not None:
            self._client = client
            return

        try:
            self._client = Redis.from_url(url, **bounded_client_options(timeout_ms))
        except ValueError as exc:
            raise ConfigError(
                f"{setting} is 'redis' but security.rate_limit_redis_url could not be parsed: {exc}"
            ) from exc

    def _key(self, identity: str, moment: float) -> str:
        """Return the counter holding ``identity``'s requests for the window at ``moment``."""
        bucket = int(moment // self.window_seconds)
        return f"{self._namespace}:{_fingerprint(identity)}:{bucket}"

    async def allow(self, identity: str, now: float | None = None) -> RateLimitDecision:
        """Count this request in the shared window and report whether it is permitted."""
        moment = time.time() if now is None else now

        try:
            pipeline = self._client.pipeline(transaction=True)
            pipeline.incr(self._key(identity, moment))
            pipeline.expire(self._key(identity, moment), self._expiry)
            results = await pipeline.execute()
        except self._failures as exc:
            return self._degrade(exc)

        if int(results[0]) > self.limit:
            remaining = self.window_seconds - (moment % self.window_seconds)
            return RateLimitDecision(False, max(1, int(remaining) + 1))
        return RateLimitDecision(True, 0)

    def _degrade(self, exc: BaseException) -> RateLimitDecision:
        """Serve the request the shared window could not rule on, and say so loudly."""
        _logger.warning(
            "the shared rate-limit backend is unreachable; this request was served with "
            "no limit enforced",
            extra={"error": type(exc).__name__, "trace_id": current_trace_id()},
        )
        return RateLimitDecision(True, 0)

    async def close(self) -> None:
        """Release the connection pool, reporting rather than raising on a bad goodbye."""
        try:
            await self._client.aclose()
        except self._failures as exc:
            _logger.warning(
                "the shared rate-limit backend did not close cleanly",
                extra={"error": type(exc).__name__, "trace_id": current_trace_id()},
            )


def create_rate_limiter(settings: Settings) -> RateLimiter:
    """Return the limiter ``security.rate_limit_backend`` names."""
    limit = settings.security.rate_limit_per_minute
    if settings.security.rate_limit_backend == "memory":
        return InProcessRateLimiter(limit)
    return RedisRateLimiter(
        settings.security.rate_limit_redis_url,
        limit=limit,
        timeout_ms=settings.reliability.timeouts.redis_ms,
    )
