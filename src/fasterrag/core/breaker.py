"""Per-provider circuit breakers (``reliability.circuit_breaker``, docs/reliability.md §3).

A retry policy answers "should I try this again". A breaker answers the different question
"should I try this at all right now" — and without one, a provider that is down turns every
request into the full retry budget spent waiting on timeouts before failing anyway. The
breaker makes that failure immediate, which is both faster for the caller and gentler on a
provider that is already struggling.

Three states, per the specification:

* **closed** — traffic flows; consecutive failures are counted.
* **open** — traffic is refused immediately, until ``reset_timeout_ms`` has passed.
* **half-open** — one probe is allowed through. It closes the breaker if it succeeds and
  re-opens it if it does not, so recovery costs one request rather than a flood.

**Only retryable failures count.** A 401 is a configuration error that will fail identically
forever, and opening a breaker on it would turn a permanent misconfiguration into an
intermittent one that appears to heal every ``reset_timeout_ms``. A breaker exists to protect
against a provider that is *temporarily* unwell.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger

__all__ = ["CircuitBreaker", "CircuitOpenError", "CircuitState", "create_breakers"]

_MILLISECONDS: Final = 1000.0

_logger = get_logger(__name__)


class CircuitState(IntEnum):
    """The three breaker states, valued to match the ``fasterrag_circuit_state`` gauge."""

    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitOpenError(FasterRagError):
    """Raised instead of calling a provider whose breaker is open."""

    def __init__(self, provider: str, retry_after_seconds: float) -> None:
        """Name the provider and when it is worth trying again.

        Retryable by definition: the breaker's whole premise is that the provider is
        expected to recover, and a caller that gives up permanently on an open circuit
        would turn a transient outage into a lost job.
        """
        super().__init__(
            f"the circuit breaker for {provider!r} is open after repeated failures; "
            f"it will probe again in {retry_after_seconds:.1f}s",
            code=ErrorCode.CIRCUIT_OPEN,
            retryable=True,
        )
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds


@dataclass
class CircuitBreaker:
    """Tracks one provider's health and refuses traffic while it is open."""

    provider: str
    failure_threshold: int = 5
    reset_timeout_ms: int = 30_000
    enabled: bool = True

    def __post_init__(self) -> None:
        """Start closed, and publish that so the gauge is never absent."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._publish()

    @property
    def state(self) -> CircuitState:
        """Return the current state, opening into half-open once the timeout has passed."""
        if self._state is CircuitState.OPEN and self._elapsed_ms() >= self.reset_timeout_ms:
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    @property
    def is_open(self) -> bool:
        """Return whether the breaker is currently shedding traffic."""
        return self.enabled and self.state is CircuitState.OPEN

    def _elapsed_ms(self) -> float:
        """Return how long the breaker has been open, in milliseconds."""
        return (time.monotonic() - self._opened_at) * _MILLISECONDS

    def _publish(self) -> None:
        """Mirror the state onto the gauge a dashboard reads."""
        metrics.CIRCUIT_STATE.set(float(self._state), provider=self.provider)

    def _transition(self, state: CircuitState) -> None:
        """Move to a new state, logging and publishing the change."""
        if state is self._state:
            return

        previous = self._state
        self._state = state
        if state is CircuitState.OPEN:
            self._opened_at = time.monotonic()
        self._publish()

        _logger.warning(
            "circuit breaker changed state",
            extra={
                "provider": self.provider,
                "from": previous.name.lower(),
                "to": state.name.lower(),
                "failures": self._failures,
            },
        )

    def check(self) -> None:
        """Refuse the call when the breaker is open.

        Raises:
            CircuitOpenError: If the provider is currently being shed.
        """
        if not self.enabled:
            return

        if self.state is CircuitState.OPEN:
            remaining = max(self.reset_timeout_ms - self._elapsed_ms(), 0.0) / _MILLISECONDS
            raise CircuitOpenError(self.provider, remaining)

    def record_success(self) -> None:
        """Note a call that worked, closing the breaker and clearing the failure count."""
        self._failures = 0
        self._transition(CircuitState.CLOSED)

    def record_failure(self, error: FasterRagError | None = None) -> None:
        """Note a call that failed, opening the breaker once the threshold is reached.

        # CRITICAL: a non-retryable failure is ignored entirely. A bad API key fails the
        # same way forever, so counting it would open the breaker, let it half-open after
        # the timeout, fail again, and re-open — presenting a permanent misconfiguration as
        # an intermittent outage and hiding the one error message that explains it.
        """
        if not self.enabled:
            return

        if error is not None and not error.retryable:
            return

        # CRITICAL: `self.state`, not `self._state`. The half-open transition is lazy — it
        # happens when the state is *read* — so consulting the raw field here would miss a
        # breaker whose timeout has expired but which nothing has looked at yet. It would
        # then fall through to the counter below, leave `_opened_at` stale, and half-open
        # again on the very next call: one probe admitted per call instead of per timeout.
        current = self.state

        # A failed probe re-opens immediately: half-open exists to test recovery with one
        # request, not to grant a fresh threshold's worth of doomed calls.
        if current is CircuitState.HALF_OPEN:
            self._opened_at = time.monotonic()
            self._transition(CircuitState.OPEN)
            return

        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._transition(CircuitState.OPEN)

    def reset(self) -> None:
        """Force the breaker closed. For tests and for an operator-driven recovery."""
        self._failures = 0
        self._opened_at = 0.0
        self._transition(CircuitState.CLOSED)


def create_breakers(settings: Settings) -> dict[str, CircuitBreaker]:
    """Return one breaker per provider fasterRag calls out to.

    Built eagerly for all three rather than on first use, so ``fasterrag_circuit_state``
    reports ``closed`` from startup. A gauge that springs into existence on the first
    failure is absent exactly when someone goes looking for it, and absent reads the same
    as healthy on every dashboard.
    """
    breaker = settings.reliability.circuit_breaker
    return {
        provider: CircuitBreaker(
            provider=provider,
            failure_threshold=breaker.failure_threshold,
            reset_timeout_ms=breaker.reset_timeout_ms,
            enabled=breaker.enabled,
        )
        for provider in ("llm", "embeddings", "vector_db")
    }
