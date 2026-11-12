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

**Admissions are tokens** (TASK-0450). ``check`` returns an :class:`Admission` carrying the
state generation it was issued in, and ``record_success`` / ``record_failure`` take it back.
Every state change starts a new generation, so the outcome of a call admitted before the
circuit opened — a late success above all — cannot close a circuit it never tested. In
half-open exactly one admission is issued; every other caller is refused until it resolves.
A probe whose caller never reports back (cancelled, crashed) is not waited on forever: its
lease lapses after ``reset_timeout_ms`` and the next caller becomes the probe.

**Only retryable failures count.** A 401 is a configuration error that will fail identically
forever, and opening a breaker on it would turn a permanent misconfiguration into an
intermittent one that appears to heal every ``reset_timeout_ms``. A breaker exists to protect
against a provider that is *temporarily* unwell.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger

__all__ = [
    "Admission",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "reset_shared_breakers",
    "shared_breaker",
]

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
        """Name the provider and when it is worth trying again."""
        super().__init__(
            f"the circuit breaker for {provider!r} is open after repeated failures; "
            f"it will probe again in {retry_after_seconds:.1f}s",
            code=ErrorCode.CIRCUIT_OPEN,
            retryable=True,
        )
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class Admission:
    """Permission for one call, valid only in the state generation that issued it."""

    generation: int


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
        self._generation = 0
        self._probe_issued_at: float | None = None
        self._lock = threading.RLock()
        self._publish()

    @property
    def state(self) -> CircuitState:
        """Return the current state, opening into half-open once the timeout has passed."""
        with self._lock:
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
        self._generation += 1
        self._probe_issued_at = None
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

    def check(self) -> Admission:
        """Admit one call and return its token, or refuse it while the provider is shed.

        Raises:
            CircuitOpenError: If the circuit is open, or half-open with its probe outstanding.
        """
        with self._lock:
            if not self.enabled:
                return Admission(self._generation)

            state = self.state
            if state is CircuitState.CLOSED:
                return Admission(self._generation)

            if state is CircuitState.OPEN:
                remaining = max(self.reset_timeout_ms - self._elapsed_ms(), 0.0)
                raise CircuitOpenError(self.provider, remaining / _MILLISECONDS)

            now = time.monotonic()
            if self._probe_issued_at is not None:
                held_ms = (now - self._probe_issued_at) * _MILLISECONDS
                # CRITICAL: the probe's admission is a lease, not a lock. A probe whose caller
                # is cancelled never reports back, and a slot held until it did would refuse
                # every call forever; after `reset_timeout_ms` the next caller probes instead.
                if held_ms < self.reset_timeout_ms:
                    remaining = (self.reset_timeout_ms - held_ms) / _MILLISECONDS
                    raise CircuitOpenError(self.provider, remaining)
            self._probe_issued_at = now
            return Admission(self._generation)

    def record_success(self, admission: Admission | None) -> None:
        """Note a call that worked, closing the breaker if it was admitted in this state."""
        if admission is None:
            return

        with self._lock:
            if admission.generation != self._generation:
                return
            self._failures = 0
            self._transition(CircuitState.CLOSED)

    def record_failure(
        self, admission: Admission | None, error: FasterRagError | None = None
    ) -> None:
        """Note a failed call, opening the breaker once the threshold is reached.

        A call ``check`` refused has no admission; passing ``None`` for it records nothing.
        """
        if not self.enabled or admission is None:
            return

        if error is not None and not error.retryable:
            return

        with self._lock:
            # CRITICAL: `self.state`, not `self._state`. The half-open transition is lazy — it
            # happens when the state is *read* — so consulting the raw field here would miss a
            # breaker whose timeout has expired but which nothing has looked at yet. It would
            # then fall through to the counter below, leave `_opened_at` stale, and half-open
            # again on the very next call: one probe admitted per call instead of per timeout.
            current = self.state
            if admission.generation != self._generation:
                return

            if current is CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
                return

            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._transition(CircuitState.OPEN)

    def reset(self) -> None:
        """Force the breaker closed. For tests and for an operator-driven recovery."""
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0
            self._transition(CircuitState.CLOSED)


_SHARED: Final[dict[tuple[object, ...], CircuitBreaker]] = {}


def shared_breaker(provider: str, settings: Settings) -> CircuitBreaker:
    """Return the process-wide breaker for a provider, creating it on first use."""
    configured = settings.reliability.circuit_breaker
    key = (provider, configured.failure_threshold, configured.reset_timeout_ms, configured.enabled)
    breaker = _SHARED.get(key)
    if breaker is None:
        breaker = CircuitBreaker(
            provider=provider,
            failure_threshold=configured.failure_threshold,
            reset_timeout_ms=configured.reset_timeout_ms,
            enabled=configured.enabled,
        )
        _SHARED[key] = breaker
    return breaker


def reset_shared_breakers() -> None:
    """Drop every shared breaker, so a test or a reload starts from closed circuits."""
    _SHARED.clear()
