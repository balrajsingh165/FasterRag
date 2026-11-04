"""Per-provider circuit breakers.

Time is advanced by moving the breaker's own clock rather than by sleeping, so the
half-open transition is asserted deterministically instead of on a timer the test hopes is
long enough.
"""

from typing import Any

import pytest

from fasterrag.core import breaker as breaker_module
from fasterrag.core.breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from fasterrag.errors import EmbedError, ErrorCode
from fasterrag.observability import metrics


class Clock:
    """A monotonic clock a test moves by hand, standing in for the module's ``time``.

    Boundary cases land on the boundary exactly, which a sleep on a 15.6 ms Windows
    tick cannot promise (TASK-0262, TASK-0288).
    """

    def __init__(self) -> None:
        """Start on a whole second so millisecond arithmetic stays exact."""
        self._milliseconds = 1_000_000.0

    def monotonic(self) -> float:
        """Return the reading in seconds, as ``time.monotonic`` does."""
        return self._milliseconds / 1000.0

    def advance(self, milliseconds: float) -> None:
        """Move the clock forward by the given number of milliseconds."""
        self._milliseconds += milliseconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Give the breaker a clock that only moves when a test says so."""
    frozen = Clock()
    monkeypatch.setattr(breaker_module, "time", frozen)
    return frozen


def breaker(**overrides: Any) -> CircuitBreaker:
    fields: dict[str, Any] = {
        "provider": "embeddings",
        "failure_threshold": 3,
        "reset_timeout_ms": 1000,
    }
    fields.update(overrides)
    return CircuitBreaker(**fields)


def transient() -> EmbedError:
    return EmbedError("provider is down", retryable=True)


def permanent() -> EmbedError:
    return EmbedError("bad api key", retryable=False)


def age(subject: CircuitBreaker, milliseconds: float) -> None:
    """Pretend the breaker opened ``milliseconds`` ago."""
    subject._opened_at -= milliseconds / 1000.0


def test_it_starts_closed() -> None:
    assert breaker().state is CircuitState.CLOSED


def test_a_closed_breaker_lets_traffic_through() -> None:
    breaker().check()


def test_it_opens_on_the_threshold() -> None:
    subject = breaker(failure_threshold=3)

    for _ in range(3):
        subject.record_failure(transient())

    assert subject.state is CircuitState.OPEN


def test_it_stays_closed_below_the_threshold() -> None:
    subject = breaker(failure_threshold=3)

    for _ in range(2):
        subject.record_failure(transient())

    assert subject.state is CircuitState.CLOSED


def test_an_open_breaker_refuses_immediately() -> None:
    """The point of the feature: fail now rather than after the whole retry budget."""
    subject = breaker(failure_threshold=1)
    subject.record_failure(transient())

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert caught.value.code is ErrorCode.CIRCUIT_OPEN


def test_the_refusal_names_the_provider_and_when_to_retry() -> None:
    subject = breaker(failure_threshold=1, reset_timeout_ms=5000)
    subject.record_failure(transient())

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert "embeddings" in caught.value.detail
    assert caught.value.retryable is True


def test_a_success_resets_the_failure_count() -> None:
    """Consecutive failures, not cumulative: an occasional blip must not accumulate."""
    subject = breaker(failure_threshold=3)

    subject.record_failure(transient())
    subject.record_failure(transient())
    subject.record_success()
    subject.record_failure(transient())

    assert subject.state is CircuitState.CLOSED


def test_a_permanent_failure_never_opens_it() -> None:
    """A bad key fails the same way forever; opening would present it as intermittent."""
    subject = breaker(failure_threshold=2)

    for _ in range(10):
        subject.record_failure(permanent())

    assert subject.state is CircuitState.CLOSED


def test_it_half_opens_after_the_timeout() -> None:
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    subject.record_failure(transient())
    age(subject, 1500)

    assert subject.state is CircuitState.HALF_OPEN


def test_it_stays_open_before_the_timeout() -> None:
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    subject.record_failure(transient())
    age(subject, 500)

    assert subject.state is CircuitState.OPEN


def test_a_half_open_breaker_admits_the_probe() -> None:
    """Recovery has to cost one request; refusing here would never let it close."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    subject.record_failure(transient())
    age(subject, 1500)

    subject.check()


def test_a_successful_probe_closes_it() -> None:
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    subject.record_failure(transient())
    age(subject, 1500)
    assert subject.state.name == "HALF_OPEN"

    subject.record_success()

    assert subject.state.name == "CLOSED"


def test_a_failed_probe_reopens_immediately() -> None:
    """Half-open tests recovery with one request, not a fresh threshold of doomed calls."""
    subject = breaker(failure_threshold=3, reset_timeout_ms=1000)
    for _ in range(3):
        subject.record_failure(transient())
    age(subject, 1500)
    assert subject.state.name == "HALF_OPEN"

    subject.record_failure(transient())

    assert subject.state.name == "OPEN"


def test_a_failed_probe_restarts_the_timeout() -> None:
    """Otherwise the breaker would half-open again on the next call, forever."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    subject.record_failure(transient())
    age(subject, 1500)
    subject.record_failure(transient())

    assert subject.state is CircuitState.OPEN


def test_a_disabled_breaker_never_opens() -> None:
    subject = breaker(failure_threshold=1, enabled=False)

    for _ in range(10):
        subject.record_failure(transient())

    subject.check()
    assert subject.state is CircuitState.CLOSED


def test_reset_forces_it_closed() -> None:
    subject = breaker(failure_threshold=1)
    subject.record_failure(transient())

    subject.reset()

    assert subject.state is CircuitState.CLOSED
    subject.check()


def test_the_state_is_published_from_the_start() -> None:
    """A gauge that appears on first failure is absent exactly when someone looks."""
    CircuitBreaker(provider="published_probe")

    assert ('{provider="published_probe"}', "0.0") in metrics.REGISTRY.series(
        "fasterrag_circuit_state"
    )


def test_opening_publishes_the_state() -> None:
    subject = CircuitBreaker(provider="opening_probe", failure_threshold=1)

    subject.record_failure(transient())

    assert ('{provider="opening_probe"}', "2.0") in metrics.REGISTRY.series(
        "fasterrag_circuit_state"
    )


def test_closing_publishes_the_state() -> None:
    subject = CircuitBreaker(provider="closing_probe", failure_threshold=1)
    subject.record_failure(transient())

    subject.record_success()

    assert ('{provider="closing_probe"}', "0.0") in metrics.REGISTRY.series(
        "fasterrag_circuit_state"
    )


def test_the_gauge_values_match_the_documented_encoding() -> None:
    """The metric is documented as 0 closed / 1 half-open / 2 open."""
    assert (CircuitState.CLOSED, CircuitState.HALF_OPEN, CircuitState.OPEN) == (0, 1, 2)


def test_the_shipped_default_opens_on_the_fifth_failure() -> None:
    """Every other test names a threshold; five is what a user who names none gets."""
    subject = CircuitBreaker(provider="default_threshold")

    for _ in range(4):
        subject.record_failure(transient())
    assert subject.state.name == "CLOSED"

    subject.record_failure(transient())

    assert subject.state is CircuitState.OPEN


def test_the_shipped_default_probes_again_after_thirty_seconds(clock: Clock) -> None:
    """Thirty seconds is the wait an unconfigured provider outage actually costs."""
    subject = CircuitBreaker(provider="default_timeout")
    for _ in range(5):
        subject.record_failure(transient())

    clock.advance(29_999)
    assert subject.state.name == "OPEN"

    clock.advance(1)

    assert subject.state is CircuitState.HALF_OPEN


def test_it_half_opens_exactly_on_the_timeout(clock: Clock) -> None:
    """The timeout is the first instant a probe is allowed, not the last one refused."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    subject.record_failure(transient())

    clock.advance(1000)

    assert subject.state is CircuitState.HALF_OPEN


def test_the_refusal_counts_down_the_remaining_wait(clock: Clock) -> None:
    """A caller that sleeps on this number must land on the probe, not before it."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=5000)
    subject.record_failure(transient())
    clock.advance(2000)

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert caught.value.retry_after_seconds == pytest.approx(3.0)
    clock.advance(caught.value.retry_after_seconds * 1000.0)
    assert subject.state is CircuitState.HALF_OPEN


def test_the_countdown_is_not_rounded_up_in_the_last_millisecond(clock: Clock) -> None:
    """A floor on the remaining wait would tell a caller to sleep longer than the breaker needs."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=5000)
    subject.record_failure(transient())
    clock.advance(4999.5)

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert caught.value.retry_after_seconds == pytest.approx(0.0005)


def test_a_success_clears_the_count_rather_than_decrementing_it() -> None:
    """A breaker that only forgets one failure per success opens on a healthy provider."""
    subject = CircuitBreaker(provider="cleared_count")

    for _ in range(4):
        subject.record_failure(transient())
    subject.record_success()
    for _ in range(4):
        subject.record_failure(transient())

    assert subject.state is CircuitState.CLOSED


def test_reset_clears_the_failure_count_too() -> None:
    """An operator-forced recovery that kept the count would re-open on the next blip."""
    subject = breaker(failure_threshold=3)
    for _ in range(3):
        subject.record_failure(transient())

    subject.reset()
    subject.record_failure(transient())
    subject.record_failure(transient())

    assert subject.state is CircuitState.CLOSED


def test_is_open_tracks_whether_traffic_is_being_shed() -> None:
    """The embedding pool reads this property to stop retrying; it has to mean what it says."""
    subject = breaker(failure_threshold=1)
    assert subject.is_open is False

    subject.record_failure(transient())

    assert subject.is_open is True


def test_disabling_an_open_breaker_stops_it_shedding() -> None:
    """The flag is the operator's off switch, so it has to work on a breaker already open."""
    subject = breaker(failure_threshold=1)
    subject.record_failure(transient())
    assert subject.state is CircuitState.OPEN

    subject.enabled = False

    assert subject.is_open is False
    subject.check()


def test_a_half_open_breaker_is_not_shedding(clock: Clock) -> None:
    """Half-open admits the probe, so reporting it as open would stop the one call that recovers."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    subject.record_failure(transient())

    clock.advance(1000)

    assert subject.is_open is False
