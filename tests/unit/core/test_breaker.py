"""Per-provider circuit breakers.

Time is advanced by moving the breaker's own clock rather than by sleeping, so the
half-open transition is asserted deterministically instead of on a timer the test hopes is
long enough.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core import breaker as breaker_module
from fasterrag.core.breaker import (
    Admission,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    reset_shared_breakers,
    shared_breaker,
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


def fail(subject: CircuitBreaker, error: EmbedError | None = None) -> None:
    """Admit one call and report that it failed, as every caller does."""
    subject.record_failure(subject.check(), error if error is not None else transient())


def succeed(subject: CircuitBreaker) -> None:
    """Admit one call and report that it worked."""
    subject.record_success(subject.check())


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
        fail(subject)

    assert subject.state is CircuitState.OPEN


def test_it_stays_closed_below_the_threshold() -> None:
    subject = breaker(failure_threshold=3)

    for _ in range(2):
        fail(subject)

    assert subject.state is CircuitState.CLOSED


def test_an_open_breaker_refuses_immediately() -> None:
    """The point of the feature: fail now rather than after the whole retry budget."""
    subject = breaker(failure_threshold=1)
    fail(subject)

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert caught.value.code is ErrorCode.CIRCUIT_OPEN


def test_the_refusal_names_the_provider_and_when_to_retry() -> None:
    subject = breaker(failure_threshold=1, reset_timeout_ms=5000)
    fail(subject)

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert "embeddings" in caught.value.detail
    assert caught.value.retryable is True


def test_a_success_resets_the_failure_count() -> None:
    """Consecutive failures, not cumulative: an occasional blip must not accumulate."""
    subject = breaker(failure_threshold=3)

    fail(subject)
    fail(subject)
    succeed(subject)
    fail(subject)

    assert subject.state is CircuitState.CLOSED


def test_a_permanent_failure_never_opens_it() -> None:
    """A bad key fails the same way forever; opening would present it as intermittent."""
    subject = breaker(failure_threshold=2)

    for _ in range(10):
        fail(subject, permanent())

    assert subject.state is CircuitState.CLOSED


def test_it_half_opens_after_the_timeout() -> None:
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    age(subject, 1500)

    assert subject.state is CircuitState.HALF_OPEN


def test_it_stays_open_before_the_timeout() -> None:
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    age(subject, 500)

    assert subject.state is CircuitState.OPEN


def test_a_half_open_breaker_admits_one_probe_and_refuses_the_rest() -> None:
    """Half-open admitted every caller once the timeout passed; recovery must cost one call.

    Rewritten for TASK-0450: this test used to assert only that the first ``check`` passed,
    which a breaker admitting every caller satisfied too.
    """
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    age(subject, 1500)

    assert isinstance(subject.check(), Admission)
    with pytest.raises(CircuitOpenError) as refused:
        subject.check()

    assert refused.value.code is ErrorCode.CIRCUIT_OPEN
    assert subject.state is CircuitState.HALF_OPEN


def test_a_successful_probe_closes_it() -> None:
    """The probe's own admission closes the circuit (TASK-0450: success now takes a token)."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    age(subject, 1500)
    assert subject.state.name == "HALF_OPEN"

    probe = subject.check()
    subject.record_success(probe)

    assert subject.state.name == "CLOSED"
    subject.check()
    subject.check()


def test_a_late_success_from_before_the_outage_does_not_close_it() -> None:
    """A call admitted while closed and finishing after the circuit opened closed it anyway."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    early = subject.check()
    fail(subject)

    subject.record_success(early)
    assert subject.state is CircuitState.OPEN

    age(subject, 1500)
    subject.record_success(early)
    assert subject.state.name == "HALF_OPEN"


def test_a_late_failure_from_before_the_outage_does_not_decide_the_probe() -> None:
    """Only the probe's outcome may re-open a half-open circuit, not a straggler's."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    early = subject.check()
    fail(subject)
    age(subject, 1500)
    probe = subject.check()

    subject.record_failure(early, transient())
    assert subject.state is CircuitState.HALF_OPEN

    subject.record_success(probe)
    assert subject.state.name == "CLOSED"


def test_a_refused_call_records_nothing() -> None:
    """A caller ``check`` refused holds no admission, so its error cannot re-open the probe."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    age(subject, 1500)
    probe = subject.check()

    with pytest.raises(CircuitOpenError) as refused:
        subject.check()
    subject.record_failure(None, refused.value)
    subject.record_success(None)

    assert subject.state is CircuitState.HALF_OPEN
    subject.record_success(probe)
    assert subject.state.name == "CLOSED"


def test_a_probe_that_never_reports_back_does_not_wedge_the_circuit(clock: Clock) -> None:
    """A cancelled probe never settles; its slot lapses after the reset timeout."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    clock.advance(1000)
    subject.check()

    clock.advance(999)
    with pytest.raises(CircuitOpenError) as refused:
        subject.check()
    assert refused.value.retry_after_seconds == pytest.approx(0.001)

    clock.advance(1)
    replacement = subject.check()
    subject.record_success(replacement)

    assert subject.state is CircuitState.CLOSED


async def test_n_waiters_on_an_open_circuit_admit_exactly_one_probe(clock: Clock) -> None:
    """When the timeout expired every waiting caller was let through at once (TASK-0450)."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    released = asyncio.Event()
    calls: list[int] = []

    async def caller(number: int) -> str:
        await released.wait()
        try:
            admission = subject.check()
        except CircuitOpenError:
            return "refused"
        calls.append(number)
        await asyncio.sleep(0)
        subject.record_success(admission)
        return "admitted"

    waiters = [asyncio.create_task(caller(number)) for number in range(50)]
    await asyncio.sleep(0)
    clock.advance(1000)
    released.set()
    outcomes = await asyncio.gather(*waiters)

    assert outcomes.count("admitted") == 1
    assert outcomes.count("refused") == 49
    assert len(calls) == 1
    assert subject.state is CircuitState.CLOSED


def test_threads_racing_into_half_open_admit_exactly_one_probe() -> None:
    """The breaker is shared per process, so the probe slot must hold across threads too."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    age(subject, 1500)
    start = threading.Barrier(16)

    def race(_: int) -> bool:
        start.wait()
        try:
            subject.check()
        except CircuitOpenError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        admitted = list(pool.map(race, range(16)))

    assert admitted.count(True) == 1


def test_a_failed_probe_reopens_immediately() -> None:
    """Half-open tests recovery with one request, not a fresh threshold of doomed calls."""
    subject = breaker(failure_threshold=3, reset_timeout_ms=1000)
    for _ in range(3):
        fail(subject)
    age(subject, 1500)
    assert subject.state.name == "HALF_OPEN"

    fail(subject)

    assert subject.state.name == "OPEN"


def test_a_failed_probe_restarts_the_timeout() -> None:
    """Otherwise the breaker would half-open again on the next call, forever."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)
    age(subject, 1500)
    fail(subject)

    assert subject.state is CircuitState.OPEN


def test_a_disabled_breaker_never_opens() -> None:
    subject = breaker(failure_threshold=1, enabled=False)

    for _ in range(10):
        fail(subject)

    subject.check()
    assert subject.state is CircuitState.CLOSED


def test_reset_forces_it_closed() -> None:
    subject = breaker(failure_threshold=1)
    fail(subject)

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

    fail(subject)

    assert ('{provider="opening_probe"}', "2.0") in metrics.REGISTRY.series(
        "fasterrag_circuit_state"
    )


def test_closing_publishes_the_state() -> None:
    subject = CircuitBreaker(provider="closing_probe", failure_threshold=1)
    fail(subject)
    age(subject, 30_000)

    succeed(subject)

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
        fail(subject)
    assert subject.state.name == "CLOSED"

    fail(subject)

    assert subject.state is CircuitState.OPEN


def test_the_shipped_default_probes_again_after_thirty_seconds(clock: Clock) -> None:
    """Thirty seconds is the wait an unconfigured provider outage actually costs."""
    subject = CircuitBreaker(provider="default_timeout")
    for _ in range(5):
        fail(subject)

    clock.advance(29_999)
    assert subject.state.name == "OPEN"

    clock.advance(1)

    assert subject.state is CircuitState.HALF_OPEN


def test_it_half_opens_exactly_on_the_timeout(clock: Clock) -> None:
    """The timeout is the first instant a probe is allowed, not the last one refused."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)

    clock.advance(1000)

    assert subject.state is CircuitState.HALF_OPEN


def test_the_refusal_counts_down_the_remaining_wait(clock: Clock) -> None:
    """A caller that sleeps on this number must land on the probe, not before it."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=5000)
    fail(subject)
    clock.advance(2000)

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert caught.value.retry_after_seconds == pytest.approx(3.0)
    clock.advance(caught.value.retry_after_seconds * 1000.0)
    assert subject.state is CircuitState.HALF_OPEN


def test_the_countdown_is_not_rounded_up_in_the_last_millisecond(clock: Clock) -> None:
    """A floor on the remaining wait would tell a caller to sleep longer than the breaker needs."""
    subject = breaker(failure_threshold=1, reset_timeout_ms=5000)
    fail(subject)
    clock.advance(4999.5)

    with pytest.raises(CircuitOpenError) as caught:
        subject.check()

    assert caught.value.retry_after_seconds == pytest.approx(0.0005)


def test_a_success_clears_the_count_rather_than_decrementing_it() -> None:
    """A breaker that only forgets one failure per success opens on a healthy provider."""
    subject = CircuitBreaker(provider="cleared_count")

    for _ in range(4):
        fail(subject)
    succeed(subject)
    for _ in range(4):
        fail(subject)

    assert subject.state is CircuitState.CLOSED


def test_reset_clears_the_failure_count_too() -> None:
    """An operator-forced recovery that kept the count would re-open on the next blip."""
    subject = breaker(failure_threshold=3)
    for _ in range(3):
        fail(subject)

    subject.reset()
    fail(subject)
    fail(subject)

    assert subject.state is CircuitState.CLOSED


def test_is_open_tracks_whether_traffic_is_being_shed() -> None:
    """The embedding pool reads this property to stop retrying; it has to mean what it says."""
    subject = breaker(failure_threshold=1)
    assert subject.is_open is False

    fail(subject)

    assert subject.is_open is True


def test_disabling_an_open_breaker_stops_it_shedding() -> None:
    """The flag is the operator's off switch, so it has to work on a breaker already open."""
    subject = breaker(failure_threshold=1)
    fail(subject)
    assert subject.state is CircuitState.OPEN

    subject.enabled = False

    assert subject.is_open is False
    subject.check()


def test_a_half_open_breaker_is_not_shedding(clock: Clock) -> None:
    """Half-open admits the probe, so reporting it as open would stop the one call that recovers.

    Kept for TASK-0450, with the probe now taken: ``is_open`` still means "open", which the
    embedding pool reads to stop retrying, while ``check`` refuses the callers behind the probe.
    """
    subject = breaker(failure_threshold=1, reset_timeout_ms=1000)
    fail(subject)

    clock.advance(1000)

    assert subject.is_open is False
    subject.check()
    assert subject.is_open is False


def test_a_shared_breaker_is_one_instance_per_provider() -> None:
    configured = Settings.model_validate({})

    assert shared_breaker("llm", configured) is shared_breaker("llm", configured)
    assert shared_breaker("llm", configured) is not shared_breaker("vector_db", configured)


def test_failures_accumulate_across_callers_of_a_shared_breaker() -> None:
    """A breaker rebuilt per request counts to one forever; shared, it reaches the threshold."""
    configured = Settings.model_validate(
        {"reliability": {"circuit_breaker": {"failure_threshold": 3}}}
    )

    for _ in range(3):
        fail(shared_breaker("llm", configured))

    assert shared_breaker("llm", configured).is_open


def test_resetting_drops_every_shared_breaker() -> None:
    configured = Settings.model_validate({})
    before = shared_breaker("llm", configured)

    reset_shared_breakers()

    assert shared_breaker("llm", configured) is not before
