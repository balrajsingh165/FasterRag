"""The job registry: the handle on work that outlives the request that started it.

Background ingestion used to be a Starlette background task nothing referenced, so the
process could not enumerate, await, or stop its own jobs (TASK-0436).
"""

import asyncio

import pytest

from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.jobs import JobRegistry


async def hold(gate: asyncio.Event, ran: list[str], name: str) -> None:
    """Record that a job started, then wait to be released."""
    ran.append(name)
    await gate.wait()


async def test_a_submitted_job_is_listed_while_it_runs() -> None:
    """Nothing could say which jobs the process was running."""
    registry = JobRegistry()
    gate = asyncio.Event()
    ran: list[str] = []

    started = await registry.submit("job_a", hold(gate, ran, "job_a"))
    await asyncio.sleep(0)

    assert started is True
    assert ran == ["job_a"]
    assert registry.active == {"job_a"}

    gate.set()
    await registry.idle()


async def test_a_finished_job_is_forgotten() -> None:
    """A registry that kept finished tasks would grow by one entry per job, forever."""
    registry = JobRegistry()
    gate = asyncio.Event()
    gate.set()

    await registry.submit("job_a", hold(gate, [], "job_a"))
    await registry.idle()

    assert registry.active == frozenset()


async def test_idle_waits_for_every_job_in_flight() -> None:
    registry = JobRegistry()
    gate = asyncio.Event()
    ran: list[str] = []
    await registry.submit("job_a", hold(gate, ran, "job_a"))
    await registry.submit("job_b", hold(gate, ran, "job_b"))

    waiting = asyncio.ensure_future(registry.idle())
    await asyncio.sleep(0.01)
    still_waiting = not waiting.done()
    gate.set()
    await waiting

    assert still_waiting
    assert sorted(ran) == ["job_a", "job_b"]
    assert registry.active == frozenset()


async def test_idle_returns_at_once_when_nothing_is_running() -> None:
    await asyncio.wait_for(JobRegistry().idle(), timeout=1)


async def test_a_job_already_running_is_not_started_a_second_time() -> None:
    """A replay arriving before the first copy began ran one corpus twice, concurrently."""
    registry = JobRegistry()
    gate = asyncio.Event()
    ran: list[str] = []
    await registry.submit("job_a", hold(gate, ran, "first"))

    started = await registry.submit("job_a", hold(gate, ran, "second"))
    gate.set()
    await registry.idle()

    assert started is False
    assert ran == ["first"]


async def test_a_job_id_can_run_again_once_its_first_run_has_finished() -> None:
    """Resuming a job submits the id it already used."""
    registry = JobRegistry()
    gate = asyncio.Event()
    gate.set()
    ran: list[str] = []
    await registry.submit("job_a", hold(gate, ran, "first"))
    await registry.idle()

    started = await registry.submit("job_a", hold(gate, ran, "second"))
    await registry.idle()

    assert started is True
    assert ran == ["first", "second"]


async def test_a_job_that_raises_is_forgotten_rather_than_left_in_flight() -> None:
    """A failed task still listed as active would make every later drain wait for it."""

    async def explode() -> None:
        raise RuntimeError("the job's own handlers missed this")

    registry = JobRegistry()

    await registry.submit("job_a", explode())
    await registry.idle()

    assert registry.active == frozenset()


async def test_a_drain_waits_for_a_job_that_finishes_inside_the_grace_period() -> None:
    """Shutdown used to wait for nothing."""
    registry = JobRegistry()
    gate = asyncio.Event()
    await registry.submit("job_a", hold(gate, [], "job_a"))
    asyncio.get_running_loop().call_later(0.05, gate.set)

    report = await registry.drain(grace_seconds=10)

    assert (report.finished, report.cancelled, report.in_flight) == (1, 0, 1)
    assert registry.active == frozenset()


async def test_a_drain_cancels_a_job_that_outlives_the_grace_period() -> None:
    """A job still running at exit was abandoned mid-write rather than stopped."""
    registry = JobRegistry()
    unwound: list[str] = []

    async def stubborn() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01)
            unwound.append("job_a")

    await registry.submit("job_a", stubborn())
    await asyncio.sleep(0)

    report = await registry.drain(grace_seconds=0.05)

    assert (report.finished, report.cancelled) == (0, 1)
    assert unwound == ["job_a"], "the drain returned before the job finished unwinding"
    assert registry.active == frozenset()


async def test_a_drain_with_nothing_in_flight_returns_at_once() -> None:
    report = await asyncio.wait_for(JobRegistry().drain(grace_seconds=60), timeout=1)

    assert report.in_flight == 0


async def test_a_job_offered_after_the_drain_began_is_refused_as_retryable() -> None:
    """Accepting it would record a job that nothing was left to run."""
    registry = JobRegistry()
    ran: list[str] = []
    gate = asyncio.Event()
    gate.set()
    await registry.drain(grace_seconds=0)

    with pytest.raises(FasterRagError) as refused:
        await registry.submit("job_a", hold(gate, ran, "job_a"))
    await asyncio.sleep(0)

    assert refused.value.code is ErrorCode.NOT_READY
    assert refused.value.retryable is True
    assert refused.value.status == 503
    assert registry.accepting is False
    assert ran == []


async def test_a_registry_opened_again_accepts_jobs_again() -> None:
    registry = JobRegistry()
    gate = asyncio.Event()
    gate.set()
    await registry.drain(grace_seconds=0)

    registry.open()
    started = await registry.submit("job_a", hold(gate, [], "job_a"))
    await registry.idle()

    assert started is True
