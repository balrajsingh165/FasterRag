"""The job registry: the handle on work that outlives the request that started it.

Background ingestion used to be a Starlette background task nothing referenced, so the
process could not enumerate, await, or stop its own jobs (TASK-0436).
"""

import asyncio

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
