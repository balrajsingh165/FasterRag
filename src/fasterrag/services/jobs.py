"""The handle on every ingestion job this process is running (TASK-0436).

``POST /v1/ingest`` answers ``202`` and the work continues after the response. It used to
continue as a Starlette background task, which nothing held a reference to: the process
could not say which jobs it was running, could not wait for them, and could not stop them,
so a shutdown closed the vector database under a job that was still writing to it and the
job's record stayed ``running`` for good.

:class:`JobRegistry` is that missing reference. A job is *submitted* under its id, runs as
an ``asyncio.Task`` the registry keeps until it finishes, and is forgotten the moment it
does. Everything a lifecycle needs follows from being able to enumerate what is in flight.

**This is the seam for a queue-backed worker** (TASK-0130). A caller knows three things
about the registry — it can ``submit`` a job, ask what is ``active``, and wait for it to be
``idle`` — and none of them says the work happens in this process. ``submit`` is ``async``
for that reason and no other: starting a task needs no ``await``, but putting a job on a
broker does, and a caller written against a synchronous method would have to change when
the broker arrives.

Submitting a job id that is already running here starts nothing. A replayed request that
arrives before the first copy of the job has begun would otherwise run the same corpus
twice, concurrently, against one journal record.

**A job the process stops working on is settled, not abandoned.** Cancellation is how a
shutdown stops a job, and ``asyncio.CancelledError`` is a ``BaseException`` that no
``except Exception`` sees, so nothing used to settle a cancelled job: its record stayed
``running`` and a client polling it could not tell that from slow progress.
:func:`settle_interrupted` writes ``failed`` with the reason :data:`INTERRUPTED`, on top of
the *current* record so the checkpoint the job reached is kept and a later run resumes from
it rather than from the start. ``INTERRUPTED`` is a reason on the job, not an error code: no
request ever fails with it.

**Draining is the registry's, not the caller's.** :meth:`JobRegistry.drain` closes the
door, waits a grace period for what is in flight, cancels what remains, and returns only
when every job has finished unwinding — so whoever called it may then close the adapters
those jobs were using. A job offered while the door is closed is refused with
``NOT_READY``, which is 503 and retryable, *before* a record is written for it: accepting a
job that will never run would leave a ``queued`` record with nothing to pick it up.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Final

from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.journal import JobRecord, Journal

__all__ = ["INTERRUPTED", "DrainReport", "JobRegistry", "settle_interrupted"]

INTERRUPTED: Final = "INTERRUPTED"

_UNSETTLED: Final[frozenset[str]] = frozenset({"queued", "running"})

_logger = get_logger(__name__)


async def settle_interrupted(journal: Journal, record: JobRecord) -> JobRecord:
    """Settle a job nothing is working on as ``failed`` / ``INTERRUPTED`` and return it.

    The current record is the base, so the last checkpoint survives; a job that already
    reached a terminal status is returned untouched.
    """
    try:
        current = await journal.load_job(record.job_id)
    except FasterRagError as exc:
        _logger.warning(
            "the record of an interrupted job could not be read; settling the accepted copy",
            extra={"job_id": record.job_id, "code": exc.code.value, "trace_id": exc.trace_id},
        )
        current = record

    if current.status not in _UNSETTLED:
        return current

    settled = replace(
        current,
        status="failed",
        reason_code=INTERRUPTED,
        finished_at=datetime.now(tz=UTC).isoformat(),
    )
    await journal.save_job(settled)
    _logger.warning(
        "ingest job interrupted",
        extra={
            "job_id": settled.job_id,
            "reason_code": INTERRUPTED,
            "resume_from": journal.resume_index(settled),
        },
    )
    return settled


@dataclass(frozen=True, slots=True)
class DrainReport:
    """What a drain found in flight, and how each job left."""

    finished: int = 0
    cancelled: int = 0

    @property
    def in_flight(self) -> int:
        """Return how many jobs were running when the drain began."""
        return self.finished + self.cancelled


class JobRegistry:
    """Runs submitted jobs as tasks it can enumerate, wait for, and stop."""

    def __init__(self) -> None:
        """Start with nothing in flight and the door open."""
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._accepting = True

    @property
    def accepting(self) -> bool:
        """Return whether a new job may be started here."""
        return self._accepting

    def open(self) -> None:
        """Accept jobs again, as an application starting up after a drain must."""
        self._accepting = True

    def require_accepting(self) -> None:
        """Refuse, as retryable, a job offered to a process that is shutting down."""
        if not self._accepting:
            raise FasterRagError(
                "this server is shutting down and is not starting new ingestion jobs; "
                "nothing was recorded, so retry the same request against another instance "
                "or once this one is back",
                code=ErrorCode.NOT_READY,
                retryable=True,
            )

    @property
    def active(self) -> frozenset[str]:
        """Return the ids of the jobs this process is running right now."""
        return frozenset(job for job, task in self._tasks.items() if not task.done())

    async def submit(self, job_id: str, work: Coroutine[Any, Any, None]) -> bool:
        """Start ``work`` for ``job_id`` and return whether it was started.

        A job already running here is left alone and ``work`` is closed unstarted, so one
        job id never has two runs writing its record.
        """
        if not self._accepting:
            work.close()
            self.require_accepting()

        if job_id in self.active:
            work.close()
            _logger.info(
                "job is already running in this process; not starting it again",
                extra={"job_id": job_id},
            )
            return False

        task = asyncio.create_task(work, name=f"ingest:{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda finished: self._forget(job_id, finished))
        return True

    async def idle(self) -> None:
        """Return once nothing submitted so far is still running."""
        while True:
            for job_id, task in list(self._tasks.items()):
                if task.done():
                    self._forget(job_id, task)
            if not self._tasks:
                return
            await asyncio.wait(list(self._tasks.values()))

    async def drain(self, grace_seconds: float) -> DrainReport:
        """Stop accepting jobs, wait ``grace_seconds`` for those in flight, cancel the rest.

        Returns only once every job has finished or has unwound from its cancellation,
        which is what makes it safe for the caller to close what the jobs were using.
        """
        self._accepting = False
        in_flight = [task for task in self._tasks.values() if not task.done()]
        if not in_flight:
            return DrainReport()

        _, unfinished = await asyncio.wait(in_flight, timeout=grace_seconds)
        for task in unfinished:
            task.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)

        await self.idle()
        return DrainReport(finished=len(in_flight) - len(unfinished), cancelled=len(unfinished))

    def _forget(self, job_id: str, task: asyncio.Task[None]) -> None:
        """Drop a finished task, reporting a failure its own handlers did not absorb."""
        if self._tasks.get(job_id) is not task:
            return

        del self._tasks[job_id]
        if task.cancelled():
            return

        error = task.exception()
        if error is not None:
            _logger.error(
                "a job ended on an error nothing handled",
                extra={
                    "job_id": job_id,
                    "code": ErrorCode.INTERNAL.value,
                    "detail": f"{type(error).__name__}: {error}",
                },
            )
