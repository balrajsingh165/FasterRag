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
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Final

from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.journal import JobRecord, Journal

__all__ = ["INTERRUPTED", "JobRegistry", "settle_interrupted"]

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


class JobRegistry:
    """Runs submitted jobs as tasks it can enumerate, wait for, and stop."""

    def __init__(self) -> None:
        """Start with nothing in flight."""
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def active(self) -> frozenset[str]:
        """Return the ids of the jobs this process is running right now."""
        return frozenset(job for job, task in self._tasks.items() if not task.done())

    async def submit(self, job_id: str, work: Coroutine[Any, Any, None]) -> bool:
        """Start ``work`` for ``job_id`` and return whether it was started.

        A job already running here is left alone and ``work`` is closed unstarted, so one
        job id never has two runs writing its record.
        """
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
