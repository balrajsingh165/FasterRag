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
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from fasterrag.errors import ErrorCode
from fasterrag.observability.logging import get_logger

__all__ = ["JobRegistry"]

_logger = get_logger(__name__)


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
