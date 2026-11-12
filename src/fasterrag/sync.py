"""The blocking facade: fasterRag from a script, a notebook, or a WSGI app.

A thin adapter over :class:`fasterrag.facade.FasterRag`, not a second implementation. Every
method here does one thing — run the async method to completion on a managed event loop —
so there is no path through this module that behaves differently from the async one
(``docs/python-api.md``).

The loop is owned, not borrowed. Entering the context manager creates a private event loop
and closes it on exit, because the alternatives are both wrong: ``asyncio.run`` per call
would tear down the connection pools between calls, and reusing an ambient loop would make
this class fail unpredictably depending on what the caller happened to be running.

**Inside a running loop the private loop gets a thread of its own** (TASK-0497). A notebook
always has a running loop, so refusing to start under one — what this class used to do —
refused the use its first line names. Driving the private loop with ``run_until_complete``
from a thread that is already running a loop is impossible, and blocking that thread on a
future its own loop must resolve deadlocks. So the first call made under a running loop
moves the private loop onto a dedicated daemon thread, which runs it until ``__exit__``
stops and joins it; every call is then handed over with ``run_coroutine_threadsafe`` and the
caller blocks on the result while its own loop stays where it was. Only a call from that
dedicated thread is refused (``USAGE_ERROR``): it would wait on the loop it is running.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Coroutine, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Self, TypeVar

from fasterrag.adapters.vectordb.base import CollectionInfo
from fasterrag.config.loader import DEFAULT_CONFIG_PATH
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.facade import FasterRag as AsyncFasterRag
from fasterrag.services.doctor import DoctorReport
from fasterrag.services.documents import DEFAULT_PAGE_SIZE, DocumentDeletion, DocumentListing
from fasterrag.services.estimation import Estimate
from fasterrag.services.generation import Answer, QueryEvent
from fasterrag.services.journal import JobRecord
from fasterrag.services.lockfile import IndexLock
from fasterrag.services.replay import ReplayResult

__all__ = ["FasterRag"]

_Result = TypeVar("_Result")

_LOOP_THREAD_NAME = "fasterrag-sync-loop"


async def _awaited[T](awaitable: Awaitable[T]) -> T:
    """Wrap any awaitable in a coroutine, which ``run_coroutine_threadsafe`` requires."""
    return await awaitable


def _ambient_loop_running() -> bool:
    """Return whether the calling thread is already running an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class FasterRag:
    """A blocking fasterRag instance."""

    def __init__(self, inner: AsyncFasterRag) -> None:
        """Wrap an async facade. Prefer :meth:`from_config` or :meth:`from_settings`."""
        self._inner = inner
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Self:
        """Load and validate ``config.yaml``, then build a blocking facade over it."""
        return cls(AsyncFasterRag.from_config(path))

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build a blocking facade over an already-validated ``Settings``."""
        return cls(AsyncFasterRag.from_settings(settings))

    @property
    def settings(self) -> Settings:
        """Return the validated settings this instance runs on."""
        return self._inner.settings

    def __enter__(self) -> Self:
        """Create the private event loop and start the underlying facade."""
        self._loop = asyncio.new_event_loop()
        try:
            self._run(self._inner.__aenter__())
        except BaseException:
            self._close_loop()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the underlying facade, then stop, join and close the loop this instance owns."""
        if self._loop is None:
            return
        try:
            self._run(self._inner.__aexit__(exc_type, exc, traceback))
        finally:
            self._close_loop()

    def _close_loop(self) -> None:
        """Stop the loop's thread if it has one, wait for it to finish, and close the loop."""
        loop, thread = self._loop, self._thread
        self._loop = None
        self._thread = None
        if loop is None:
            return
        if thread is not None:
            loop.call_soon_threadsafe(loop.stop)
            if thread is not threading.current_thread():
                thread.join()
        if not loop.is_running():
            loop.close()

    def _start_loop_thread(self, loop: asyncio.AbstractEventLoop) -> threading.Thread:
        """Run the private loop on a dedicated daemon thread until :meth:`_close_loop`."""
        thread = threading.Thread(target=loop.run_forever, name=_LOOP_THREAD_NAME, daemon=True)
        thread.start()
        self._thread = thread
        return thread

    def _run(self, awaitable: Awaitable[_Result]) -> _Result:
        """Drive one awaitable to completion on the owned loop, from whatever thread calls."""
        loop = self._loop
        if loop is None:
            self._refuse(
                awaitable,
                "this FasterRag instance has not been started; use it as a context manager: "
                "'with FasterRag.from_config(...) as rag:'",
            )
        thread = self._thread
        if thread is None and not _ambient_loop_running():
            return loop.run_until_complete(awaitable)
        if thread is threading.current_thread():
            self._refuse(
                awaitable,
                "fasterrag.sync.FasterRag was called from the thread that runs its own event "
                "loop, which would wait on itself forever. Use 'from fasterrag import "
                "FasterRag' and 'async with' there instead",
            )
        if thread is None:
            self._start_loop_thread(loop)

        future = asyncio.run_coroutine_threadsafe(_awaited(awaitable), loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    @staticmethod
    def _refuse(awaitable: Awaitable[Any], detail: str) -> NoReturn:
        """Raise ``USAGE_ERROR``, closing the coroutine so it is not reported as never awaited."""
        if isinstance(awaitable, Coroutine):
            awaitable.close()
        raise FasterRagError(detail, code=ErrorCode.USAGE_ERROR)

    def ingest(
        self,
        sources: Sequence[str],
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
        recursive: bool = False,
    ) -> JobRecord:
        """Ingest sources, blocking until the job settles."""
        return self._run(
            self._inner.ingest(
                sources,
                collection=collection,
                metadata=metadata,
                tenant=tenant,
                idempotency_key=idempotency_key,
                recursive=recursive,
            )
        )

    def query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        tenant: str | None = None,
    ) -> Answer:
        """Answer a question through the full pipeline, blocking until it is complete."""
        return self._run(
            self._inner.query(
                text, collection=collection, top_k=top_k, filters=filters, tenant=tenant
            )
        )

    def query_stream(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        tenant: str | None = None,
    ) -> Iterator[QueryEvent]:
        """Yield the answer's events as an ordinary iterator."""
        events = self._inner.query_stream(
            text, collection=collection, top_k=top_k, filters=filters, tenant=tenant
        )
        while True:
            try:
                yield self._run(events.__anext__())
            except StopAsyncIteration:
                return

    def retrieve(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        tenant: str | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve without generating, blocking until the results are in."""
        return self._run(
            self._inner.retrieve(
                text, collection=collection, top_k=top_k, filters=filters, tenant=tenant
            )
        )

    def doctor(self) -> DoctorReport:
        """Run the D10 preflight checks, blocking.

        Uses its own short-lived loop when the facade has not been entered, because doctor
        must work on an installation that cannot start — that is what it is for.
        """
        if self._loop is None:
            if not _ambient_loop_running():
                return asyncio.run(self._inner.doctor())
            with ThreadPoolExecutor(1, thread_name_prefix=_LOOP_THREAD_NAME) as worker:
                return worker.submit(asyncio.run, self._inner.doctor()).result()
        return self._run(self._inner.doctor())

    def collections(self) -> list[CollectionInfo]:
        """List the collections the vector database holds."""
        return self._run(self._inner.collections())

    def create_collection(
        self,
        name: str,
        *,
        distance: str | None = None,
        shard_number: int = 1,
        replication_factor: int = 1,
    ) -> None:
        """Create a collection sized for the configured embedding model."""
        self._run(
            self._inner.create_collection(
                name,
                distance=distance,
                shard_number=shard_number,
                replication_factor=replication_factor,
            )
        )

    def drop_collection(self, name: str) -> bool:
        """Drop a collection, returning whether it existed."""
        return self._run(self._inner.drop_collection(name))

    def delete_document(
        self,
        document_id: str,
        *,
        collection: str | None = None,
        tenant: str | None = None,
    ) -> DocumentDeletion:
        """Delete one document's chunks and every record that it was indexed, blocking."""
        return self._run(
            self._inner.delete_document(document_id, collection=collection, tenant=tenant)
        )

    def documents(
        self,
        collection: str | None = None,
        *,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> DocumentListing:
        """Return one page of the documents the journal records in a collection, blocking."""
        return self._run(self._inner.documents(collection, offset=offset, limit=limit))

    def replay(self, trace_id: str, candidate: Settings | None = None) -> ReplayResult:
        """Re-execute a stored query under a candidate configuration and diff it (D8)."""
        return self._run(self._inner.replay(trace_id, candidate))

    def estimate(self, sources: Sequence[str], *, all_providers: bool = False) -> Estimate:
        """Report what ingesting ``sources`` would cost (D9).

        Already synchronous on the async facade, so this delegates directly and works
        without entering the context manager.
        """
        return self._inner.estimate(sources, all_providers=all_providers)

    def index_lock(self, collection: str | None = None) -> IndexLock | None:
        """Return a collection's index lockfile, or ``None`` if there is none (D1)."""
        return self._inner.index_lock(collection)
