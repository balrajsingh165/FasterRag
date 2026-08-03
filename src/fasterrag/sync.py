"""The blocking facade: fasterRag from a script, a notebook, or a WSGI app.

A thin adapter over :class:`fasterrag.facade.FasterRag`, not a second implementation. Every
method here does one thing — run the async method to completion on a managed event loop —
so there is no path through this module that behaves differently from the async one
(``docs/python-api.md``).

The loop is owned, not borrowed. Entering the context manager creates a private event loop
and closes it on exit, because the alternatives are both wrong: ``asyncio.run`` per call
would tear down the connection pools between calls, and reusing an ambient loop would make
this class fail unpredictably depending on what the caller happened to be running.

That ownership is also why this refuses to start inside a running loop. Blocking on a future
from the thread already driving the loop deadlocks — the loop cannot make progress because
the thread that would advance it is waiting. Detected up front with a message naming the
async facade, rather than left to hang.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine, Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypeVar

from fasterrag.config.loader import DEFAULT_CONFIG_PATH
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.facade import FasterRag as AsyncFasterRag
from fasterrag.services.estimation import Estimate
from fasterrag.services.generation import Answer, QueryEvent
from fasterrag.services.journal import JobRecord
from fasterrag.services.lockfile import IndexLock

__all__ = ["FasterRag"]

_Result = TypeVar("_Result")


class FasterRag:
    """A blocking fasterRag instance.

    Mirrors :class:`fasterrag.facade.FasterRag` method for method::

        from fasterrag.sync import FasterRag

        with FasterRag.from_config("config.yaml") as rag:
            rag.ingest(["./docs"])
            print(rag.query("What does the spec say about retries?").answer)

    Use the async facade instead whenever the calling code already has an event loop; this
    one exists for the code that does not.
    """

    def __init__(self, inner: AsyncFasterRag) -> None:
        """Wrap an async facade. Prefer :meth:`from_config` or :meth:`from_settings`."""
        self._inner = inner
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Self:
        """Load and validate ``config.yaml``, then build a blocking facade over it.

        Raises:
            ConfigError: On the same fail-fast contract the async facade has — missing file,
                malformed YAML, schema violation, or an unset referenced environment
                variable, naming the offending key.
        """
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
        """Create the private event loop and start the underlying facade.

        Raises:
            FasterRagError: If a loop is already running on this thread. Blocking on it from
                here would deadlock, and a deadlock reports nothing at all.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise FasterRagError(
                "fasterrag.sync.FasterRag cannot be used inside a running event loop, "
                "because blocking on it from the thread driving that loop would deadlock. "
                "Use 'from fasterrag import FasterRag' and 'async with' instead",
                code=ErrorCode.CONFIG_INVALID,
            )

        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._inner.__aenter__())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the underlying facade and close the loop this instance owns.

        The loop is closed in a ``finally`` so a shutdown failure cannot leak it. Leaking an
        event loop is quiet — nothing raises, the process just holds a selector and its file
        descriptors until it exits.
        """
        if self._loop is None:
            return
        try:
            self._loop.run_until_complete(self._inner.__aexit__(exc_type, exc, traceback))
        finally:
            self._loop.close()
            self._loop = None

    def _run(self, awaitable: Awaitable[_Result]) -> _Result:
        """Drive one coroutine to completion on the owned loop.

        Raises:
            FasterRagError: If the instance was never entered. Closing the coroutine before
                raising keeps Python from also emitting an unawaited-coroutine warning that
                would bury the real message.
        """
        if self._loop is None:
            if isinstance(awaitable, Coroutine):
                awaitable.close()
            raise FasterRagError(
                "this FasterRag instance has not been started; use it as a context manager: "
                "'with FasterRag.from_config(...) as rag:'",
                code=ErrorCode.CONFIG_INVALID,
            )
        return self._loop.run_until_complete(awaitable)

    def ingest(
        self,
        sources: Sequence[str],
        *,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        tenant: str | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        """Ingest sources, blocking until the job settles."""
        return self._run(
            self._inner.ingest(
                sources,
                collection=collection,
                metadata=metadata,
                tenant=tenant,
                idempotency_key=idempotency_key,
            )
        )

    def query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Answer:
        """Answer a question through the full pipeline, blocking until it is complete."""
        return self._run(
            self._inner.query(text, collection=collection, top_k=top_k, filters=filters)
        )

    def query_stream(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Iterator[QueryEvent]:
        """Yield the answer's events as an ordinary iterator.

        Each event is pulled with its own ``run_until_complete``, so tokens arrive as the
        model produces them rather than after the whole answer is buffered. Buffering would
        make the return type honest and the behavior useless — the reason to stream is to
        see the first token early.
        """
        events = self._inner.query_stream(text, collection=collection, top_k=top_k, filters=filters)
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
    ) -> list[ScoredChunk]:
        """Retrieve without generating, blocking until the results are in."""
        return self._run(
            self._inner.retrieve(text, collection=collection, top_k=top_k, filters=filters)
        )

    def estimate(self, sources: Sequence[str], *, all_providers: bool = False) -> Estimate:
        """Report what ingesting ``sources`` would cost (D9).

        Already synchronous on the async facade, so this delegates directly and works
        without entering the context manager.
        """
        return self._inner.estimate(sources, all_providers=all_providers)

    def index_lock(self, collection: str | None = None) -> IndexLock | None:
        """Return a collection's index lockfile, or ``None`` if there is none (D1)."""
        return self._inner.index_lock(collection)
