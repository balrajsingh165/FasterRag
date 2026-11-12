"""CPU worker pool: load, parse, and chunk.

Loading, parsing, and chunking are CPU-bound, so they run in worker processes rather than
on the event loop (``docs/architecture.md`` §2). Chunks stream to the embedding pool as
each document finishes instead of after the whole corpus is parsed, which is what keeps
expensive embedding workers from idling behind the parser.

Failure is per-document by design. A corrupt file becomes a dead-letter entry with a reason
code and the pipeline continues — including a parser that raises something untyped, which is
converted to a ``ParseError`` inside the worker rather than aborting the job. A worker
*process* dying is different: the standard library marks the whole pool broken, so the job
fails rather than continuing on a pool that can no longer accept work. A pool that dies
before its first worker has run anything is reported as a script without a ``__main__``
guard; one that dies later was killed or crashed, and is reported as that (TASK-0509).

Leaving the pool with a parse still running — the job was cancelled, or failed elsewhere —
kills the workers instead of joining them. Joining runs on the event loop and lasts as long
as the slowest parse, which for a hung one is for ever, so a shutdown drain was bounded by
the parser rather than by ``reliability.shutdown_grace_seconds`` (TASK-0525).

Documents are parsed concurrently, up to the pool size at a time, and consumed in
submission order. Awaiting each document before submitting the next — which this module did
until 2026-11-11 — keeps exactly one worker busy however many the pool holds.

A parse has a deadline, ``ingestion.parse_timeout_seconds``, counted from the moment the
document is handed to the pool (TASK-0473). A parser that hangs — a pathological PDF, a
regular expression that backtracks for ever — used to stall the job without limit, because
the pool future was awaited with none. Cancelling that future does not help: a process-pool
task that is already running cannot be cancelled, and its worker stays occupied until it
returns, which a hung one never does. So an overdue document costs its *pool*: the workers
are killed, a fresh pool is started, the documents that were in flight beside it are
submitted again with a new deadline, and only the overdue one is dead-lettered, as
``PARSE_TIMEOUT``. One worker is started at pool start and after every recycle, and with a
deadline set, as many as the documents will occupy are started before any is submitted, so
that process start-up is not billed to a document's deadline. Before 2026-11-12 only the
first was: a pool that spawns starts the others on demand, and on a loaded machine a
healthy document could be dead-lettered while its worker was still starting.

A document is journalled as indexed only when the :class:`IndexLedger` says every chunk of
it was written downstream (``workers/ledger.py``). With no ledger passed there is no
downstream to wait for, and the chunks are acknowledged as they are enqueued.

A document that parses without error and yields no chunk — zero bytes, whitespace only —
is a third outcome, neither indexed nor dead-lettered: it is recorded ``skipped`` with the
reason ``EMPTY_DOCUMENT``, counted in :attr:`PoolReport.empty`, and never reaches the dedup
file, where the hash of nothing would match every empty file that came after (TASK-0470).

The semantic strategy needs a sentence embedder, and a pool task receives only the task and
the settings, so no model object can be handed to a worker. Each worker builds its own, once,
on first use, cached by model name (TASK-0431) — a local model only, because a provider
reached over the network cannot be driven from here. Loading a model can take longer than a
parse is allowed, so before any document is submitted — at pool start and after every
recycle — every worker loads it: warm-up tasks are sent until each worker process has
answered one, and no document's deadline is running while a model loads. A worker that never
takes a warm-up task within ``_WARM_STALL_SECONDS`` is logged and left to load on first use
rather than hanging the job. Each worker holds its own copy of the model, so the memory it
takes is ``workers.cpu_pool_size`` times the model's.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from multiprocessing.process import BaseProcess
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Final

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.chunking import SentenceEmbedder, create_chunker, create_token_counter
from fasterrag.core.identity import chunk_id, chunker_config_hash, content_hash, document_id
from fasterrag.core.parsing import create_parsing_options, parse_bytes
from fasterrag.errors import ConfigError, ErrorCode, FasterRagError, IngestionError, ParseError
from fasterrag.observability.logging import current_trace_id, get_logger
from fasterrag.services.journal import DocumentRecord, Journal
from fasterrag.workers.ledger import IndexLedger
from fasterrag.workers.queues import BoundedQueue, ChunkPayload, DocumentTask, ParseOutcome

__all__ = [
    "EMPTY_DOCUMENT",
    "LOCAL_SENTENCE_EMBEDDERS",
    "CpuWorkerPool",
    "PoolReport",
    "SentenceEmbedderFactory",
    "parse_and_chunk",
    "require_runnable_chunker",
    "reset_shared_sentence_embedders",
    "resolve_pool_size",
    "sentence_embedder",
    "use_sentence_embedder_factory",
]

EMPTY_DOCUMENT: Final = "EMPTY_DOCUMENT"
"""The reason a document is ``skipped``: it parsed, and there was nothing in it to index."""

_MEGABYTE = 1024 * 1024

_SLOW_TOKENIZER_LOAD_SECONDS = 0.5

_OVERDUE_GRACE_SECONDS: Final = 0.05
"""How close to its own deadline a document counts as overdue when the pool is recycled.

Documents submitted together are due microseconds apart. Without this, the second of two
hung documents would be resubmitted and given a whole new deadline to hang through.
"""

_WARM_RETRY_SECONDS: Final = 0.05

_WARM_STALL_SECONDS: Final = 30.0
"""How long warm-up waits for a worker that has started no warm-up task before giving up."""

LOCAL_SENTENCE_EMBEDDERS: Final[frozenset[str]] = frozenset({"huggingface"})
"""The ``embeddings.provider`` values whose model a parse worker can load for itself."""

SentenceEmbedderFactory = Callable[[Settings], SentenceEmbedder]
"""Builds the sentence embedder a worker process uses for the semantic strategy."""

_logger = get_logger(__name__)


def resolve_pool_size(configured: int) -> int:
    """Return the worker count, expanding the documented ``0`` to the CPU count."""
    if configured > 0:
        return configured
    return os.cpu_count() or 1


def _local_sentence_embedder(settings: Settings) -> SentenceEmbedder:
    """Return a sentence embedder over the configured local model, loaded on first use."""
    from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder
    from fasterrag.adapters.embeddings.sync import SentenceEmbedderBridge

    return SentenceEmbedderBridge(HuggingFaceEmbedder(settings))


_sentence_embedders: Final[dict[str, SentenceEmbedder]] = {}
_sentence_embedder_factory: SentenceEmbedderFactory = _local_sentence_embedder
_SENTENCE_EMBEDDER_LOCK: Final = Lock()


def use_sentence_embedder_factory(factory: SentenceEmbedderFactory | None) -> None:
    """Set how this process builds sentence embedders, dropping any it already built.

    ``None`` restores the local model. A spawned worker does not see the parent's state, so
    a test installs its own factory from the executor's worker initializer.
    """
    global _sentence_embedder_factory
    with _SENTENCE_EMBEDDER_LOCK:
        _sentence_embedder_factory = factory or _local_sentence_embedder
        _sentence_embedders.clear()


def reset_shared_sentence_embedders() -> None:
    """Drop this process's sentence embedders and restore the local-model factory."""
    use_sentence_embedder_factory(None)


def require_runnable_chunker(settings: Settings) -> None:
    """Refuse a chunking strategy the parse workers cannot run with these settings."""
    provider = settings.embeddings.provider
    if settings.chunking.strategy != "semantic" or provider in LOCAL_SENTENCE_EMBEDDERS:
        return
    raise ConfigError(
        "chunking.strategy is 'semantic', which needs a local embedding model: the parse "
        "workers find sentence boundaries with a model each of them loads, and "
        f"embeddings.provider is {provider!r}, a provider reached over the network, whose "
        "client cannot run there. Set embeddings.provider to 'huggingface' (a local "
        "sentence-transformers model), or set chunking.strategy to 'recursive', 'layout', "
        "'fixed' or 'late'"
    )


def sentence_embedder(settings: Settings) -> SentenceEmbedder:
    """Return this process's sentence embedder for the configured model, building it once."""
    require_runnable_chunker(settings)
    model = settings.embeddings.model
    with _SENTENCE_EMBEDDER_LOCK:
        embedder = _sentence_embedders.get(model)
        if embedder is None:
            embedder = _sentence_embedder_factory(settings)
            _sentence_embedders[model] = embedder
    return embedder


def _warm_sentence_embedder(settings: Settings) -> int:
    """Load this worker's sentence embedder, and return the process it lives in."""
    sentence_embedder(settings).embed(["warm"])
    return os.getpid()


@dataclass(frozen=True, slots=True)
class PoolReport:
    """What a pass over a job's documents produced."""

    parsed: int = 0
    chunked: int = 0
    deduplicated: int = 0
    dead_lettered: int = 0
    skipped: int = 0
    empty: int = 0
    flags: dict[str, int] = field(default_factory=dict)


def parse_and_chunk(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Load, parse, and chunk one document."""
    try:
        return _parse_and_chunk(task, settings)
    except FasterRagError:
        raise
    except Exception as exc:
        # CRITICAL: a parser library raising its own exception type (a corrupt DOCX, a
        # RecursionError on deeply nested JSON) must cost one document, not the job. Only
        # typed errors are dead-lettered by the pool, so the conversion happens here.
        raise ParseError(f"{task.source} could not be parsed: {type(exc).__name__}: {exc}") from exc


def _parse_and_chunk(task: DocumentTask, settings: Settings) -> ParseOutcome:
    """Do the work of :func:`parse_and_chunk`, raising whatever the parsers raise."""
    limit = settings.ingestion.max_document_mb * _MEGABYTE
    data = _read(task.readable, limit)
    document = parse_bytes(
        data,
        filename=Path(task.source).name,
        max_bytes=limit,
        options=create_parsing_options(settings),
    )

    embedder = sentence_embedder(settings) if settings.chunking.strategy == "semantic" else None
    chunker = create_chunker(settings, embedder=embedder)
    chunker_hash = chunker_config_hash(settings)
    digest = content_hash(data)

    payloads = [
        ChunkPayload(
            chunk_id=chunk_id(task.document_id, chunk.chunk_index, chunker_hash),
            document_id=task.document_id,
            source=task.source,
            content_hash=digest,
            chunk=chunk,
            metadata={**dict(task.metadata), **document.metadata},
            tenant=task.tenant,
        )
        for chunk in chunker.split(document)
    ]

    return ParseOutcome(
        task=task,
        chunks=payloads,
        content_hash=digest,
        parser=document.parser,
        mime_type=document.mime_type,
        parse_flags=document.flags,
        document_text=(
            "\n\n".join(block.text for block in document.blocks)
            if settings.chunking.contextual_enrichment or settings.chunking.strategy == "late"
            else ""
        ),
    )


def _read(source: str, max_bytes: int) -> bytes:
    """Read a source's bytes, refusing an oversized file before loading it.

    The size is checked from the directory entry rather than after reading, so a file far
    larger than memory is rejected instead of ingested.
    """
    path = Path(source)
    if not path.is_file():
        raise ParseError(f"{source} is not a readable file")

    size = path.stat().st_size
    if size > max_bytes:
        raise IngestionError(
            f"{source} is {size} bytes, above the configured limit of {max_bytes}",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            retryable=False,
        )

    try:
        return path.read_bytes()
    except OSError as exc:
        raise ParseError(f"{source} could not be read: {exc.strerror}") from exc


@dataclass(slots=True)
class _InFlight:
    """One document in the pool: its task, its pending result, and when it is due."""

    task: DocumentTask
    future: asyncio.Future[ParseOutcome]
    deadline: float | None
    timed_out: bool = False


def _ready() -> bool:
    """Return from a worker process, which proves the pool can start one."""
    return True


def _process_id(_settings: Settings) -> int:
    """Return the process this task ran in, which proves that worker has started."""
    return os.getpid()


def _abandon(executor: Executor) -> None:
    """Shut a pool down without waiting for it, killing any worker that is still busy."""
    workers: list[BaseProcess] = []
    if isinstance(executor, ProcessPoolExecutor) and executor._processes:
        # CRITICAL: `_processes` is private, and before Python 3.14 (`kill_workers`) there
        # is no public way to reach a pool's workers. Cancelling a future does not stop a
        # task that is already running and `shutdown` would wait for it, so a hung parser
        # can only be stopped by killing its process. Copied first: `shutdown` clears it.
        workers = list(executor._processes.values())
    executor.shutdown(wait=False, cancel_futures=True)
    for worker in workers:
        if worker.is_alive():
            worker.kill()


def _parse_timeout_error(task: DocumentTask, limit: float) -> ParseError:
    """Return the error for a document whose parse outlived its deadline."""
    return ParseError(
        f"{task.source} was still being parsed after {limit:g} seconds, the limit set by "
        "ingestion.parse_timeout_seconds; its worker was stopped and the job continued",
        code=ErrorCode.PARSE_TIMEOUT,
        retryable=False,
    )


def _unguarded_entry_point_error() -> IngestionError:
    """Return the error for a worker pool that died before doing any work."""
    return IngestionError(
        "the parsing worker pool died before processing anything. On Windows and macOS "
        "child processes re-import the module that started them, so a script must guard its "
        'entry point with: if __name__ == "__main__": asyncio.run(main()) — without the '
        "guard every worker re-runs the script instead of parsing. This is a multiprocessing "
        "requirement, not a fasterRag one",
        code=ErrorCode.CHUNK_FAILED,
        retryable=False,
    )


def _worker_died_error(during: str) -> IngestionError:
    """Return the error for a worker that died after the pool had started and run work."""
    return IngestionError(
        f"a parsing worker process died {during}, after the pool had started and run work, "
        "so the script's entry point is not the cause. Something outside fasterRag ended "
        "it: most often the operating system's out-of-memory killer, or a crash in a "
        "parser's native code. The job was stopped; if memory is short, lower "
        "workers.cpu_pool_size or ingestion.max_document_mb, then ingest the sources again",
        code=ErrorCode.CHUNK_FAILED,
        retryable=False,
    )


class CpuWorkerPool:
    """Parses and chunks documents in worker processes, streaming chunks downstream."""

    def __init__(
        self,
        settings: Settings,
        *,
        journal: Journal | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
        llm: LLMAdapter | None = None,
        trim: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> None:
        """Build the pool without starting any worker."""
        self.settings = settings
        self.journal = journal
        self.llm = llm
        self._trim = trim
        self.size = resolve_pool_size(settings.workers.cpu_pool_size)
        self._executor_factory = executor_factory or (
            lambda workers: ProcessPoolExecutor(max_workers=workers)
        )
        self._executor: Executor | None = None
        self._stranded = False

    async def __aenter__(self) -> CpuWorkerPool:
        """Start the worker processes and pay the tokenizer load up front."""
        self._executor = self._executor_factory(self.size)
        try:
            await self._start_worker()
        except BaseException:
            _abandon(self._executor)
            self._executor = None
            raise
        await self._warm_token_counter()
        _logger.info("cpu worker pool started", extra={"workers": self.size})
        return self

    async def _start_worker(self) -> None:
        """Start one worker now, so process start-up is not billed to a document's deadline."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, _ready)
        except BrokenProcessPool as exc:
            raise _unguarded_entry_point_error() from exc
        await self._warm_sentence_embedders()

    async def _warm_sentence_embedders(self) -> None:
        """Load the sentence embedder in every worker before any document's deadline runs.

        Threads share their process's embedder, so a thread pool needs one warm-up.
        """
        if self.settings.chunking.strategy != "semantic":
            return

        started = perf_counter()
        expected = self.size if isinstance(self._executor, ProcessPoolExecutor) else 1
        try:
            warmed = await self._warmed_processes(expected)
        except BrokenProcessPool as exc:
            raise IngestionError(
                "a parse worker died while loading the sentence embedding model "
                f"{self.settings.embeddings.model!r} for chunking.strategy 'semantic'. Every "
                "worker loads its own copy, so the memory needed is workers.cpu_pool_size "
                "times the model's; lower workers.cpu_pool_size or choose a smaller model",
                code=ErrorCode.CHUNK_FAILED,
                retryable=False,
            ) from exc

        details = {
            "seconds": round(perf_counter() - started, 2),
            "workers": warmed,
            "model": self.settings.embeddings.model,
        }
        if warmed < expected:
            _logger.warning(
                "not every parse worker loaded the sentence embedding model before the first "
                "document; one that loads it later does so inside a document's parse deadline",
                extra={**details, "expected": expected},
            )
        else:
            _logger.info("loaded the sentence embedding model in every parse worker", extra=details)

    async def _warmed_processes(
        self, expected: int, target: Callable[[Settings], int] = _warm_sentence_embedder
    ) -> int:
        """Send warm-up tasks until ``expected`` processes have answered, and return how many did.

        Gives up after ``_WARM_STALL_SECONDS`` in which only already-warm processes answered, so
        an executor that never starts the workers it was asked for cannot hang the job.
        """
        loop = asyncio.get_running_loop()
        warmed: set[int] = set()
        stalled_since: float | None = None
        while len(warmed) < expected:
            replies = await asyncio.gather(
                *(
                    loop.run_in_executor(self._executor, target, self.settings)
                    for _ in range(expected - len(warmed))
                )
            )
            fresh = set(replies) - warmed
            warmed |= fresh
            if fresh:
                stalled_since = None
                continue
            stalled_since = stalled_since or perf_counter()
            if perf_counter() - stalled_since > _WARM_STALL_SECONDS:
                break
            await asyncio.sleep(_WARM_RETRY_SECONDS)
        return len(warmed)

    async def _start_workers(self, documents: int) -> None:
        """Start the workers ``documents`` will occupy before any of their deadlines runs.

        A pool that spawns starts workers on demand, which billed every worker's start-up but
        the first to a document's ``ingestion.parse_timeout_seconds``.
        """
        expected = min(documents, self.size)
        if (
            expected <= 1
            or self.settings.ingestion.parse_timeout_seconds <= 0
            or not isinstance(self._executor, ProcessPoolExecutor)
        ):
            return

        try:
            started = await self._warmed_processes(expected, _process_id)
        except BrokenProcessPool as exc:
            raise _worker_died_error("while the pool was starting its workers") from exc
        if started < expected:
            _logger.warning(
                "not every parse worker started before the first document; one that starts "
                "later does so inside a document's parse deadline",
                extra={"workers": started, "expected": expected},
            )

    async def _warm_token_counter(self) -> None:
        """Load the embedding tokenizer before the first document rather than during it."""
        counter = create_token_counter(self.settings)
        started = perf_counter()
        await asyncio.get_running_loop().run_in_executor(None, counter.count, "warm")
        elapsed = perf_counter() - started
        if elapsed > _SLOW_TOKENIZER_LOAD_SECONDS:
            _logger.info(
                "loaded the embedding tokenizer",
                extra={"seconds": round(elapsed, 2), "model": self.settings.embeddings.model},
            )

    async def __aexit__(self, *exc_info: object) -> None:
        """Shut the worker processes down, killing them if the block left work in them.

        Joining a busy worker blocks the event loop until its parse returns (TASK-0525).
        """
        executor, self._executor = self._executor, None
        stranded, self._stranded = self._stranded, False
        if executor is not None and (stranded or exc_info[0] is not None):
            _abandon(executor)
            if stranded:
                _logger.warning(
                    "the cpu worker pool was stopped with documents still being parsed; "
                    "its workers were killed rather than waited for",
                    extra={"trace_id": current_trace_id()},
                )
        elif executor is not None:
            executor.shutdown(wait=True)
        _logger.info("cpu worker pool stopped")

    async def process(
        self,
        tasks: Iterable[DocumentTask],
        sink: BoundedQueue[ChunkPayload],
        *,
        job: str | None = None,
        collection: str = "default",
        resume_from: int = 0,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
        ledger: IndexLedger | None = None,
    ) -> PoolReport:
        """Parse and chunk every task, streaming chunks into ``sink``."""
        if self._executor is None:
            raise IngestionError(
                "the cpu worker pool is not running; use it as an async context manager",
                code=ErrorCode.INTERNAL,
                retryable=False,
            )

        known: dict[str, str] = {}
        if self.journal is not None and self.settings.ingestion.dedup:
            known = await self.journal.known_content(collection)
        indexed_before = frozenset(known.values())
        standalone = ledger is None
        tracker = ledger or IndexLedger(on_progress, start=resume_from)
        parsed = chunked = deduplicated = dead_lettered = empty = 0
        flags: dict[str, int] = {}

        ordered = list(tasks)
        pending = [task for task in ordered if task.index >= resume_from]
        skipped = len(ordered) - len(pending)

        stream = self._parsed(pending)
        try:
            async for task, result in stream:
                if isinstance(result, FasterRagError):
                    dead_lettered += 1
                    async with tracker.journal_lock:
                        await self._dead_letter(job, task, result)
                    await tracker.settle(task.index)
                    continue

                outcome = result
                if not outcome.chunks:
                    # CRITICAL: before the dedup check, and never through the ledger's
                    # commit. "indexed" is a statement about the vector database, and the
                    # hash of nothing would match every later empty file (TASK-0470).
                    empty += 1
                    async with tracker.journal_lock:
                        if task.document_id in indexed_before:
                            await self._retire(collection, task)
                        await self._skip_empty(job, task, outcome)
                    await tracker.settle(task.index)
                    continue

                if outcome.content_hash in known:
                    deduplicated += 1
                    async with tracker.journal_lock:
                        await self._record(job, task, "deduplicated", outcome.content_hash)
                    await tracker.settle(task.index)
                    continue

                outcome = await self._enriched(outcome)

                for flag in outcome.parse_flags:
                    flags[flag] = flags.get(flag, 0) + 1

                payloads = self._addressed(outcome)
                await tracker.expect(
                    task.document_id,
                    index=task.index,
                    source=task.source,
                    chunks=len(payloads),
                    commit=self._commit(
                        job, collection, task, outcome, update=task.document_id in indexed_before
                    ),
                )
                for payload in payloads:
                    await sink.put(payload)

                parsed += 1
                chunked += len(outcome.chunks)
                known[outcome.content_hash] = task.document_id
                if standalone:
                    await tracker.written(payloads)
        finally:
            await stream.aclose()

        return PoolReport(
            parsed=parsed,
            chunked=chunked,
            deduplicated=deduplicated,
            dead_lettered=dead_lettered,
            skipped=skipped,
            empty=empty,
            flags=flags,
        )

    async def _parsed(
        self, tasks: Sequence[DocumentTask]
    ) -> AsyncGenerator[tuple[DocumentTask, ParseOutcome | FasterRagError]]:
        """Yield each task with its outcome or its typed failure, in submission order.

        Up to ``self.size`` documents are in the pool at once. Order is kept so the dedup
        decision between two identical documents in one job stays deterministic.
        """
        window: deque[_InFlight] = deque()
        queued = iter(tasks)

        def fill() -> None:
            while len(window) < self.size:
                task = next(queued, None)
                if task is None:
                    return
                window.append(self._submit(task))

        try:
            await self._start_workers(len(tasks))
            fill()
            while window:
                entry = window[0]
                if not entry.timed_out and not await self._returned(entry):
                    await self._recycle(window, len(tasks))
                window.popleft()
                if entry.timed_out:
                    yield entry.task, self._timeout_error(entry.task)
                    fill()
                    continue
                try:
                    outcome = await entry.future
                except BrokenProcessPool as exc:
                    raise _worker_died_error(f"while {entry.task.source} was in the pool") from exc
                except FasterRagError as exc:
                    # CRITICAL: the error was built in a worker, whose context is not this
                    # job's, so it carries a trace id the worker minted for itself. Without
                    # this the dead-letter entry cannot be joined to the job's own log lines.
                    exc.trace_id = current_trace_id()
                    yield entry.task, exc
                else:
                    yield entry.task, outcome
                fill()
        finally:
            for entry in window:
                self._stranded = self._stranded or not entry.future.done()
                entry.future.cancel()

    def _submit(self, task: DocumentTask) -> _InFlight:
        """Hand one document to the pool, due ``ingestion.parse_timeout_seconds`` from now."""
        loop = asyncio.get_running_loop()
        limit = self.settings.ingestion.parse_timeout_seconds
        try:
            future = loop.run_in_executor(self._executor, parse_and_chunk, task, self.settings)
        except BrokenProcessPool as exc:
            raise _worker_died_error(f"before {task.source} could be handed to the pool") from exc
        return _InFlight(task, future, loop.time() + limit if limit > 0 else None)

    async def _returned(self, entry: _InFlight) -> bool:
        """Wait for a document until it is due, and return whether its parse came back."""
        if entry.deadline is None:
            await asyncio.wait([entry.future])
            return True

        loop = asyncio.get_running_loop()
        while not entry.future.done():
            remaining = entry.deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.wait([entry.future], timeout=remaining)
        return True

    def _timeout_error(self, task: DocumentTask) -> ParseError:
        """Return the typed failure for a document that outlived its deadline."""
        return _parse_timeout_error(task, self.settings.ingestion.parse_timeout_seconds)

    async def _recycle(self, window: deque[_InFlight], documents: int) -> None:
        """Replace the pool because the document at the head of ``window`` is overdue.

        Every other unfinished document is submitted again to the new pool with a fresh
        deadline, unless it is overdue too; a finished one keeps the result it already has.
        """
        due = asyncio.get_running_loop().time() + _OVERDUE_GRACE_SECONDS
        window[0].timed_out = True
        unfinished = [entry for entry in window if not entry.future.done()]
        for entry in unfinished:
            # CRITICAL: cancelled before the workers are killed. Killing them fails every
            # future still in the pool with BrokenProcessPool; a cancelled future ignores
            # it, where a live one would carry an exception nobody is left to retrieve.
            entry.future.cancel()
            if entry.deadline is not None and entry.deadline <= due:
                entry.timed_out = True

        stale = self._executor
        self._executor = self._executor_factory(self.size)
        if stale is not None:
            _abandon(stale)
        await self._start_worker()
        await self._start_workers(documents)

        resubmitted = [entry for entry in unfinished if not entry.timed_out]
        for entry in resubmitted:
            fresh = self._submit(entry.task)
            entry.future, entry.deadline = fresh.future, fresh.deadline

        _logger.warning(
            "a document outlived its parse timeout; the cpu worker pool was recycled",
            extra={
                "document_id": window[0].task.document_id,
                "timeout_seconds": self.settings.ingestion.parse_timeout_seconds,
                "resubmitted": len(resubmitted),
                "trace_id": current_trace_id(),
            },
        )

    def _commit(
        self,
        job: str | None,
        collection: str,
        task: DocumentTask,
        outcome: ParseOutcome,
        *,
        update: bool = False,
    ) -> Callable[[], Awaitable[None]]:
        """Return the callback that journals this document once it is fully indexed.

        For an ``update`` — a document the journal already held — the chunks past the new
        length are trimmed first, so a failed trim leaves the document unrecorded (TASK-0443).
        """

        async def commit() -> None:
            if update and self._trim is not None:
                await self._trim(task.document_id, len(outcome.chunks))
            await self._index_document(job, collection, task, outcome)

        return commit

    def _addressed(self, outcome: ParseOutcome) -> list[ChunkPayload]:
        """Return the outcome's chunks, carrying the document text late chunking needs."""
        if self.settings.chunking.strategy != "late" or not outcome.document_text:
            return outcome.chunks

        return [replace(payload, document_text=outcome.document_text) for payload in outcome.chunks]

    async def _dead_letter(self, job: str | None, task: DocumentTask, exc: FasterRagError) -> None:
        """Route a failed document to the dead-letter queue, off the event loop."""
        _logger.warning(
            "document failed to parse",
            extra={
                "document_id": task.document_id,
                "code": exc.code.value,
                "trace_id": exc.trace_id,
            },
        )
        if self.journal is None or job is None:
            return

        await self.journal.dead_letter(
            job,
            document=task.document_id,
            source=task.source,
            reason_code=exc.code,
            detail=exc.detail,
            attempts=self.settings.ingestion.dlq.max_retries,
            trace_id=exc.trace_id,
        )

    async def _record(
        self, job: str | None, task: DocumentTask, status: str, digest: str | None = None
    ) -> None:
        """Record a document outcome when journalling is active, off the event loop."""
        if self.journal is None or job is None:
            return

        await self.journal.record_document(job, self._outcome(task, status, digest))

    async def _retire(self, collection: str, task: DocumentTask) -> None:
        """Remove the chunks and the dedup record of an indexed document that is now empty.

        Its whole previous version is the stale tail; the chunks go first, so a failed trim
        leaves the document known and the next ingest retries it (TASK-0443).
        """
        if self._trim is None:
            return

        await self._trim(task.document_id, 0)
        if self.journal is not None:
            await self.journal.forget_document(collection, task.document_id)

    async def _skip_empty(self, job: str | None, task: DocumentTask, outcome: ParseOutcome) -> None:
        """Record a document that parsed to nothing as skipped, with the reason."""
        _logger.info(
            "document skipped: nothing to index",
            extra={"document_id": task.document_id, "reason_code": EMPTY_DOCUMENT},
        )
        if self.journal is None or job is None:
            return

        record = replace(
            self._outcome(task, "skipped", outcome.content_hash),
            reason_code=EMPTY_DOCUMENT,
            detail=f"{task.source} parsed without error and produced no text to index",
        )
        await self.journal.record_document(job, record)

    async def _index_document(
        self, job: str | None, collection: str, task: DocumentTask, outcome: ParseOutcome
    ) -> None:
        """Record an indexed document and remember its content hash, in one thread hop.

        One hop rather than two: this runs once per document, and the handoff costs more
        than either append does (``docs/benchmarks.md`` BENCH-0006).
        """
        if self.journal is None or job is None:
            return

        await self.journal.record_indexed(
            job,
            collection,
            self._outcome(task, "indexed", outcome.content_hash),
            dedup=self.settings.ingestion.dedup,
        )

    def _outcome(
        self, task: DocumentTask, status: str, digest: str | None = None
    ) -> DocumentRecord:
        """Return the journal entry for one document's outcome."""
        return DocumentRecord(
            document_id=task.document_id,
            source=task.source,
            status=status,
            content_hash=digest,
            trace_id=current_trace_id(),
        )

    async def _enriched(self, outcome: ParseOutcome) -> ParseOutcome:
        """Prepend a situating context to each chunk, when the toggle and a model allow it."""
        if self.llm is None or not self.settings.chunking.contextual_enrichment:
            return outcome
        if not outcome.chunks or not outcome.document_text:
            return outcome

        from fasterrag.core.chunking.enrichment import enrich_chunks

        enriched = await enrich_chunks(
            [payload.chunk for payload in outcome.chunks],
            outcome.document_text,
            self.llm,
            self.settings,
        )
        return replace(
            outcome,
            chunks=[
                replace(payload, chunk=chunk)
                for payload, chunk in zip(outcome.chunks, enriched, strict=True)
            ],
        )

    @staticmethod
    def tasks_for(
        sources: Sequence[str],
        *,
        tenant: str | None = None,
        metadata: dict[str, object] | None = None,
        locations: Mapping[str, str] | None = None,
    ) -> list[DocumentTask]:
        """Build ordered document tasks with deterministic ids."""
        resolved = locations or {}
        return [
            DocumentTask(
                document_id=document_id(source, tenant),
                source=source,
                index=index,
                metadata=dict(metadata or {}),
                tenant=tenant,
                location=resolved.get(source),
            )
            for index, source in enumerate(sources)
        ]
