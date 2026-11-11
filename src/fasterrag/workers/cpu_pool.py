"""CPU worker pool: load, parse, and chunk.

Loading, parsing, and chunking are CPU-bound, so they run in worker processes rather than
on the event loop (``docs/architecture.md`` §2). Chunks stream to the embedding pool as
each document finishes instead of after the whole corpus is parsed, which is what keeps
expensive embedding workers from idling behind the parser.

Failure is per-document by design. A corrupt file becomes a dead-letter entry with a reason
code and the pipeline continues — including a parser that raises something untyped, which is
converted to a ``ParseError`` inside the worker rather than aborting the job. A worker
*process* dying is different: the standard library marks the whole pool broken, so the job
fails rather than continuing on a pool that can no longer accept work.

Documents are parsed concurrently, up to the pool size at a time, and consumed in
submission order. Awaiting each document before submitting the next — which this module did
until 2026-11-11 — keeps exactly one worker busy however many the pool holds.

A document is journalled as indexed only when the :class:`IndexLedger` says every chunk of
it was written downstream (``workers/ledger.py``). With no ledger passed there is no
downstream to wait for, and the chunks are acknowledged as they are enqueued.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.chunking import create_chunker, create_token_counter
from fasterrag.core.identity import chunk_id, chunker_config_hash, content_hash, document_id
from fasterrag.core.parsing import create_parsing_options, parse_bytes
from fasterrag.errors import ErrorCode, FasterRagError, IngestionError, ParseError
from fasterrag.observability.logging import current_trace_id, get_logger
from fasterrag.services.journal import DocumentRecord, Journal
from fasterrag.workers.ledger import IndexLedger
from fasterrag.workers.queues import BoundedQueue, ChunkPayload, DocumentTask, ParseOutcome

__all__ = ["CpuWorkerPool", "PoolReport", "parse_and_chunk", "resolve_pool_size"]

_MEGABYTE = 1024 * 1024

_SLOW_TOKENIZER_LOAD_SECONDS = 0.5

_logger = get_logger(__name__)


def resolve_pool_size(configured: int) -> int:
    """Return the worker count, expanding the documented ``0`` to the CPU count."""
    if configured > 0:
        return configured
    return os.cpu_count() or 1


@dataclass(frozen=True, slots=True)
class PoolReport:
    """What a pass over a job's documents produced."""

    parsed: int = 0
    chunked: int = 0
    deduplicated: int = 0
    dead_lettered: int = 0
    skipped: int = 0
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

    chunker = create_chunker(settings)
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


class CpuWorkerPool:
    """Parses and chunks documents in worker processes, streaming chunks downstream."""

    def __init__(
        self,
        settings: Settings,
        *,
        journal: Journal | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
        llm: LLMAdapter | None = None,
    ) -> None:
        """Build the pool without starting any worker."""
        self.settings = settings
        self.journal = journal
        self.llm = llm
        self.size = resolve_pool_size(settings.workers.cpu_pool_size)
        self._executor_factory = executor_factory or (
            lambda workers: ProcessPoolExecutor(max_workers=workers)
        )
        self._executor: Executor | None = None

    async def __aenter__(self) -> CpuWorkerPool:
        """Start the worker processes and pay the tokenizer load up front."""
        self._executor = self._executor_factory(self.size)
        await self._warm_token_counter()
        _logger.info("cpu worker pool started", extra={"workers": self.size})
        return self

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
        """Shut the worker processes down."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
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
        standalone = ledger is None
        tracker = ledger or IndexLedger(on_progress, start=resume_from)
        parsed = chunked = deduplicated = dead_lettered = 0
        flags: dict[str, int] = {}

        ordered = list(tasks)
        pending = [task for task in ordered if task.index >= resume_from]
        skipped = len(ordered) - len(pending)

        stream = self._parsed(pending)
        try:
            async for task, result in self._parsed(pending):
                if isinstance(result, FasterRagError):
                    dead_lettered += 1
                    async with tracker.journal_lock:
                        await self._dead_letter(job, task, result)
                    await tracker.settle(task.index)
                    continue

                outcome = result
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
                    commit=self._commit(job, collection, task, outcome),
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
            flags=flags,
        )

    async def _parsed(
        self, tasks: Sequence[DocumentTask]
    ) -> AsyncGenerator[tuple[DocumentTask, ParseOutcome | FasterRagError]]:
        """Yield each task with its outcome or its typed failure, in submission order.

        Up to ``self.size`` documents are in the pool at once. Order is kept so the dedup
        decision between two identical documents in one job stays deterministic.
        """
        executor = self._executor
        loop = asyncio.get_running_loop()
        window: deque[tuple[DocumentTask, asyncio.Future[ParseOutcome]]] = deque()
        queued = iter(tasks)

        def fill() -> None:
            while len(window) < self.size:
                task = next(queued, None)
                if task is None:
                    return
                future = loop.run_in_executor(executor, parse_and_chunk, task, self.settings)
                window.append((task, future))

        try:
            fill()
            while window:
                task, future = window.popleft()
                try:
                    outcome = await future
                except BrokenProcessPool as exc:
                    raise _unguarded_entry_point_error() from exc
                except FasterRagError as exc:
                    yield task, exc
                else:
                    yield task, outcome
                fill()
        finally:
            for _, future in window:
                future.cancel()

    def _commit(
        self, job: str | None, collection: str, task: DocumentTask, outcome: ParseOutcome
    ) -> Callable[[], Awaitable[None]]:
        """Return the callback that journals this document once it is fully indexed."""

        async def commit() -> None:
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
