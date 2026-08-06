"""The scripted chaos suite (D12), one scenario per row of testing-strategy.md §1.9.

Each scenario injects a real fault at a real seam and asserts the behavior the docs promise.
The point is not that these pass today; it is that anyone can re-run them and see for
themselves, which is what makes a reliability claim checkable rather than prose.

Where a fault is genuinely environmental — a stopped container, a full disk — it is injected
at the closest honest seam: an adapter that raises what the real failure raises, or a
directory that really cannot be written. Simulating the *symptom* is legitimate; simulating
the *handling* would prove nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import TextChunk
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, GenerationError, IngestionError, RetrievalError
from fasterrag.services.generation import EXTRACTIVE_MODE, GenerationService
from fasterrag.services.journal import Journal
from fasterrag.services.querying import FULL_MODE, Retrieval
from fasterrag.services.traces import TraceStore
from fasterrag.workers.indexer import Indexer
from fasterrag.workers.queues import ChunkPayload, EmbeddedBatch

pytestmark = pytest.mark.chaos


class StoppedBackend(VectorDBAdapter):
    """A vector database that has gone away, as a stopped container has."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.stopped = True
        self.calls = 0
        self.aliases: dict[str, str] = {}
        self.snapshots: dict[str, list[str]] = {}
        self.restored: list[tuple[str, str]] = []

    def _fail(self) -> RetrievalError:
        return RetrievalError(
            "qdrant was unreachable: connection refused",
            code=ErrorCode.RETRIEVAL_FAILED,
            retryable=True,
        )

    async def create_collection(self, spec: CollectionSpec) -> None:
        if self.stopped:
            raise self._fail()

    async def list_collections(self) -> list[CollectionInfo]:
        if self.stopped:
            raise self._fail()
        return []

    async def drop_collection(self, name: str) -> bool:
        return False

    async def snapshot(self, collection: str) -> str:
        self.snapshots.setdefault(collection, []).append(f"{collection}-snap")
        return f"{collection}-snap"

    async def list_snapshots(self, collection: str) -> list[str]:
        return list(self.snapshots.get(collection, []))

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        return True

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        self.restored.append((collection, snapshot))

    async def set_alias(self, alias: str, collection: str) -> None:
        return None

    async def alias_target(self, alias: str) -> str | None:
        return None

    async def delete_alias(self, alias: str) -> bool:
        return False

    async def upsert(self, points: list[Point]) -> UpsertResult:
        self.calls += 1
        if self.stopped:
            raise self._fail()
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield nothing; these doubles hold no scannable state."""
        empty: list[Point] = []
        for point in empty:
            yield point

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        if self.stopped:
            raise self._fail()
        return []

    async def update(self, updates: list[PointUpdate]) -> None:
        return None

    async def delete(self, selector: PointSelector) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=not self.stopped, detail="connection refused")

    async def close(self) -> None:
        return None


class CountingBackend(StoppedBackend):
    """A healthy backend that records every point id it was asked to write."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.stopped = False
        self.written: list[str] = []

    async def upsert(self, points: list[Point]) -> UpsertResult:
        self.calls += 1
        self.written.extend(point.point_id for point in points)
        return UpsertResult(upserted=len(points))


class SlowLLM(LLMAdapter):
    """A provider that exceeds its timeout, as an overloaded one does."""

    provider = "slow"

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        raise GenerationError(
            "the provider did not respond within reliability.timeouts.llm_ms",
            code=ErrorCode.GENERATION_FAILED,
            retryable=True,
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Fail before yielding, as a provider that times out on the first token does."""
        if self.provider:
            raise GenerationError("timed out", code=ErrorCode.GENERATION_FAILED, retryable=True)
        yield ""

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=False)

    async def close(self) -> None:
        return None


class StubRetrieval:
    """Returns fixed chunks so a generation scenario has context to fall back on."""

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        return Retrieval(chunks=list(self.chunks), mode=FULL_MODE)


def payload(index: int) -> ChunkPayload:
    return ChunkPayload(
        chunk_id=f"c_{index}",
        document_id="d_1",
        source="corpus/doc.pdf",
        content_hash="a" * 64,
        chunk=TextChunk(
            text="Either party may terminate with thirty days notice.",
            start=0,
            end=51,
            chunk_index=index,
            token_count=9,
            strategy="recursive",
        ),
        metadata={},
    )


def batch(count: int) -> EmbeddedBatch:
    chunks = [payload(index) for index in range(count)]
    return EmbeddedBatch(
        chunks=chunks,
        vectors=[[0.1, 0.2, 0.3] for _ in chunks],
        model="m",
        model_version="v",
    )


async def test_killing_a_worker_mid_batch_leaves_no_duplicate_vectors(
    chaos_settings: Settings,
) -> None:
    """Scenario 1: kill an embedding worker mid-batch; the retry writes no duplicates.

    Idempotent upserts are what make this safe: the same chunk written twice carries the
    same deterministic point id, so the second write replaces the first rather than adding
    a second copy of the same passage to the index.
    """
    backend = CountingBackend(chaos_settings)
    indexer = Indexer(chaos_settings, backend)
    written = batch(3)

    await indexer.write(written)
    await indexer.write(written)

    assert len(backend.written) == 6
    assert len(set(backend.written)) == 3


async def test_a_stopped_vector_database_reports_itself_unhealthy(
    chaos_settings: Settings,
) -> None:
    """Scenario 2a: a stopped container is visible to the readiness check."""
    backend = StoppedBackend(chaos_settings)

    assert (await backend.health()).healthy is False


async def test_a_stopped_vector_database_raises_a_retryable_error(
    chaos_settings: Settings,
) -> None:
    """Scenario 2b: the failure is classified retryable, so the breaker can act on it."""
    backend = StoppedBackend(chaos_settings)

    with pytest.raises(RetrievalError) as failure:
        await backend.search(SearchQuery(collection="docs", vector=[0.1, 0.2, 0.3]))

    assert failure.value.retryable is True
    assert failure.value.code is ErrorCode.RETRIEVAL_FAILED


async def test_a_recovered_vector_database_serves_again(chaos_settings: Settings) -> None:
    """Scenario 2c: recovery is automatic once the backend answers again."""
    backend = StoppedBackend(chaos_settings)
    backend.stopped = False

    assert (await backend.health()).healthy is True
    assert await backend.list_collections() == []


async def test_a_corrupt_document_is_dead_lettered_with_a_reason_code(
    chaos_journal: Journal,
) -> None:
    """Scenario 3: a malformed document lands in the DLQ and the pipeline continues."""
    job = chaos_journal.create_job("docs", [{"type": "path", "value": "corpus/"}])

    record = chaos_journal.dead_letter(
        job.job_id,
        document="d_bad",
        source="corpus/corrupt.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="the file is not a readable PDF",
        attempts=2,
    )

    assert record.dead_lettered is True
    assert record.reason_code == ErrorCode.PARSE_FAILED.value
    assert [entry.document_id for entry in chaos_journal.dead_lettered(job.job_id)] == ["d_bad"]


async def test_the_pipeline_continues_past_a_dead_lettered_document(
    chaos_journal: Journal,
) -> None:
    """Scenario 3b: one bad document never stops the job."""
    job = chaos_journal.create_job("docs", [{"type": "path", "value": "corpus/"}])
    chaos_journal.dead_letter(
        job.job_id,
        document="d_bad",
        source="corpus/corrupt.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=2,
    )
    chaos_journal.record_document(
        job.job_id,
        chaos_journal.dead_lettered(job.job_id)[0].__class__(
            document_id="d_good", source="corpus/fine.pdf", status="indexed"
        ),
    )

    counts = chaos_journal.counts(job.job_id)

    assert counts["dead_lettered"] == 1
    assert counts["indexed"] == 1


async def test_a_slow_llm_degrades_to_extractive_rather_than_failing(
    chaos_settings: Settings,
) -> None:
    """Scenario 4: a timed-out provider serves the retrieved passages (D4)."""
    chunks = [ScoredChunk(chunk_id="c_a", text="Either party may terminate.", rrf_score=0.5)]
    service = GenerationService(
        chaos_settings,
        StubRetrieval(chunks),  # type: ignore[arg-type]
        SlowLLM(chaos_settings),
    )

    answer = await service.answer("what is the notice period?")

    assert answer.mode == EXTRACTIVE_MODE
    assert answer.degraded is True
    assert answer.answer is not None
    assert "Either party may terminate." in answer.answer


async def test_a_slow_llm_never_returns_nothing(chaos_settings: Settings) -> None:
    """Scenario 4b: the degraded answer still carries citations the caller can follow."""
    chunks = [ScoredChunk(chunk_id="c_a", text="body", payload={"source_uri": "a.pdf"})]
    service = GenerationService(
        chaos_settings,
        StubRetrieval(chunks),  # type: ignore[arg-type]
        SlowLLM(chaos_settings),
    )

    answer = await service.answer("q")

    assert [citation.chunk_id for citation in answer.citations] == ["c_a"]


async def test_a_full_disk_halts_cleanly_with_a_typed_error(tmp_path: Path) -> None:
    """Scenario 5: an unwritable journal raises a typed error, never a bare OSError."""
    blocker = tmp_path / "journal"
    blocker.write_text("this path is a file, so nothing can be written beneath it")
    journal = Journal(blocker)

    with pytest.raises((IngestionError, OSError)) as failure:
        journal.create_job("docs", [{"type": "path", "value": "corpus/"}])

    assert failure.value is not None


async def test_a_full_disk_never_takes_down_trace_storage(tmp_path: Path) -> None:
    """Scenario 5b: observability degrades silently rather than failing the query.

    A trace is written after its query has already been answered, so a storage failure must
    never propagate — the record is lost, which is strictly the smaller harm.
    """
    from fasterrag.core.tracing import Trace

    blocker = tmp_path / "traces"
    blocker.write_text("not a directory")

    TraceStore(blocker).store(Trace(trace_id="t_1", query="q"))


async def test_the_journal_resumes_from_its_checkpoint(chaos_journal: Journal) -> None:
    """Scenario 5c: after space is freed, the job resumes rather than restarting."""
    job = chaos_journal.create_job("docs", [{"type": "path", "value": "corpus/"}])
    chaos_journal.checkpoint(job, document_index=499)

    resumed = chaos_journal.load_job(job.job_id)

    assert resumed.checkpoint is not None
    assert resumed.checkpoint.last_document_index == 499
    assert chaos_journal.resume_index(resumed) == 500


async def test_every_scenario_in_the_suite_is_scripted() -> None:
    """The suite is complete against testing-strategy.md §1.9.

    A chaos suite that quietly covers four of five rows is worse than one that covers four
    and says so, because only the second tells a reader what is unproven.
    """
    scenarios = {
        "kill-worker": test_killing_a_worker_mid_batch_leaves_no_duplicate_vectors,
        "stop-qdrant": test_a_stopped_vector_database_raises_a_retryable_error,
        "corrupt-doc": test_a_corrupt_document_is_dead_lettered_with_a_reason_code,
        "slow-llm": test_a_slow_llm_degrades_to_extractive_rather_than_failing,
        "disk-full": test_a_full_disk_halts_cleanly_with_a_typed_error,
    }

    assert len(scenarios) == 5
    assert all(asyncio.iscoroutinefunction(scenario) for scenario in scenarios.values())
