"""Deduplication honours its switch within a job and holds across concurrent jobs (TASK-0471).

Two defects, both read from the code before they were run. The parse loop added every
indexed hash to its in-job map even with ``ingestion.dedup: false``, so the second of two
identical files in one job was reported ``deduplicated`` by a feature that was off. And the
map was loaded once per job, so two jobs running at once with the same content under
different names both missed it and both indexed it. The dedup record is now checked again
when a document commits, under the dedup file's lock.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import Point, UpsertResult
from fasterrag.config.schema import Settings
from fasterrag.core.identity import document_id
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from tests.unit.services.test_documents import MemoryAdapter
from tests.unit.services.test_ingestion import StubEmbedder, settings, threads

COLLECTION = "default"
TEXT = "# Leave\n\nParental leave is 26 weeks at full pay.\n"


class BarrierAdapter(MemoryAdapter):
    """Holds every upsert until ``parties`` of them have arrived, so two jobs overlap."""

    def __init__(self, configured: Settings, parties: int) -> None:
        super().__init__(configured)
        self.parties = parties
        self.arrived = 0
        self.released = asyncio.Event()

    async def upsert(self, points: list[Point]) -> UpsertResult:
        result = await super().upsert(points)
        self.arrived += 1
        if self.arrived >= self.parties:
            self.released.set()
        await asyncio.wait_for(self.released.wait(), timeout=10)
        return result


def service(configured: Settings, journal: Journal, adapter: MemoryAdapter) -> IngestionService:
    """Return an ingestion service over a shared journal and backend."""
    return IngestionService(
        configured,
        journal=journal,
        adapter=adapter,
        router=TieringRouter(StubEmbedder(configured)),
        executor_factory=threads,
    )


def twins(tmp_path: Path) -> tuple[str, str]:
    """Write the same content under two names and return both paths."""
    root = tmp_path / "corpus"
    root.mkdir()
    for name in ("leave.md", "leave-copy.md"):
        (root / name).write_text(TEXT, encoding="utf-8")
    return str(root / "leave.md"), str(root / "leave-copy.md")


def documents_held(adapter: MemoryAdapter) -> set[str]:
    """Return the document ids the backend still holds a chunk of."""
    return {str(point.payload["document_id"]) for point in adapter.held(COLLECTION)}


async def test_with_dedup_off_identical_documents_in_one_job_are_all_indexed(
    tmp_path: Path,
) -> None:
    """The in-job map was updated with dedup off, so the second copy was 'deduplicated'."""
    configured = settings(ingestion={"dedup": False})
    adapter = MemoryAdapter(configured)
    first, second = twins(tmp_path)
    ingest = service(configured, Journal(tmp_path / "journal"), adapter)

    settled = await ingest.run(await ingest.accept([first, second]))

    assert settled.counts["deduplicated"] == 0
    assert settled.counts["parsed"] == 2
    assert documents_held(adapter) == {document_id(first), document_id(second)}


async def test_two_concurrent_jobs_with_the_same_content_index_it_once(tmp_path: Path) -> None:
    """Each job loaded the known hashes before either committed, so both indexed the content."""
    configured = settings()
    adapter = BarrierAdapter(configured, parties=2)
    journal = Journal(tmp_path / "journal")
    first, second = twins(tmp_path)
    one, other = service(configured, journal, adapter), service(configured, journal, adapter)
    jobs = [await one.accept([first]), await other.accept([second])]

    settled = await asyncio.gather(one.run(jobs[0]), other.run(jobs[1]))

    assert sorted(record.counts["deduplicated"] for record in settled) == [0, 1]
    assert len(documents_held(adapter)) == 1
    assert len(await journal.known_content(COLLECTION)) == 1
    late = next(record for record in settled if record.counts["deduplicated"] == 1)
    assert [entry.status for entry in await journal.documents(late.job_id)] == ["deduplicated"]


async def test_two_concurrent_jobs_with_dedup_off_both_index(tmp_path: Path) -> None:
    configured = settings(ingestion={"dedup": False})
    adapter = BarrierAdapter(configured, parties=2)
    journal = Journal(tmp_path / "journal")
    first, second = twins(tmp_path)
    one, other = service(configured, journal, adapter), service(configured, journal, adapter)
    jobs = [await one.accept([first]), await other.accept([second])]

    settled = await asyncio.gather(one.run(jobs[0]), other.run(jobs[1]))

    assert [record.counts["deduplicated"] for record in settled] == [0, 0]
    assert documents_held(adapter) == {document_id(first), document_id(second)}
