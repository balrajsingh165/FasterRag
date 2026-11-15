"""A delete issued while an ingest job runs, and a dropped collection's records (TASK-0519).

A job loads its dedup map once, at the start, and remembers each document when its chunks
are all written. A delete in between used to interleave with that freely: chunks written
after the delete stayed in the index, or the job remembered a document whose chunks the
delete had just removed, so the listing named a document no query could find and the next
ingest reported it ``deduplicated``. These cases put the delete inside the job, against an
in-memory backend that really deletes, and look at what is left.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import Point, UpsertResult
from fasterrag.core.identity import document_id
from fasterrag.services.documents import DocumentService
from fasterrag.services.journal import DELETED_DURING_INGEST, DocumentRecord, Journal
from tests.unit.services.test_document_updates import (
    COLLECTION,
    LONG,
    OrderedAdapter,
    build,
)


class DeletingAdapter(OrderedAdapter):
    """Deletes one document from inside the job, before or after its first chunks land."""

    def __init__(self, inner: OrderedAdapter, journal: Journal, target: str, when: str) -> None:
        self.__dict__.update(inner.__dict__)
        self.service = DocumentService(self, journal)
        self.target = target
        self.when = when
        self.fired = False

    async def upsert(self, points: list[Point]) -> UpsertResult:
        hits = any(point.payload.get("document_id") == self.target for point in points)
        if hits and not self.fired and self.when == "before":
            self.fired = True
            await self.service.delete(COLLECTION, self.target)
        result = await super().upsert(points)
        if hits and not self.fired and self.when == "after":
            self.fired = True
            await self.service.delete(COLLECTION, self.target)
        return result


@pytest.fixture
def contract(tmp_path: Path) -> Path:
    """Return a document long enough to be several chunks."""
    path = tmp_path / "corpus" / "contract.md"
    path.parent.mkdir()
    path.write_text(LONG, encoding="utf-8")
    return path


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("known", [False, True])
async def test_a_document_deleted_while_its_job_runs_is_neither_indexed_nor_remembered(
    tmp_path: Path, contract: Path, when: str, known: bool
) -> None:
    """Chunks written after the delete survived it, or the job remembered a deleted document."""
    service, inner = build(tmp_path)
    target = document_id(str(contract))
    if known:
        await service.run(await service.accept([str(contract)]))
        contract.write_text(LONG + "\n\nA clause added later.\n", encoding="utf-8")
    adapter = DeletingAdapter(inner, service.journal, target, when)
    service.adapter = adapter

    record = await service.run(await service.accept([str(contract)]))

    assert adapter.fired
    assert adapter.chunks_of(COLLECTION, target) == []
    assert target not in await service.journal.document_hashes(COLLECTION)
    entries = await service.journal.documents(record.job_id)
    assert [(entry.status, entry.reason_code) for entry in entries] == [
        ("skipped", DELETED_DURING_INGEST)
    ]


async def test_a_document_deleted_while_its_job_runs_is_indexed_by_the_next_job(
    tmp_path: Path, contract: Path
) -> None:
    """The delete wins over the job that was running, not over every later one."""
    service, inner = build(tmp_path)
    target = document_id(str(contract))
    adapter = DeletingAdapter(inner, service.journal, target, "after")
    service.adapter = adapter
    await service.run(await service.accept([str(contract)]))

    record = await service.run(await service.accept([str(contract)]))

    assert record.counts["parsed"] == 1
    assert adapter.chunks_of(COLLECTION, target) != []
    assert target in await service.journal.document_hashes(COLLECTION)


def _indexed(document: str) -> DocumentRecord:
    """Return the journal entry a commit writes for one indexed document."""
    return DocumentRecord(document_id=document, source=f"{document}.md", status="indexed")


async def test_a_commit_waits_for_a_delete_in_progress_and_then_yields_to_it(
    tmp_path: Path,
) -> None:
    """A commit that landed between a delete's tombstone and its vector delete undid it."""
    journal = Journal(tmp_path / "journal")
    watch = await journal.watch_deletions(COLLECTION)
    started, release = asyncio.Event(), asyncio.Event()

    async def remove() -> None:
        started.set()
        await release.wait()

    deleting = asyncio.create_task(journal.delete_document(COLLECTION, "d_1", remove))
    await started.wait()
    committing = asyncio.create_task(
        journal.record_indexed("job", COLLECTION, _indexed("d_1"), dedup=True, watch=watch)
    )
    await asyncio.sleep(0.3)
    assert not committing.done()

    release.set()
    assert await deleting is False
    assert await committing is False
    assert await journal.document_hashes(COLLECTION) == {}


async def test_a_delete_finished_before_the_job_started_does_not_cancel_it(
    tmp_path: Path,
) -> None:
    async def remove() -> None:
        return None

    journal = Journal(tmp_path / "journal")
    await journal.delete_document(COLLECTION, "d_1", remove)
    watch = await journal.watch_deletions(COLLECTION)
    record = DocumentRecord(document_id="d_1", source="d_1.md", status="indexed", content_hash="h")

    assert await journal.record_indexed("job", COLLECTION, record, dedup=True, watch=watch)
    assert await journal.document_hashes(COLLECTION) == {"d_1": "h"}


async def test_a_backend_failure_inside_the_delete_releases_the_lock(tmp_path: Path) -> None:
    """The lock is held across the vector delete, so a raise there must not strand it."""
    journal = Journal(tmp_path / "journal")

    async def refuse() -> None:
        raise RuntimeError("backend unreachable")

    with pytest.raises(RuntimeError):
        await journal.delete_document(COLLECTION, "d_1", refuse)

    watch = await asyncio.wait_for(journal.watch_deletions(COLLECTION), timeout=5)
    assert watch.offset > 0


async def test_dropping_a_collection_forgets_its_dedup_record(
    tmp_path: Path, contract: Path
) -> None:
    """Re-creating a dropped collection and re-ingesting reported everything deduplicated."""
    service, adapter = build(tmp_path)
    await service.run(await service.accept([str(contract)]))
    documents = DocumentService(adapter, service.journal)

    assert await documents.drop_collection(COLLECTION) is True
    record = await service.run(await service.accept([str(contract)]))

    assert record.counts["deduplicated"] == 0
    assert record.counts["parsed"] == 1
    assert adapter.chunks_of(COLLECTION, document_id(str(contract))) != []


async def test_dropping_an_absent_collection_still_clears_stale_records(tmp_path: Path) -> None:
    """Records an earlier version left behind a dropped collection are cleared by dropping again."""
    service, adapter = build(tmp_path)
    service.journal.remember_content_blocking("gone", "hash-1", "d_1")

    await DocumentService(adapter, service.journal).drop_collection("gone")

    assert await service.journal.document_hashes("gone") == {}
    assert await service.journal.forget_collection("gone") is False
