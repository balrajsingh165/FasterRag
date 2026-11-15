"""Stale chunks a re-ingest used to leave behind, in the three cases TASK-0443 did not cover.

The trim of a shrunken document needed the dedup record, so it did not run with
``ingestion.dedup: false``; it did not run for a document whose earlier ingest wrote some
chunks and then failed, because that document was never recorded; and it deleted by chunk
index, so chunks written under another chunker configuration — different ids at every index —
survived it. Each document's chunk extent (chunker and count) is now recorded whatever the
dedup switch says, before its chunks are written, and the old chunks are deleted by id.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import Point, UpsertResult
from fasterrag.config.schema import Settings
from fasterrag.core.identity import document_id
from fasterrag.errors import EmbedError
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Extent, Journal
from tests.unit.services.test_document_updates import LONG, SHORT, OrderedAdapter, build
from tests.unit.services.test_ingestion import settings

COLLECTION = "default"


@pytest.fixture
def contract(tmp_path: Path) -> Path:
    """Return a document long enough to be several chunks."""
    path = tmp_path / "corpus" / "contract.md"
    path.parent.mkdir()
    path.write_text(LONG, encoding="utf-8")
    return path


class FailingSecondBatch(OrderedAdapter):
    """Writes the first batch, then refuses every later one until told to stop."""

    def __init__(self, configured: Settings) -> None:
        super().__init__(configured)
        self.batches = 0
        self.refusing = True

    async def upsert(self, points: list[Point]) -> UpsertResult:
        self.batches += 1
        if self.refusing and self.batches > 1:
            raise EmbedError("backend refused the batch", retryable=False)
        return await super().upsert(points)


async def run(service: IngestionService, path: Path) -> dict[str, int]:
    """Ingest one file and return the settled job's counts, whatever its status."""
    record = await service.run(await service.accept([str(path)]))
    return record.counts


def assert_only_current(adapter: OrderedAdapter, document: str, chunked: int) -> None:
    """Assert the document holds exactly one chunk per index of its current version."""
    assert adapter.chunks_of(COLLECTION, document) == list(range(chunked))


async def test_with_dedup_off_a_shrunken_document_leaves_no_tail(
    tmp_path: Path, contract: Path
) -> None:
    """With dedup off the journal did not know the document, so its old tail stayed."""
    service, adapter = build(tmp_path, settings(ingestion={"dedup": False}))
    document = document_id(str(contract))
    await run(service, contract)

    contract.write_text(SHORT, encoding="utf-8")
    counts = await run(service, contract)

    assert_only_current(adapter, document, counts["chunked"])


async def test_chunks_left_by_an_ingest_that_failed_part_way_are_deleted(
    tmp_path: Path, contract: Path
) -> None:
    """The failed document was never recorded, so nothing knew its written chunks existed."""
    configured = settings()
    adapter = FailingSecondBatch(configured)
    service, _ = build(tmp_path, configured)
    service.adapter = adapter
    document = document_id(str(contract))
    first = await run(service, contract)
    assert first["dead_lettered"] == 1
    assert adapter.chunks_of(COLLECTION, document)

    adapter.refusing = False
    contract.write_text(SHORT, encoding="utf-8")
    counts = await run(service, contract)

    assert_only_current(adapter, document, counts["chunked"])


async def test_a_changed_chunker_leaves_none_of_the_old_chunks(
    tmp_path: Path, contract: Path
) -> None:
    """Old chunks had different ids at every index, so a trim by index missed all of them."""
    service, adapter = build(tmp_path, settings(chunking={"chunk_size": 128, "overlap": 16}))
    document = document_id(str(contract))
    await run(service, contract)

    rechunked, _ = build(tmp_path, settings(chunking={"chunk_size": 256, "overlap": 0}))
    rechunked.adapter = adapter
    contract.write_text(LONG.replace("obligation", "duty"), encoding="utf-8")
    counts = await run(rechunked, contract)

    assert_only_current(adapter, document, counts["chunked"])
    assert len({point.payload["content_hash"] for point in adapter.held(COLLECTION)}) == 1


async def test_with_dedup_off_a_changed_chunker_replaces_an_unchanged_document(
    tmp_path: Path, contract: Path
) -> None:
    off = {"dedup": False}
    service, adapter = build(tmp_path, settings(ingestion=off))
    document = document_id(str(contract))
    await run(service, contract)

    rechunked, _ = build(tmp_path, settings(ingestion=off, chunking={"chunk_size": 256}))
    rechunked.adapter = adapter
    counts = await run(rechunked, contract)

    assert_only_current(adapter, document, counts["chunked"])


async def test_re_ingesting_an_unchanged_document_with_dedup_off_deletes_nothing(
    tmp_path: Path, contract: Path
) -> None:
    """Its chunk ids are the same, so there is nothing stale and no delete to pay for."""
    service, adapter = build(tmp_path, settings(ingestion={"dedup": False}))
    await run(service, contract)
    await run(service, contract)

    assert adapter.selectors == []


def test_compacting_the_extent_log_keeps_what_it_means(tmp_path: Path) -> None:
    """Retention may shorten the log, never change which chunks a document may hold."""
    journal = Journal(tmp_path / "journal", retention_days=1)
    journal.record_extent_blocking(COLLECTION, "doc-a", "c1", 9, settled=False)
    journal.record_extent_blocking(COLLECTION, "doc-a", "c1", 9, settled=True)
    journal.record_extent_blocking(COLLECTION, "doc-a", "c2", 4, settled=False)
    journal.record_extent_blocking(COLLECTION, "doc-b", "c1", 3, settled=False)
    before = journal.chunk_extents_blocking(COLLECTION)

    report = journal.prune_blocking()

    assert report.dedup_lines == 1
    assert journal.chunk_extents_blocking(COLLECTION) == before
    assert before["doc-a"] == [Extent("c1", 9, True), Extent("c2", 4, False)]
